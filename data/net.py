"""网络层：双通道 Session 隔离 + 智能分流 + 防断连。

为什么需要这一层
----------------
akshare 内部大量直接使用 ``requests.get`` / ``requests.post``
（实测 1.18.94 有 1116 处 ``requests.get`` 调用），它**不接受外部传入 Session**。
在开了系统级 VPN / 全局代理的机器上，请求会被强行绕到境外出口，
访问东方财富、新浪这类国内源时经常直接抛::

    ConnectionError: ('Connection aborted.', RemoteDisconnected(...))

因此这里做两件事：

1. **准备两条长期存在的通道 Session**，作为网络配置的单一来源：

   - :data:`session_domestic` —— A 股 / 场内 ETF 用。
     ``trust_env = False`` 且 ``proxies = {}``，**彻底剥离系统 VPN / 全局代理**，
     直连国内数据源。
   - :data:`session_overseas` —— 港股 / 美股用。
     ``trust_env = True``，继承本地代理网络通道。

2. **把配置注入到 akshare 自己的请求里**。
   补丁打在唯一的汇聚点 ``requests.sessions.Session.request``：
   ``requests.get`` → ``requests.api.request`` → 新建 ``Session`` → ``Session.request``，
   ``requests.Session()`` 用法与 ``session.get`` 同样走这里。
   用 :func:`use_channel` 上下文管理器切换通道，内部靠 ``ContextVar``
   传递，因此对多线程（界面里的取数线程）安全。

用法::

    from data.net import DOMESTIC, OVERSEAS, install_patch, use_channel

    install_patch()                       # 幂等，首次取数时自动调用
    with use_channel(DOMESTIC):
        df = ak.stock_zh_a_hist(...)      # 走国内直连，无视 VPN
    with use_channel(OVERSEAS):
        df = ak.stock_us_daily(...)       # 继承本地代理
"""

from __future__ import annotations

import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 2.x
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover - urllib3 1.x
    from urllib3.util import Retry  # type: ignore


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

#: 国内直连通道（A 股 / 场内 ETF）
DOMESTIC = "domestic"
#: 境外代理通道（港股 / 美股）
OVERSEAS = "overseas"

#: 两条通道的取值
CHANNELS = (DOMESTIC, OVERSEAS)

#: 默认请求超时。
#: 使用 ``(connect_timeout, read_timeout)`` 元组：连接阶段 3.05s 无响应即放弃，
#: 读取阶段仍给 8s。这在「TCP 能连上但 HTTP/TLS 层挂死」的场景下（如某些
#: VPN 环境访问东方财富）能显著减少空转时间。
DEFAULT_TIMEOUT = (3.05, 8.0)

#: 最大自动重试次数
MAX_RETRIES: int = 3

#: 重试退避基数（秒）：0.4 / 0.8 / 1.6
BACKOFF_FACTOR: float = 0.4

#: 触发重试的 HTTP 状态码
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

#: 连接池大小
POOL_SIZE: int = 10

#: 标准浏览器请求头（国内源普遍会对非常规 UA 直接断连）
BROWSER_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

#: 通道中文名，用于界面/日志展示
CHANNEL_LABELS = {
    DOMESTIC: "国内直连",
    OVERSEAS: "境外代理",
}

#: 通道网络策略：``(是否继承系统代理, 是否强制清空显式代理)``
#:
#: - 国内直连：剥离系统代理，并清空调用方可能设置的代理 → 保证直连
#: - 境外代理：继承系统代理（VPN / 系统代理设置），但**保留**调用方
#:   显式设置的代理，不擅自丢弃用户配置
CHANNEL_POLICY = {
    DOMESTIC: (False, True),
    OVERSEAS: (True, False),
}

#: 库默认 UA 的识别标记。
#: ``requests.Session()`` 会自带 ``User-Agent: python-requests/x.y.z``，
#: 国内数据源对这类 UA 常常直接断连，因此**只有**这些默认值才允许被
#: 浏览器 UA 覆盖；调用方自己显式设置的 UA 一律尊重。
_LIBRARY_UA_MARKERS = ("python-requests", "python-urllib", "urllib3", "aiohttp")


def _is_library_ua(value: object) -> bool:
    low = str(value or "").lower()
    return (not low) or any(marker in low for marker in _LIBRARY_UA_MARKERS)


def apply_browser_headers(session: requests.Session) -> None:
    """把浏览器 Headers 注入 Session（就地修改）。

    规则：
    - 缺失的头部 → 直接补上；
    - ``User-Agent`` 若仍是库默认值（``python-requests/...``）→ 替换为浏览器 UA；
    - 调用方自己设过的头部 → 原样保留，绝不覆盖。
    """
    for key, value in BROWSER_HEADERS.items():
        current = session.headers.get(key)
        if current is None or (key.lower() == "user-agent" and _is_library_ua(current)):
            session.headers[key] = value


# --------------------------------------------------------------------------- #
# 重试策略
# --------------------------------------------------------------------------- #

def build_retry() -> Retry:
    """构造 ``max_retries=3`` 的重试策略。

    同时覆盖三类失败：
    - **连接失败**（``connect``）：DNS 失败、TCP 被拒、TLS 握手失败
    - **读取失败**（``read``）：``RemoteDisconnected`` / ``ProtocolError`` / 读超时
    - **状态码**（``status``）：429 / 5xx

    ``RemoteDisconnected`` 是 ``OSError`` 子类，urllib3 会把它包成
    ``ProtocolError``，属于 read 错误 —— 这正是 VPN 环境下最常见的症状，
    因此 ``read`` 必须显式设置，否则不会重试。
    """
    kwargs = dict(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        other=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_CODES,
        raise_on_status=False,   # 让上层拿到响应自行判断，避免抛 RetryError
    )
    try:  # urllib3 >= 1.26
        return Retry(allowed_methods=frozenset(["GET", "POST", "HEAD", "OPTIONS"]), **kwargs)
    except TypeError:  # pragma: no cover - 老版本 urllib3
        return Retry(method_whitelist=frozenset(["GET", "POST", "HEAD", "OPTIONS"]), **kwargs)


def build_adapter() -> HTTPAdapter:
    """构造挂载了重试策略的 ``HTTPAdapter``。"""
    return HTTPAdapter(
        max_retries=build_retry(),
        pool_connections=POOL_SIZE,
        pool_maxsize=POOL_SIZE,
    )


# --------------------------------------------------------------------------- #
# 双通道 Session
# --------------------------------------------------------------------------- #

def build_session(trust_env: bool) -> requests.Session:
    """按通道语义构造一个 Session。

    :param trust_env: ``False`` → 剥离系统代理（国内直连）；
                      ``True``  → 继承本地代理（境外通道）
    """
    session = requests.Session()
    apply_browser_headers(session)
    session.trust_env = bool(trust_env)
    if not trust_env:
        # 双保险：trust_env=False 已能屏蔽环境代理，这里再显式清空
        session.proxies = {}
    session.mount("http://", build_adapter())
    session.mount("https://", build_adapter())
    return session


#: 国内直连 Session —— 强制剥离本地系统 VPN / 全局代理
session_domestic: requests.Session = build_session(trust_env=False)

#: 境外代理 Session —— 继承本地代理网络通道
session_overseas: requests.Session = build_session(trust_env=True)


def session_for(channel: str) -> requests.Session:
    """按通道名取 Session。"""
    return session_domestic if channel == DOMESTIC else session_overseas


def channel_label(channel: str) -> str:
    """通道中文名。"""
    return CHANNEL_LABELS.get(channel, channel or "未指定")


# --------------------------------------------------------------------------- #
# 通道切换（对 akshare 生效）
# --------------------------------------------------------------------------- #

_current: ContextVar[str] = ContextVar("stockchan_channel", default="")


def current_channel() -> str:
    """当前生效的通道；未设置时返回空串。"""
    return _current.get()


@contextmanager
def use_channel(channel: str) -> Iterator[None]:
    """在 ``with`` 块内把 akshare 的所有 HTTP 请求切到指定通道。

    用 ``ContextVar`` 而非全局变量，因此在界面里的取数线程中同样安全，
    且嵌套使用时退出后能精确还原。
    """
    token = _current.set(channel or "")
    try:
        yield
    finally:
        _current.reset(token)


_PATCHED_FLAG = "_stockchan_patched"
_ORIG_REQUEST = None


def install_patch() -> bool:
    """让 akshare 内部的 ``requests`` 调用也遵守通道配置。幂等。

    :return: 本次调用是否真的装上了补丁
    """
    global _ORIG_REQUEST

    cls = requests.sessions.Session
    if getattr(cls, _PATCHED_FLAG, False):
        return False

    _ORIG_REQUEST = cls.request

    def _patched_request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        channel = _current.get()
        if not channel:
            # 未指定通道 → 完全透传，绝不影响宿主环境里其他 requests 用法
            return _ORIG_REQUEST(self, method, url, **kwargs)

        saved_trust = self.trust_env
        saved_proxies = self.proxies
        saved_headers = dict(self.headers)
        try:
            # 1) 代理策略：国内直连剥离代理，境外继承系统代理
            trust_env, clear_proxies = CHANNEL_POLICY[channel]
            self.trust_env = trust_env
            if clear_proxies:
                self.proxies = {}
            # 2) 浏览器头：补缺失项，并把库默认 UA 换成浏览器 UA
            apply_browser_headers(self)
            # 3) 防断连：短生命周期的 Session 也要挂上重试适配器
            ensure_adapter(self)
            # 4) 统一超时
            kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
            return _ORIG_REQUEST(self, method, url, **kwargs)
        finally:
            self.trust_env = saved_trust
            self.proxies = saved_proxies
            self.headers.clear()
            self.headers.update(saved_headers)

    cls.request = _patched_request
    setattr(cls, _PATCHED_FLAG, True)
    return True


_ADAPTER_FLAG = "_stockchan_adapter"


def ensure_adapter(session: requests.Session) -> None:
    """给 Session 挂上带重试的适配器（幂等）。

    ``requests.get()`` 每次都会新建 Session，所以不能只在模块级 Session 上挂，
    必须在补丁里按需挂载。
    """
    if getattr(session, _ADAPTER_FLAG, False):
        return
    session.mount("http://", build_adapter())
    session.mount("https://", build_adapter())
    setattr(session, _ADAPTER_FLAG, True)


def patch_installed() -> bool:
    """补丁是否已安装。"""
    return bool(getattr(requests.sessions.Session, _PATCHED_FLAG, False))


# --------------------------------------------------------------------------- #
# 直连辅助（不经过 akshare 的自定义请求）
# --------------------------------------------------------------------------- #

def get(url: str, *, channel: str = DOMESTIC, timeout = DEFAULT_TIMEOUT,
        **kwargs) -> requests.Response:
    """用指定通道的 Session 发 GET。"""
    kwargs.setdefault("timeout", timeout)
    return session_for(channel).get(url, **kwargs)


def post(url: str, *, channel: str = DOMESTIC, timeout = DEFAULT_TIMEOUT,
         **kwargs) -> requests.Response:
    """用指定通道的 Session 发 POST。"""
    kwargs.setdefault("timeout", timeout)
    return session_for(channel).post(url, **kwargs)


#: 用于连通性探测的轻量国内 / 境外地址
PROBE_URLS = {
    DOMESTIC: "https://www.baidu.com",
    OVERSEAS: "https://www.google.com",
}


def probe_channel(channel: str, url: Optional[str] = None,
                  timeout: float = 4.0) -> bool:
    """探测某条通道是否可用（仅用于诊断展示，失败不影响取数）。"""
    target = url or PROBE_URLS.get(channel)
    if not target:
        return False
    try:
        resp = get(target, channel=channel, timeout=timeout, stream=True)
        return resp.status_code < 500
    except Exception:  # noqa: BLE001 - 诊断用途，吞掉所有异常
        return False


def probe_channels() -> Dict[str, bool]:
    """同时探测两条通道，返回 ``{通道: 是否可用}``。"""
    return {name: probe_channel(name) for name in CHANNELS}


def probe_report(timeout: float = 4.0) -> List[Dict[str, object]]:
    """逐通道探测并返回结构化诊断结果（供 ``main.py --probe`` 使用）。

    每项包含 ``channel / label / url / ok / status / elapsed / error``，
    便于在开了 VPN 的机器上快速定位到底是哪条通道断了。
    """
    report: List[Dict[str, object]] = []
    for name in CHANNELS:
        url = PROBE_URLS.get(name, "")
        item: Dict[str, object] = {
            "channel": name,
            "label": CHANNEL_LABELS.get(name, name),
            "url": url,
            "ok": False,
            "status": None,
            "elapsed": 0.0,
            "error": "",
        }
        started = time.perf_counter()
        try:
            resp = get(url, channel=name, timeout=timeout, stream=True)
            item["status"] = resp.status_code
            item["ok"] = resp.status_code < 500
            resp.close()
        except Exception as exc:  # noqa: BLE001 - 诊断用途
            item["error"] = f"{type(exc).__name__}: {exc}"
        item["elapsed"] = round(time.perf_counter() - started, 2)
        report.append(item)
    return report


def describe() -> str:
    """给出一条人类可读的通道配置摘要。"""
    def _state(sess: requests.Session) -> str:
        proxy = "继承系统代理" if sess.trust_env else "已剥离系统代理"
        return f"trust_env={sess.trust_env}（{proxy}）"

    return (f"国内直连 {_state(session_domestic)} / "
            f"境外代理 {_state(session_overseas)} | "
            f"timeout={DEFAULT_TIMEOUT!r} 重试={MAX_RETRIES} 次")


def system_proxies() -> Dict[str, str]:
    """读取当前进程可见的系统代理（VPN / 全局代理环境诊断用）。

    国内直连通道会用 ``trust_env=False`` + ``proxies={}`` 无视这里的值。
    """
    try:
        return dict(requests.utils.getproxies())
    except Exception:  # noqa: BLE001 - 诊断用途
        return {}


# --------------------------------------------------------------------------- #
# 数据源可达性预探测
# --------------------------------------------------------------------------- #
#
# 用途：把"整片域名都连不上"的数据源提前降级，避免每个接口都空转
# ``timeout × (重试次数+1)`` 秒。以本机实测为例，东方财富全系不可达时，
# 上证指数要先后试两个东财接口，每个约 47s —— 一次取数白等 90s 以上。
# 先做一次 TCP 预探测（2.5s 上限）就能把这段时间省掉。
#
# 注意：探测结果只用于**降级排序**，不会删除任何接口 ——
# 万一探测误判，接口仍会在其他候选都失败后被执行。

#: 数据源家族 -> 代表主机（用于可达性预探测）
SOURCE_HOSTS = {
    "eastmoney": "push2his.eastmoney.com",
    "sina": "finance.sina.com.cn",
}

#: 预探测超时（秒）—— 只判断 TCP 能否建连，不做完整 HTTP 请求
HOST_PROBE_TIMEOUT = 2.5
#: 预探测结果缓存时长（秒）
HOST_TTL = 300.0

#: 主机 -> (是否可达, 探测时刻)
_HOST_CACHE: Dict[str, Tuple[bool, float]] = {}


def host_reachable(host: str, port: int = 443,
                   timeout: float = HOST_PROBE_TIMEOUT,
                   ttl: float = HOST_TTL) -> bool:
    """短超时 TCP 探测主机是否可达（结果缓存 ``ttl`` 秒）。"""
    if not host:
        return True
    now = time.monotonic()
    cached = _HOST_CACHE.get(host)
    if cached and now - cached[1] < ttl:
        return cached[0]

    ok = False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            ok = True
    except OSError:
        ok = False
    _HOST_CACHE[host] = (ok, now)
    return ok


def source_reachable(source: str) -> bool:
    """某数据源家族是否可达（未知数据源一律视为可达）。"""
    host = SOURCE_HOSTS.get(source)
    return True if not host else host_reachable(host)


def reset_host_cache() -> None:
    """清空可达性缓存（自检与手动重试用）。"""
    _HOST_CACHE.clear()
