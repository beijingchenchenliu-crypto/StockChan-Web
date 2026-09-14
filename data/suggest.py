# -*- coding: utf-8 -*-
"""证券名称 / 拼音 → 代码 的智能联想解析。

用户在代码框里可以只写「粤传媒」「ycm」「贵州茅台」，由本模块向新浪 / 腾讯的
联想接口查询并解析出可请求的代码，避免把中文名直接送进行情接口后抛
``ValueError: 不是有效的 A 股代码``。

主接口（用户指定）::

    http://suggest3.sinajs.cn/suggest/type=11,12,13,14,15&key={关键词}

返回体形如::

    var suggestvalue="粤传媒,11,002181,sz002181,粤传媒,,粤传媒,99,1,,,";

字段依次为 ``名称, 类型, 代码, 带市场前缀代码, 显示名, ...``，多条以 ``;`` 分隔。

该接口的 ``type=11,12,13,14,15`` 只覆盖 A/B 股、权证、期货、债券，因此
ETF / 指数 / 港股 / 美股 会返回空 —— 命中为空时再补一次扩展类型查询
（追加基金、指数、港股、美股）。若两条都落空，最后用腾讯 ``smartbox``
兜底（它对**拼音首字母**的支持比新浪更宽），但它只用于 A 股 / 港股，
美股结果一律忽略（腾讯会返回 ``ndaq.oq`` / ``ixic`` 这类非交易代码）。
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

from .fetcher import (
    HK_INDEX_CODES,
    MARKET_CN,
    MARKET_HK,
    MARKET_US,
    _CN_CODE_RE,
    _US_CODE_RE,
    detect_market,
    guess_kind,
)

#: 用户指定的主查询类型：A股 / B股 / 权证 / 期货 / 债券
SUGGEST_TYPES_PRIMARY = "11,12,13,14,15"

#: 扩展类型：追加 开放式基金(21) 货币基金(22) QDII(23) 封闭式基金(24)
#: 板块与ETF(25) 指数(26) 港股(31) 中证/境外指数(33) 美股(41)
SUGGEST_TYPES_EXTENDED = "11,12,13,14,15,21,22,23,24,25,26,31,33,41"

_SINA_URL = "http://suggest3.sinajs.cn/suggest/type={types}&key={key}"
_TENCENT_URL = "https://smartbox.gtimg.cn/s3/?q={key}&t=all"

_SINA_REFERER = "https://finance.sina.com.cn"
_TENCENT_REFERER = "https://gu.qq.com/"

#: 联想接口超时（连接, 读取）。刻意比行情接口更短：联想是"打字即用"的
#: 交互，宁可快速失败也不要让界面卡住。
SUGGEST_TIMEOUT = (2.5, 3.5)

#: 新浪返回的"类型"字段 → 品种。``None`` 表示交给 :func:`guess_kind`
#: 按代码前缀判定（A 股里既有个股也有指数和 ETF，无法只看类型）。
_SINA_TYPE_KIND = {
    "11": None,       # A 股 / 指数 / 部分 ETF
    "12": "stock",    # B 股
    "13": "stock",    # 权证
    "14": None,       # 期货（代码非 6 位数字，会被过滤）
    "15": None,       # 债券（会被过滤）
    "21": "etf", "22": "etf", "23": "etf", "24": "etf",
    "25": None,       # 板块 / QDII 混杂，按代码判定
    "26": "index",
    "31": "hk",
    "33": "index",
    "41": "us",
}

#: 腾讯 ``v_hint`` 里需要跳过的品种：权证(QZ) / 指数(ZS) / 未分类(*)
_TENCENT_SKIP_TYPES = {"QZ", "ZS", "*", ""}


class Suggestion(NamedTuple):
    """一条联想结果。"""

    code: str          # 可直接请求的代码（002181 / 00700 / QQQ）
    name: str          # 完整名称（粤传媒）
    kind: str          # index / etf / stock / hk / hk_index / us
    market: str        # CN / HK / US
    source: str        # sina-primary / sina-extended / tencent

    @property
    def display(self) -> str:
        return f"{self.name}（{self.code}）"


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #

def looks_like_code(value: str) -> bool:
    """判断输入本身是否已是一个"像代码"的写法。

    覆盖 A 股 6 位数字、港股 5 位数字 / ``HK00700``、A 股带前缀
    ``sh600519``、港股指数 ``HSI``、美股 ``QQQ`` / ``BRK.B``。
    """
    s = str(value or "").strip().upper()
    if not s:
        return False
    if _CN_CODE_RE.match(s):
        return True
    if s.isdigit() and len(s) == 5:
        return True
    if s.startswith("HK") and s[2:].isdigit():
        return True
    if s.startswith(("SH", "SZ", "BJ")) and s[2:].isdigit():
        return True
    if s in HK_INDEX_CODES or (s.startswith("HS") and s[2:].isdigit()):
        return True
    return bool(_US_CODE_RE.match(s))


def needs_lookup(value: str) -> bool:
    """判断是否需要走在线联想。

    规则（与界面提示一致）：

    * 含中文字符 → 一定是名称，需要联想；
    * **全小写**字母 → 大概率是拼音（``ycm`` / ``gzmt``），需要联想；
    * 其余（6 位数字、大写美股代码、``00700``）→ 当作代码，直接请求。
    """
    s = str(value or "").strip()
    if not s:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return True
    if s.isalpha() and s.islower():
        return True
    return False


def _usable_code(code: str, market: str) -> bool:
    """过滤掉非交易代码（``.ixic`` / ``csi300`` / ``new_blhy`` …）。"""
    s = str(code or "").strip()
    if not s or s.startswith(".") or "_" in s:
        return False
    if market == MARKET_CN:
        return bool(_CN_CODE_RE.match(s))
    if market == MARKET_HK:
        return s.isdigit() and len(s) == 5
    if market == MARKET_US:
        return bool(_US_CODE_RE.match(s))
    return False


def _kind_for(code: str, market: str, hint: Optional[str]) -> str:
    """决定品种：优先用接口给的提示，否则按代码前缀推断。"""
    if hint:
        return hint
    return guess_kind(code)


def _http_get(url: str, referer: str, timeout=SUGGEST_TIMEOUT):
    """走数据层统一 Session（已关闭 trust_env、带浏览器 UA），失败再退回裸 requests。"""
    headers = {"Referer": referer}
    try:
        from . import net as _net
    except Exception:  # pragma: no cover - 网络层异常时仍要能用
        _net = None
    if _net is not None:
        try:
            return _net.get(url, channel=_net.DOMESTIC, timeout=timeout, headers=headers)
        except Exception:
            pass
    import requests

    return requests.get(url, timeout=timeout, headers=headers)


def _unescape(text: str) -> str:
    """腾讯接口把中文写成 ``\\u7ca4\\u4f20\\u5a92``，这里还原。"""
    if "\\u" not in text:
        return text
    try:
        return text.encode("latin-1", "backslashreplace").decode("unicode_escape")
    except Exception:
        return text


def _extract_payload(text: str) -> str:
    """取出 ``var suggestvalue="…";`` 里引号之间的内容。"""
    start = text.find('"')
    end = text.rfind('"')
    if start == -1 or end <= start:
        return ""
    return text[start + 1:end]


# --------------------------------------------------------------------------- #
# 各数据源解析
# --------------------------------------------------------------------------- #

def _parse_sina(text: str, source: str) -> List[Suggestion]:
    payload = _extract_payload(text or "")
    if not payload:
        return []

    results: List[Suggestion] = []
    seen: set = set()
    for chunk in payload.split(";"):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) < 3:
            continue
        name, type_code, code = parts[0], parts[1], parts[2]
        # parts[4] 是"显示名"，比 parts[0] 更可读：
        # ETF 的 parts[0] 是 ``of510300`` 这种带接口前缀的串，
        # 而 parts[4] 才是「沪深300ETF华泰柏瑞」。
        display_name = parts[4] if len(parts) > 4 and parts[4] else name
        if not code or code in seen:
            continue

        hint = _SINA_TYPE_KIND.get(type_code)
        if type_code in ("14", "15"):
            # 期货 / 债券：本项目不分析，直接跳过
            continue

        market = detect_market(code)
        if market == MARKET_CN and type_code in ("31",):
            market = MARKET_HK
        if not _usable_code(code, market):
            continue

        seen.add(code)
        results.append(Suggestion(
            code=code,
            name=display_name or code,
            kind=_kind_for(code, market, hint),
            market=market,
            source=source,
        ))
    return results


def _parse_tencent(text: str) -> List[Suggestion]:
    payload = _extract_payload(text or "")
    if not payload or payload.strip().upper() == "N":
        return []

    results: List[Suggestion] = []
    seen: set = set()
    for chunk in payload.split("^"):
        parts = [p.strip() for p in chunk.split("~")]
        if len(parts) < 4:
            continue
        market_tag, code, name, _pinyin = parts[0], parts[1], parts[2], parts[3]
        type_tag = parts[4].upper() if len(parts) > 4 else ""
        if type_tag in _TENCENT_SKIP_TYPES or not code or code in seen:
            continue

        market = {
            "sh": MARKET_CN, "sz": MARKET_CN, "bj": MARKET_CN,
            "hk": MARKET_HK,
        }.get(market_tag.lower())
        # 腾讯的美股结果（``us~ixic~…`` / ``us~ndaq.oq~…``）多为指数或
        # Yahoo 风格符号，不是可交易代码，一律忽略 —— 美股由新浪 41 类型覆盖。
        if market is None:
            continue
        if not _usable_code(code, market):
            continue

        hint = "hk" if market == MARKET_HK else None
        seen.add(code)
        results.append(Suggestion(
            code=code,
            name=_unescape(name) or code,
            kind=_kind_for(code, market, hint),
            market=market,
            source="tencent",
        ))
    return results


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #

def _fetch_sina(keyword: str, types: str, source: str) -> List[Suggestion]:
    from urllib.parse import quote

    url = _SINA_URL.format(types=types, key=quote(str(keyword).strip()))
    resp = _http_get(url, _SINA_REFERER)
    if resp is None or resp.status_code >= 400:
        return []
    # 接口返回 GBK 编码；requests 常猜成 ISO-8859-1，必须显式指定。
    resp.encoding = "gbk"
    return _parse_sina(resp.text, source)


def suggest(keyword: str, timeout_note: bool = True) -> List[Suggestion]:
    """查询联想结果（按"有效性优先"的原始顺序返回）。

    依次尝试：新浪指定类型 → 新浪扩展类型 → 腾讯兜底。任一环节命中即返回，
    避免为一个中文名连打三个接口。
    """
    value = str(keyword or "").strip()
    if not value:
        return []

    for types, source in ((SUGGEST_TYPES_PRIMARY, "sina-primary"),
                          (SUGGEST_TYPES_EXTENDED, "sina-extended")):
        try:
            hits = _fetch_sina(value, types, source)
        except Exception:
            hits = []
        if hits:
            return hits

    try:
        from urllib.parse import quote

        resp = _http_get(_TENCENT_URL.format(key=quote(value)), _TENCENT_REFERER)
        if resp is not None and resp.status_code < 400:
            resp.encoding = "gbk"
            return _parse_tencent(resp.text)
    except Exception:
        pass
    return []


def resolve_name(keyword: str) -> Optional[Suggestion]:
    """取"第 1 条有效匹配项"；没有可用结果时返回 ``None``。"""
    hits = suggest(keyword)
    return hits[0] if hits else None


__all__ = [
    "SUGGEST_TYPES_PRIMARY",
    "SUGGEST_TYPES_EXTENDED",
    "SUGGEST_TIMEOUT",
    "Suggestion",
    "looks_like_code",
    "needs_lookup",
    "suggest",
    "resolve_name",
]
