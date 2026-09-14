# -*- coding: utf-8 -*-
import os
import re
import time
import requests
from collections.abc import Mapping
from requests.adapters import HTTPAdapter
from datetime import datetime, timedelta
import pandas as pd
import akshare as ak

# ==================== 0. 强力脱离本地代理 ====================
for proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_key, None)

_original_session_init = requests.Session.__init__
def _patched_session_init(self, *args, **kwargs):
    _original_session_init(self, *args, **kwargs)
    self.trust_env = False
    self.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
requests.Session.__init__ = _patched_session_init

try:
    from .net import session_domestic, session_overseas, DEFAULT_TIMEOUT
except ImportError:
    pass

# ==================== 1. 系统核心常量（必须在所有函数之前定义） ====================
COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]
COOLDOWN = 0.5
DEFAULT_START = "2023-09-13"

MARKET_CN = "CN"
MARKET_HK = "HK"
MARKET_US = "US"

MARKET_LABELS = {
    MARKET_CN: "A股/国内",
    MARKET_HK: "港股",
    MARKET_US: "美股"
}

KINDS = {
    "指数": "index",
    "ETF": "etf",
    "股票": "stock"
}

_STRATEGY_MEMORY = {}
_INTERFACE_HEALTH = {"status": "ok", "last_check": None}


def _as_frame(raw) -> pd.DataFrame:
    """将行情接口的常见返回形态安全转换为二维表。

    部分分钟线接口在无数据、限流或只返回一条记录时会给出
    ``{"时间": ..., "开盘": ...}`` 这样的标量字典。直接调用
    ``pd.DataFrame(raw)`` 会触发 ``If using all scalar values, you must
    pass an index``，因此单条标量记录必须显式包成一行。
    """
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, pd.Series):
        return raw.to_frame().T.reset_index(drop=True)
    if isinstance(raw, Mapping):
        if not raw:
            return pd.DataFrame()
        values = list(raw.values())
        is_scalar_record = all(
            not pd.api.types.is_list_like(value)
            or isinstance(value, (str, bytes))
            for value in values
        )
        if is_scalar_record:
            return pd.DataFrame({key: [value] for key, value in raw.items()})
    try:
        return pd.DataFrame(raw)
    except ValueError as exc:
        if "all scalar values" not in str(exc):
            raise
        return pd.DataFrame([raw])

def reset_strategy_memory():
    global _STRATEGY_MEMORY
    _STRATEGY_MEMORY.clear()

def strategy_memory():
    return _STRATEGY_MEMORY

def interface_health():
    _INTERFACE_HEALTH["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return _INTERFACE_HEALTH

def last_route(df: pd.DataFrame = None) -> dict:
    if df is not None and hasattr(df, "attrs") and "route" in df.attrs:
        return df.attrs["route"]
    return {"market": MARKET_CN, "channel": "domestic", "source": "direct"}

# ==================== 2. 标的与市场识别 ====================

#: A 股 / 场内基金代码形如 6 位数字（000001、510300、159915…）
_CN_CODE_RE = re.compile(r"^\d{6}$")


#: 美股代码：1~5 位字母，允许 ``BRK.B`` / ``BRK-B`` 这类带分隔符的写法
_US_CODE_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def detect_market(symbol: str) -> str:
    """按代码形态判断所属市场。

    判定顺序很重要：港股指数（``HSI`` / ``HS2083``）与 A 股指数一样是纯字母
    或字母+数字，若不先拦下就会被 ``^[A-Z]{1,5}$`` 误判成美股。
    """
    s = str(symbol).strip().upper()
    # 港股指数：HSI / HSTECH / HS2083 …
    if s in HK_INDEX_CODES or (s.startswith("HS") and s[2:].isdigit()):
        return MARKET_HK
    # 港股：显式 HK 前缀，或 5 位数字（00700）
    if s.startswith("HK") and s[2:].isdigit():
        return MARKET_HK
    if s.isdigit() and len(s) == 5:
        return MARKET_HK
    # 美股
    if _US_CODE_RE.match(s):
        return MARKET_US
    return MARKET_CN

#: 港股指数代码（恒生指数 / 国企指数 / 恒生科技 / 恒生沪深港通 AH）
HK_INDEX_CODES = {"HSI", "HSCEI", "HSTECH", "HSAHP"}

#: A 股指数代码
_CN_INDEX_CODES = {"000001", "399001", "399006", "000300", "000905",
                   "000852", "000688", "H30021"}


def guess_kind(symbol: str) -> str:
    """猜测标的类型。

    注意识别顺序：先判港股指数（``HSI`` 这类纯字母代码否则会被
    ``detect_market`` 当成美股），再判 A 股指数，最后按市场归类。
    """
    s = str(symbol).strip().upper()
    if s in HK_INDEX_CODES or (s.startswith("HS") and s[2:].isdigit()):
        return "hk_index"
    if s in _CN_INDEX_CODES:
        return "index"

    market = detect_market(s)
    if market == MARKET_US:
        return "us"
    if market == MARKET_HK:
        return "hk"

    if s.startswith(("51", "56", "58", "15", "16", "50")):
        return "etf"
    return "stock"

def market_of_kind(kind: str) -> str:
    return MARKET_CN

def default_symbol(kind: str = "index") -> str:
    mapping = {
        "index": "000001", "指数": "000001",
        "etf": "510300", "ETF": "510300",
        "stock": "600519", "股票": "600519"
    }
    return mapping.get(str(kind).lower(), "000001")

def normalize_symbol(symbol: str, kind: str = None) -> str:
    """把用户可能带交易所前缀的写法归一成裸代码。

    ``HK0700`` → ``00700``（补足 5 位）、``sh600519`` → ``600519``。
    """
    s = str(symbol).strip().upper()
    if s.startswith("HK") and s[2:].isdigit():
        return s[2:].zfill(5)
    if s.startswith(("SH", "SZ", "BJ")) and s[2:].isdigit():
        return s[2:]
    return s

# ==================== 3. 稳健数据规整 ====================
def _clean_df(df: pd.DataFrame, market: str, channel: str, source: str) -> pd.DataFrame:
    df = _as_frame(df)
    if df.empty:
        raise ValueError("拉取到的行情数据为空")

    if isinstance(df.index, pd.DatetimeIndex) or "date" in str(df.index.name).lower():
        df = df.reset_index()

    date_col = None
    for c in df.columns:
        c_str = str(c).lower().strip()
        if c_str in ["date", "datetime", "day", "日期", "时间", "trade_date"]:
            date_col = c
            break

    if date_col is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c].iloc[0])
                date_col = c
                break
            except Exception:
                continue

    if date_col is None:
        raise KeyError("无法定位日期列")

    df["date"] = pd.to_datetime(df[date_col])

    col_mapping = {
        "开盘": "open", "open": "open",
        "最高": "high", "high": "high",
        "最低": "low", "low": "low",
        "收盘": "close", "close": "close",
        "成交量": "volume", "volume": "volume", "vol": "volume",
        "成交额": "amount", "amount": "amount", "amt": "amount"
    }
    for orig, target in col_mapping.items():
        for c in df.columns:
            if str(c).lower().strip() == orig:
                df[target] = df[c]

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise KeyError(f"缺少必要列: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "amount" not in df.columns:
        df["amount"] = 0.0
    else:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df = df.sort_values("date").reset_index(drop=True)
    res = df[COLUMNS].copy()
    res.attrs["route"] = {"market": market, "channel": channel, "source": source}
    return res

# ==================== 4. 抓取与多通道降级 ====================
def fetch_index_daily(symbol: str, start_date: str = DEFAULT_START) -> pd.DataFrame:
    s_date = str(start_date).replace("-", "")
    e_date = datetime.now().strftime("%Y%m%d")
    try:
        df = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=s_date, end_date=e_date)
        return _clean_df(df, MARKET_CN, "domestic", "akshare.index_zh_a_hist")
    except Exception:
        sym = f"sh{symbol}" if symbol.startswith("0") else f"sz{symbol}"
        df = ak.stock_zh_index_daily(symbol=sym)
        return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_index_daily")

def fetch_etf_daily(symbol: str, start_date: str = DEFAULT_START) -> pd.DataFrame:
    s_date = str(start_date).replace("-", "")
    e_date = datetime.now().strftime("%Y%m%d")
    
    errors = []
    
    # 策略 1: 尝试 fund_etf_hist_em，带重试
    for attempt in range(2):
        try:
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=s_date, end_date=e_date)
            if df is not None and not df.empty:
                return _clean_df(df, MARKET_CN, "domestic", "akshare.fund_etf_hist_em")
        except Exception as e:
            if attempt == 1:
                errors.append(f"fund_etf_hist_em: {e}")
            time.sleep(0.5)

    # 策略 2: 尝试新浪场内基金接口（已修复日期比较类型错误）
    try:
        market_prefix = "sh" if symbol.startswith(("51", "56", "58", "50")) else "sz"
        full_sym = f"{market_prefix}{symbol}"
        df = ak.fund_etf_hist_sina(symbol=full_sym)
        if df is not None and not df.empty:
            if "date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(datetime.now())
                df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
            return _clean_df(df, MARKET_CN, "domestic", "akshare.fund_etf_hist_sina")
    except Exception as e:
        errors.append(f"fund_etf_hist_sina: {e}")

    # 策略 3: 回退使用 stock_zh_a_hist
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=s_date, end_date=e_date, adjust="qfq")
        if df is not None and not df.empty:
            return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_hist.fallback")
    except Exception as e:
        errors.append(f"stock_zh_a_hist_fallback: {e}")

    raise ValueError(f"ETF {symbol} 数据获取失败: {'; '.join(errors)}")

def fetch_stock_daily(symbol: str, start_date: str = DEFAULT_START) -> pd.DataFrame:
    s_date = str(start_date).replace("-", "")
    e_date = datetime.now().strftime("%Y%m%d")
    errors = []
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=s_date, end_date=e_date, adjust="qfq")
        if df is not None and not df.empty:
            return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_hist")
    except Exception as exc:
        errors.append(f"stock_zh_a_hist: {type(exc).__name__}")

    # 新浪兜底。注意：代码非法时新浪会返回没有 date 列的报错体，
    # akshare 内部直接 ``data_df["date"]`` → 抛出裸 ``KeyError: 'date'``，
    # 用户看到的是一个毫无信息量的报错。这里统一转成可读消息。
    try:
        sym = f"sh{symbol}" if str(symbol).startswith("6") else f"sz{symbol}"
        df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
        if df is not None and not df.empty:
            return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_daily")
    except KeyError as exc:
        errors.append(f"新浪接口未返回该代码的数据（{exc}）")
    except Exception as exc:
        errors.append(f"stock_zh_a_daily: {type(exc).__name__}: {exc}")

    raise ValueError(
        f"A 股代码「{symbol}」没有取到数据，请确认代码是否正确。"
        f"（{'；'.join(errors)}）"
    )

def _trim_start(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """按起始日期裁剪（新浪的美股/港股接口不支持传日期区间，只能本地裁）。

    注意：新浪的 ``stock_hk_daily`` 返回的 ``date`` 列是 ``datetime.date`` 对象，
    直接与 ``Timestamp`` 比较会抛
    ``TypeError: Cannot compare Timestamp with datetime.date``，
    因此必须先统一 ``to_datetime``。
    """
    if df is None or df.empty or "date" not in df.columns:
        return df
    try:
        start_dt = pd.to_datetime(str(start_date))
        dates = pd.to_datetime(df["date"], errors="coerce")
    except Exception:
        return df
    return df[dates >= start_dt].reset_index(drop=True)


def fetch_hk_daily(symbol: str, start_date: str = DEFAULT_START) -> pd.DataFrame:
    """港股日线。

    本机实测东方财富行情主机（``push2his.eastmoney.com``）在部分网络环境下
    **完全不可达**，而新浪的 ``stock_hk_daily`` 稳定可用且直接接受 5 位港股代码，
    因此把它作为首选，东财接口降级为兜底。
    """
    sym = str(symbol).strip().upper().replace("HK", "")
    sym = sym.zfill(5)  # 港股代码统一补足 5 位，如 700 -> 00700
    errors = []

    # 策略 1：新浪（无需 secid，直接吃 00700）
    try:
        df = ak.stock_hk_daily(symbol=sym, adjust="")
        if df is not None and not df.empty:
            return _clean_df(_trim_start(df, start_date), MARKET_HK, "overseas",
                             "akshare.stock_hk_daily")
    except Exception as exc:
        errors.append(f"stock_hk_daily: {type(exc).__name__}: {exc}")

    # 策略 2：东方财富
    try:
        df = ak.stock_hk_hist(symbol=sym, period="daily",
                              start_date=str(start_date).replace("-", ""),
                              end_date=datetime.now().strftime("%Y%m%d"))
        if df is not None and not df.empty:
            return _clean_df(df, MARKET_HK, "overseas", "akshare.stock_hk_hist")
    except Exception as exc:
        errors.append(f"stock_hk_hist: {type(exc).__name__}: {exc}")

    raise ValueError(f"港股 {sym} 行情获取失败：{'；'.join(errors)}")


#: 东方财富美股 secid 前缀：105=纳斯达克 106=纽交所 107=美交所
_US_SECID_PREFIXES = ("105", "106", "107")


def fetch_us_daily(symbol: str, start_date: str = DEFAULT_START) -> pd.DataFrame:
    """美股日线。

    同样优先新浪 ``stock_us_daily``（直接接受 ``QQQ`` 这类裸代码，实测可用）；
    东方财富的 ``stock_us_hist`` 要求 ``105.QQQ`` 形式的 secid，作为兜底逐个前缀试探。
    """
    sym = str(symbol).strip().upper()
    errors = []

    # 策略 1：新浪（裸代码即可）
    try:
        df = ak.stock_us_daily(symbol=sym)
        if df is not None and not df.empty:
            return _clean_df(_trim_start(df, start_date), MARKET_US, "overseas",
                             "akshare.stock_us_daily")
    except Exception as exc:
        errors.append(f"stock_us_daily: {type(exc).__name__}: {exc}")

    # 策略 2：东方财富（需要 secid 前缀，逐个试探）
    for prefix in _US_SECID_PREFIXES:
        try:
            df = ak.stock_us_hist(symbol=f"{prefix}.{sym}", period="daily",
                                  start_date=str(start_date).replace("-", ""),
                                  end_date=datetime.now().strftime("%Y%m%d"))
            if df is not None and not df.empty:
                return _clean_df(df, MARKET_US, "overseas",
                                 f"akshare.stock_us_hist.{prefix}")
        except Exception as exc:
            errors.append(f"stock_us_hist({prefix}): {type(exc).__name__}")

    raise ValueError(f"美股 {sym} 行情获取失败：{'；'.join(errors)}")

def _sina_prefixed(symbol: str) -> str:
    """给新浪接口补上交易所前缀（新浪只认 ``sh600519`` / ``sz000001`` 这类写法）。

    旧实现直接把裸代码 ``000001`` 传给 ``stock_zh_a_minute``，且把周期写成
    ``"5m"``，两处都会失败，导致分钟线在本机完全不可用。
    """
    s = str(symbol).strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        return s
    # 指数：上证 / 沪深系列以 000 开头 → sh；深证系列 399 开头 → sz
    if s.startswith("399"):
        return f"sz{s}"
    if s.startswith("000"):
        return f"sh{s}"
    # 北交所：43/83/87/920 开头（新浪写作 bj）
    if s.startswith(("43", "83", "87", "92")):
        return f"bj{s}"
    if s.startswith(("5", "6", "9")):
        return f"sh{s}"
    return f"sz{s}"


def fetch_index_min(symbol: str, period: str = "5", start_date: str = None) -> pd.DataFrame:
    p_val = str(period).replace("分", "").replace("m", "").strip()
    calc_start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    calc_end = datetime.now().strftime("%Y%m%d")
    errors = []
    try:
        df = ak.index_zh_a_hist_min_em(symbol=symbol, period=p_val, start_date=calc_start, end_date=calc_end)
        return _clean_df(df, MARKET_CN, "domestic", "akshare.index_zh_a_hist_min_em")
    except Exception as exc:
        errors.append(f"东方财富分钟接口: {type(exc).__name__}")

    # 新浪回退接口在无网络/限流时可能收到标量 JSON，AkShare 内部会先
    # 调用 pd.DataFrame(data_json) 并抛出 scalar-values ValueError。这里
    # 只把异常收敛成可读错误，避免把第三方实现细节显示给用户。
    try:
        df = ak.stock_zh_a_minute(symbol=_sina_prefixed(symbol), period=p_val)
        return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_minute")
    except Exception as exc:
        errors.append(f"新浪分钟接口: {type(exc).__name__}")

    detail = "；".join(errors)
    raise ValueError(
        f"{symbol} 的 {p_val} 分钟行情暂时无法获取。"
        "数据源可能无响应、限流或返回了无效数据，请稍后重试或使用演示数据。"
        f"（{detail}）"
    ) from None

def fetch_min(symbol: str, period: str = "5", start_date: str = None) -> pd.DataFrame:
    p_val = str(period).replace("分", "").replace("m", "").strip()
    calc_start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    calc_end = datetime.now().strftime("%Y%m%d")
    kind = guess_kind(symbol)
    try:
        if kind == "etf":
            try:
                df = ak.fund_etf_hist_min_em(symbol=symbol, period=p_val, start_date=calc_start, end_date=calc_end)
                return _clean_df(df, MARKET_CN, "domestic", "akshare.fund_etf_hist_min_em")
            except Exception:
                df = ak.stock_zh_a_hist_min_em(symbol=symbol, period=p_val, start_date=calc_start, end_date=calc_end)
                return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_hist_min_em")
        elif kind == "index":
            return fetch_index_min(symbol, period=p_val)
        else:
            df = ak.stock_zh_a_hist_min_em(symbol=symbol, period=p_val, start_date=calc_start, end_date=calc_end)
            return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_hist_min_em")
    except Exception as exc:
        raise ValueError(f"无法获取 [{symbol}] 的 {p_val} 分钟高频数据: {exc}")

#: pandas 2.x 已废弃的聚合频率别名。
#: 旧写法 "M"/"Q"/"Y" 在 pandas >= 2.2 会直接抛
#: ``ValueError: Invalid frequency: M``，而界面上的「月线」按钮传的正是 "M"。
#: 统一在这里翻译，调用方不必记住新旧两种写法。
_RULE_ALIASES = {
    "M": "ME", "Q": "QE", "Y": "YE", "A": "YE",
    "BM": "BME", "BQ": "BQE", "BA": "BYE",
}


def resample_ohlc(df: pd.DataFrame, rule: str = "W") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "date" not in df.columns:
        raise ValueError("无法重采样：行情数据缺少 date 列")

    rule = _RULE_ALIASES.get(str(rule).strip().upper(), rule)
    tmp = df.copy().set_index("date")
    agg_dict = {
        "open": "first", "high": "max",
        "low": "min", "close": "last",
        "volume": "sum", "amount": "sum"
    }
    # 允许缺少 amount 的数据（例如用户导入的同花顺 TXT）
    agg_dict = {key: how for key, how in agg_dict.items() if key in tmp.columns}
    res = tmp.resample(rule).agg(agg_dict).dropna().reset_index()
    return res[[col for col in COLUMNS if col in res.columns]]

# ==================== 5. 统一对外入口 ====================
def fetch(symbol: str, kind: str = None, period: str = "日线", start_date: str = DEFAULT_START, **kwargs) -> pd.DataFrame:
    sym = normalize_symbol(symbol)
    mkt = detect_market(sym)
    is_min = any(k in str(period) for k in ["1", "5", "15", "30", "60", "分", "m"])

    if is_min:
        return fetch_min(sym, period=period, start_date=start_date)

    if mkt == MARKET_HK:
        return fetch_hk_daily(sym, start_date=start_date)
    elif mkt == MARKET_US:
        return fetch_us_daily(sym, start_date=start_date)

    # A 股分支必须拦住非法代码：否则会一路透传到 akshare 的新浪兜底接口，
    # 那里会抛出毫无信息量的裸 ``KeyError: 'date'``（用户截图里的报错）。
    if not _CN_CODE_RE.match(sym):
        raise ValueError(
            f"「{sym}」不是有效的 A 股代码（应为 6 位数字）。"
            "美股请输入 QQQ / SPY / AAPL 等，港股请输入 00700 / 09988 等；"
            "纳指相关场内基金可试用 513100。"
        )

    if kind is None or kind == "自动":
        real_kind = guess_kind(sym)
    else:
        real_kind = KINDS.get(kind, kind).lower()

    if real_kind == "index":
        df = fetch_index_daily(sym, start_date=start_date)
    elif real_kind == "etf":
        df = fetch_etf_daily(sym, start_date=start_date)
    else:
        df = fetch_stock_daily(sym, start_date=start_date)

    if "周" in str(period) or str(period).lower() == "weekly":
        df = resample_ohlc(df, "W")
    elif "月" in str(period) or str(period).lower() == "monthly":
        df = resample_ohlc(df, "ME")

    return df
