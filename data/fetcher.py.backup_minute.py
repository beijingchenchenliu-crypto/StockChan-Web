# -*- coding: utf-8 -*-
import os
import re
import time
import requests
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
def detect_market(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if re.match(r"^[A-Z]{1,5}$", s):
        return MARKET_US
    if s.startswith("HK") or (s.isdigit() and len(s) == 5):
        return MARKET_HK
    return MARKET_CN

def guess_kind(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if s in ["000001", "399001", "399006", "000300", "000905", "000852", "000688", "H30021"]:
        return "index"
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
    return str(symbol).strip().upper()

# ==================== 3. 稳健数据规整 ====================
def _clean_df(df: pd.DataFrame, market: str, channel: str, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("拉取到的行情数据为空")

    df = df.copy()

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
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=s_date, end_date=e_date, adjust="qfq")
        return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_hist")
    except Exception:
        sym = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"
        df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
        return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_daily")

def fetch_hk_daily(symbol: str, start_date: str = DEFAULT_START) -> pd.DataFrame:
    sym = symbol.replace("HK", "")
    df = ak.stock_hk_hist(symbol=sym, period="daily", start_date=str(start_date).replace("-", ""))
    return _clean_df(df, MARKET_HK, "overseas", "akshare.stock_hk_hist")

def fetch_us_daily(symbol: str, start_date: str = DEFAULT_START) -> pd.DataFrame:
    df = ak.stock_us_hist(symbol=symbol.upper(), period="daily", start_date=str(start_date).replace("-", ""))
    return _clean_df(df, MARKET_US, "overseas", "akshare.stock_us_hist")

def fetch_index_min(symbol: str, period: str = "5", start_date: str = None) -> pd.DataFrame:
    p_val = str(period).replace("分", "").replace("m", "").strip()
    calc_start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    calc_end = datetime.now().strftime("%Y%m%d")
    try:
        df = ak.index_zh_a_hist_min_em(symbol=symbol, period=p_val, start_date=calc_start, end_date=calc_end)
        return _clean_df(df, MARKET_CN, "domestic", "akshare.index_zh_a_hist_min_em")
    except Exception:
        df = ak.stock_zh_a_minute(symbol=symbol, period=f"{p_val}m")
        return _clean_df(df, MARKET_CN, "domestic", "akshare.stock_zh_a_minute")

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

def resample_ohlc(df: pd.DataFrame, rule: str = "W") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tmp = df.copy().set_index("date")
    agg_dict = {
        "open": "first", "high": "max",
        "low": "min", "close": "last",
        "volume": "sum", "amount": "sum"
    }
    res = tmp.resample(rule).agg(agg_dict).dropna().reset_index()
    return res[COLUMNS]

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