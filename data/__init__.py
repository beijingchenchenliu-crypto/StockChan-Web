"""数据层：行情获取 + 网络通道 + 本地缓存。

对外主要接口::

    from data import fetch                      # 统一入口（自动识别市场）
    from data import fetch_index_daily, fetch_etf_daily, fetch_stock_daily
    from data import fetch_hk_daily, fetch_us_daily
    from data import detect_market, guess_kind, last_route

所有 fetch_* 函数都返回统一的 DataFrame，列固定为::

    date | open | high | low | close | volume | amount

其中 ``date`` 为 ``pandas.Timestamp``，其余为 ``float``，按时间升序排列。
取数成功后 ``df.attrs["route"]`` 会记录实际使用的市场 / 网络通道 / 接口::

    from data import fetch, last_route
    df = fetch("AAPL")
    print(last_route(df))     # {'market': 'us', 'channel': 'overseas', ...}
"""

from . import net
from .cache import CACHE_DIR, clear_cache, load_cache, save_cache
from .fetcher import (
    COLUMNS,
    COOLDOWN,
    KINDS,
    MARKET_CN,
    MARKET_HK,
    MARKET_LABELS,
    MARKET_US,
    default_symbol,
    detect_market,
    fetch,
    fetch_etf_daily,
    fetch_hk_daily,
    fetch_index_daily,
    fetch_index_min,
    fetch_min,
    fetch_stock_daily,
    fetch_us_daily,
    guess_kind,
    interface_health,
    last_route,
    market_of_kind,
    normalize_symbol,
    resample_ohlc,
    reset_strategy_memory,
    strategy_memory,
)
from .mock import make_mock_daily
from .suggest import (
    SUGGEST_TIMEOUT,
    SUGGEST_TYPES_EXTENDED,
    SUGGEST_TYPES_PRIMARY,
    Suggestion,
    looks_like_code,
    needs_lookup,
    resolve_name,
    suggest,
)
from .net import (
    CHANNELS,
    CHANNEL_LABELS,
    DEFAULT_TIMEOUT,
    DOMESTIC,
    MAX_RETRIES,
    OVERSEAS,
    install_patch,
    probe_channels,
    probe_report,
    session_domestic,
    session_overseas,
    system_proxies,
    use_channel,
)

__all__ = [
    # 取数
    "fetch",
    "fetch_min",
    "fetch_index_daily",
    "fetch_etf_daily",
    "fetch_stock_daily",
    "fetch_hk_daily",
    "fetch_us_daily",
    "fetch_index_min",
    "resample_ohlc",
    "make_mock_daily",
    # 市场识别
    "detect_market",
    "normalize_symbol",
    "guess_kind",
    "market_of_kind",
    "default_symbol",
    "last_route",
    # 名称 / 拼音联想
    "suggest",
    "resolve_name",
    "looks_like_code",
    "needs_lookup",
    "Suggestion",
    "SUGGEST_TYPES_PRIMARY",
    "SUGGEST_TYPES_EXTENDED",
    "SUGGEST_TIMEOUT",
    "strategy_memory",
    "interface_health",
    "reset_strategy_memory",
    "COOLDOWN",
    "KINDS",
    "COLUMNS",
    "MARKET_CN",
    "MARKET_HK",
    "MARKET_US",
    "MARKET_LABELS",
    # 网络层
    "net",
    "DOMESTIC",
    "OVERSEAS",
    "CHANNELS",
    "CHANNEL_LABELS",
    "DEFAULT_TIMEOUT",
    "MAX_RETRIES",
    "session_domestic",
    "session_overseas",
    "use_channel",
    "install_patch",
    "probe_channels",
    "probe_report",
    "system_proxies",
    # 缓存
    "load_cache",
    "save_cache",
    "clear_cache",
    "CACHE_DIR",
]
