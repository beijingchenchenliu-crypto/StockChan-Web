"""缠论与形态计算引擎。

模块划分::

    models.py     数据结构（合并K线 / 分型 / 笔 / 中枢 / 波浪点 / 买卖点）
    merge.py      包含关系合并
    fractal.py    分型识别
    stroke.py     画笔
    pivot.py      中枢识别
    indicator.py  MACD 力度 + ATR/布林/肯特纳/Squeeze/CHOP/SuperTrend/PVT/TRIX
    signals.py    一/二/三类买卖点判定
    channel.py    趋势通道（支撑轨 / 阻力轨）
    wave.py       波浪形态初步标记
    analyzer.py   编排入口 ChanAnalyzer

快速使用::

    from core import ChanAnalyzer
    result = ChanAnalyzer(df).run()
    result.strokes, result.pivots, result.trade_points
    result.channel, result.supertrend, result.latest_regime
"""

from .analyzer import ChanAnalyzer, ChanResult
from .channel import (
    DEFAULT_EXTEND,
    EXTEND_RANGE,
    TrendChannel,
    clamp_extend,
    find_channel,
)
from .fractal import find_fractals
from .indicator import (
    SUBPLOT_INDICATORS,
    StrokePower,
    compute_atr,
    compute_bollinger,
    compute_choppiness,
    compute_indicators,
    compute_keltner,
    compute_macd,
    compute_pvt,
    compute_stroke_powers,
    compute_supertrend,
    compute_trix,
    compute_ttm_squeeze,
    market_regime,
)
from .merge import merge_inclusion
from .models import (
    BOTTOM,
    TOP,
    Fractal,
    MergedBar,
    Pivot,
    Stroke,
    TradePoint,
    WavePoint,
)
from .pivot import find_pivots
from .signals import find_trade_points
from .stroke import build_strokes
from .wave import label_waves

__all__ = [
    "ChanAnalyzer",
    "ChanResult",
    "merge_inclusion",
    "find_fractals",
    "build_strokes",
    "find_pivots",
    "label_waves",
    "compute_macd",
    "compute_stroke_powers",
    "compute_atr",
    "compute_bollinger",
    "compute_keltner",
    "compute_ttm_squeeze",
    "compute_choppiness",
    "compute_supertrend",
    "compute_pvt",
    "compute_trix",
    "compute_indicators",
    "market_regime",
    "SUBPLOT_INDICATORS",
    "find_trade_points",
    "find_channel",
    "TrendChannel",
    "DEFAULT_EXTEND",
    "EXTEND_RANGE",
    "clamp_extend",
    "MergedBar",
    "Fractal",
    "Stroke",
    "Pivot",
    "WavePoint",
    "TradePoint",
    "StrokePower",
    "TOP",
    "BOTTOM",
]
