"""技术指标库。

包含两类内容：

**一、背驰量化（缠论专用）**
    把 MACD 红绿柱的"面积"落到每一笔上，作为走势力度衰减的判据。

**二、过滤/辅助指标**
    - ATR            波动幅度，多个指标的底座
    - 布林带 / 肯特纳   TTM Squeeze 的两个通道
    - TTM Squeeze     波动率挤压 → 释放，判断变盘
    - CHOP 混沌指数    趋势市 (<38.2) / 震荡市 (>61.8)
    - SuperTrend      动态移动止损参考
    - PVT             价量趋势
    - TRIX            三重指数平滑均线

MACD 采用国内常用参数 (12, 26, 9)，柱值 = 2 × (DIF − DEA)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .models import Stroke

# =========================================================================== #
# 一、MACD 与背驰量化
# =========================================================================== #


def compute_macd(
    close: Sequence[float] | pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """计算 MACD。

    :return: 含 ``dif`` / ``dea`` / ``hist`` 三列的 DataFrame，索引与输入对齐
    """
    s = pd.Series(close, dtype="float64").reset_index(drop=True)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2.0

    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


@dataclass
class StrokePower:
    """一笔的 MACD 力度。"""

    area: float = 0.0         #: 同向柱面积之和
    peak: float = 0.0         #: 柱绝对值峰值
    dif_extreme: float = 0.0  #: 该笔终点侧 DIF 极值

    def __repr__(self) -> str:  # pragma: no cover
        return f"StrokePower(area={self.area:.2f}, peak={self.peak:.2f})"


def stroke_power(
    hist: np.ndarray,
    dif: np.ndarray,
    i0: int,
    i1: int,
    direction: str,
) -> StrokePower:
    """统计 ``[i0, i1]`` 区间内某一笔的 MACD 力度。"""
    lo, hi = max(0, int(i0)), min(len(hist) - 1, int(i1))
    if hi < lo:
        return StrokePower()

    seg = np.asarray(hist[lo:hi + 1], dtype=float)
    dif_seg = np.asarray(dif[lo:hi + 1], dtype=float)

    if direction == "down":
        same_dir = seg[seg < 0]
        area = float(np.abs(same_dir).sum())
        dif_extreme = float(dif_seg.min()) if dif_seg.size else 0.0
    else:
        same_dir = seg[seg > 0]
        area = float(same_dir.sum())
        dif_extreme = float(dif_seg.max()) if dif_seg.size else 0.0

    peak = float(np.abs(seg).max()) if seg.size else 0.0
    return StrokePower(area=area, peak=peak, dif_extreme=dif_extreme)


def compute_stroke_powers(
    strokes: Sequence[Stroke],
    macd: pd.DataFrame,
) -> list[StrokePower]:
    """为每一笔计算 MACD 力度，返回与 ``strokes`` 等长的列表。"""
    hist = macd["hist"].to_numpy(dtype=float)
    dif = macd["dif"].to_numpy(dtype=float)
    return [
        stroke_power(hist, dif, s.raw_start, s.raw_end, s.direction)
        for s in strokes
    ]


# =========================================================================== #
# 二、基础工具
# =========================================================================== #


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真实波幅（Wilder 平滑）。"""
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _rolling_linreg_endpoint(values: pd.Series, window: int) -> pd.Series:
    """滚动最小二乘拟合，取每个窗口末端点的拟合值（TTM 动量用）。"""
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) < window or window < 2:
        return pd.Series(out, index=values.index)

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = float(((x - x_mean) ** 2).sum())

    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1: i + 1]
        if not np.isfinite(y).all():
            continue
        y_mean = y.mean()
        slope = float(((x - x_mean) * (y - y_mean)).sum() / denom) if denom else 0.0
        out[i] = y_mean + slope * (window - 1 - x_mean)

    return pd.Series(out, index=values.index)


# =========================================================================== #
# 三、波动率通道与挤压
# =========================================================================== #


def compute_bollinger(
    close: Sequence[float] | pd.Series,
    period: int = 20,
    mult: float = 2.0,
) -> pd.DataFrame:
    """布林带。"""
    s = pd.Series(close, dtype="float64").reset_index(drop=True)
    mid = s.rolling(period, min_periods=period).mean()
    std = s.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame({
        "mid": mid,
        "upper": mid + mult * std,
        "lower": mid - mult * std,
    })


def compute_keltner(
    df: pd.DataFrame,
    period: int = 20,
    mult: float = 1.5,
) -> pd.DataFrame:
    """肯特纳通道（EMA ± mult × ATR）。"""
    close = df["close"].astype("float64")
    mid = close.ewm(span=period, adjust=False).mean()
    atr = compute_atr(df, period)
    return pd.DataFrame({
        "mid": mid,
        "upper": mid + mult * atr,
        "lower": mid - mult * atr,
    })


def compute_ttm_squeeze(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    mom_period: int = 20,
) -> pd.DataFrame:
    """TTM Squeeze。

    当**布林带完全落入肯特纳通道内部**时，波动率被挤压（``on=True``）；
    挤压解除（``on`` 由 True 变 False）往往意味着方向选择、变盘启动。

    :return: ``on``（挤压中）/ ``release``（当根刚释放）/ ``momentum``（动量柱）
    """
    bb = compute_bollinger(df["close"], bb_period, bb_mult)
    kc = compute_keltner(df, kc_period, kc_mult)

    on = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
    on = on.fillna(False)

    # 动量：收盘价 − (唐奇安中轨 + 均线) / 2，再取线性回归末端值
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")

    donchian_mid = (high.rolling(mom_period, min_periods=1).max()
                    + low.rolling(mom_period, min_periods=1).min()) / 2.0
    sma_close = close.rolling(mom_period, min_periods=1).mean()
    base = close - (donchian_mid + sma_close) / 2.0
    momentum = _rolling_linreg_endpoint(base, mom_period)

    release = on.shift(1, fill_value=False) & (~on)

    return pd.DataFrame({
        "on": on.to_numpy(),
        "release": release.to_numpy(),
        "momentum": momentum.to_numpy(),
        "bb_upper": bb["upper"].to_numpy(),
        "bb_lower": bb["lower"].to_numpy(),
        "kc_upper": kc["upper"].to_numpy(),
        "kc_lower": kc["lower"].to_numpy(),
    })


def compute_choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """CHOP 混沌指数（0–100）。

    ``< 38.2`` 趋势市（顺势指标更可靠）；``> 61.8`` 震荡市（信号易被反复打脸）。
    """
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    atr_sum = compute_atr(df, 1).rolling(period, min_periods=period).sum()
    rng = high.rolling(period, min_periods=period).max() - low.rolling(period, min_periods=period).min()

    ratio = atr_sum / rng.replace(0.0, np.nan)
    chop = 100.0 * np.log10(ratio) / np.log10(period)
    return chop.rename("chop")


# =========================================================================== #
# 四、趋势与量能
# =========================================================================== #


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    mult: float = 3.0,
) -> pd.DataFrame:
    """SuperTrend 超级趋势线。

    :return: ``line``（趋势线，可作动态移动止损）/ ``direction``（1 多头 / -1 空头）
             / ``upper`` / ``lower``（未平滑的上下轨）
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    atr = compute_atr(df, period).to_numpy(dtype=float)
    hl2 = (high + low) / 2.0
    raw_upper = hl2 + mult * atr
    raw_lower = hl2 - mult * atr

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    line = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    for i in range(n):
        if i == 0 or not np.isfinite(atr[i]) or not np.isfinite(final_upper[i - 1]):
            final_upper[i] = raw_upper[i]
            final_lower[i] = raw_lower[i]
            direction[i] = 1 if close[i] >= hl2[i] else -1
        else:
            final_upper[i] = (min(raw_upper[i], final_upper[i - 1])
                              if close[i - 1] <= final_upper[i - 1] else raw_upper[i])
            final_lower[i] = (max(raw_lower[i], final_lower[i - 1])
                              if close[i - 1] >= final_lower[i - 1] else raw_lower[i])

            if close[i] > final_upper[i - 1]:
                direction[i] = 1
            elif close[i] < final_lower[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1] or 1

        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.DataFrame({
        "line": line,
        "direction": direction,
        "upper": final_upper,
        "lower": final_lower,
    })


def compute_pvt(df: pd.DataFrame) -> pd.Series:
    """PVT 价量趋势（累积量能）。"""
    close = df["close"].astype("float64")
    volume = df["volume"].astype("float64")

    prev = close.shift(1)
    change = (close - prev) / prev.replace(0.0, np.nan)
    pvt = (volume * change).fillna(0.0).cumsum()
    return pvt.rename("pvt")


def compute_trix(
    close: Sequence[float] | pd.Series,
    period: int = 12,
    signal: int = 9,
) -> pd.DataFrame:
    """TRIX 三重指数平滑均线及其信号线。"""
    s = pd.Series(close, dtype="float64").reset_index(drop=True)

    ema1 = s.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()

    trix = (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0.0, np.nan) * 100.0
    return pd.DataFrame({
        "trix": trix,
        "signal": trix.ewm(span=signal, adjust=False).mean(),
    })


# =========================================================================== #
# 五、统一入口
# =========================================================================== #

#: 副图可切换的指标
SUBPLOT_INDICATORS = ("MACD", "PVT", "TRIX", "Squeeze")


def compute_bull_bear_line(df: pd.DataFrame, period: int = 60) -> pd.Series:
    """牛熊分界线：收盘价的 60 周期均线，数据不足时从首根K线起计算。"""
    close = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    return close.rolling(max(int(period), 2), min_periods=1).mean().rename("bull_bear")


def compute_indicators(df: pd.DataFrame) -> Dict[str, object]:
    """一次性算齐所有指标，供 analyzer 与界面使用。"""
    if df is None or df.empty:
        return {}

    return {
        "macd": compute_macd(df["close"]),
        "atr": compute_atr(df),
        "bollinger": compute_bollinger(df["close"]),
        "keltner": compute_keltner(df),
        "squeeze": compute_ttm_squeeze(df),
        "chop": compute_choppiness(df),
        "bull_bear": compute_bull_bear_line(df),
        "pvt": compute_pvt(df),
        "trix": compute_trix(df["close"]),
    }


def market_regime(chop_value: Optional[float]) -> str:
    """根据 CHOP 值给出市场状态文字。"""
    if chop_value is None or not np.isfinite(chop_value):
        return "未知"
    if chop_value < 38.2:
        return "趋势市"
    if chop_value > 61.8:
        return "震荡市"
    return "过渡区"
