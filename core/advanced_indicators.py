"""独立高级技术指标。

本模块不参与缠论笔、分型、线段、中枢或行情数据拉取流程；所有函数均为
纯计算函数：不修改传入的 DataFrame，也不保存任何全局状态。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """校验输入行情列，避免静默使用错误数据。"""
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"缺少指标计算所需列: {', '.join(missing)}")


def compute_anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    """计算从 ``anchor_idx`` 开始向右延伸的 Anchored VWAP。

    使用典型价格 ``(high + low + close) / 3`` 与成交量加权。锚点之前的值为
    ``NaN``，这样可直接叠加到原始 K 线横轴而不会错误画到锚点左侧。
    """
    _require_columns(df, ("high", "low", "close", "volume"))
    if not 0 <= int(anchor_idx) < len(df):
        raise IndexError(f"anchor_idx 必须位于 0 到 {max(len(df) - 1, 0)} 之间")

    idx = int(anchor_idx)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").clip(lower=0).fillna(0.0)
    typical = (high + low + close) / 3.0

    weighted_price = (typical.iloc[idx:] * volume.iloc[idx:]).fillna(0.0).cumsum()
    cumulative_volume = volume.iloc[idx:].cumsum().replace(0.0, np.nan)
    result = pd.Series(np.nan, index=df.index, dtype="float64", name="anchored_vwap")
    result.iloc[idx:] = weighted_price / cumulative_volume
    return result


def detect_macd_patterns(
    df: pd.DataFrame,
    dif: pd.Series,
    dea: pd.Series,
    hist: pd.Series,
) -> Dict[str, List[int]]:
    """量化检测 MACD 经典多空加速中继。

    ``air_refuel`` 为水上洗盘结束后的多头中继，``frost_on_snow`` 为水下
    弱反弹夭折后的空头二次加速，``bullish_divergence`` 为底背离。结果仅用于
    观察和副图标记，不参与缠论买卖点的确认计算。
    """
    _require_columns(df, ("open", "low", "close"))
    n = len(df)
    patterns: Dict[str, List[int]] = {
        "air_refuel": [],
        "frost_on_snow": [],
        "bullish_divergence": [],
    }
    if not (len(dif) == len(dea) == len(hist) == n):
        raise ValueError("df、dif、dea、hist 长度必须一致")
    if n < 20:
        return patterns

    d = pd.Series(dif, dtype="float64").reset_index(drop=True)
    e = pd.Series(dea, dtype="float64").reset_index(drop=True)
    h = pd.Series(hist, dtype="float64").reset_index(drop=True)
    frame = df.reset_index(drop=True)

    for i in range(5, n):
        h_cur, h_prev, h_prev2 = h.iloc[i], h.iloc[i - 1], h.iloc[i - 2]
        if not (np.isfinite(h_cur) and np.isfinite(h_prev) and np.isfinite(h_prev2)):
            continue

        # 1. 空中加油：水上多头中继，红柱收缩后再度放大。
        if d.iloc[i] > -0.05 and e.iloc[i] > -0.05:
            was_cooling = (h_prev2 > h_prev) or (h_prev < 0 and h_prev > -0.1)
            turned_up = h_cur > h_prev
            dif_dea_dist = d.iloc[i] - e.iloc[i]
            refuel_ready = (-0.02 <= dif_dea_dist <= 0.08) and (d.iloc[i] >= d.iloc[i - 1] * 0.98)
            if (was_cooling and turned_up and refuel_ready
                    and frame["close"].iloc[i] >= frame["open"].iloc[i]):
                patterns["air_refuel"].append(i)

        # 2. 雪上加霜：水下弱反弹结束，绿柱二次向下放大。
        elif d.iloc[i] < 0.05 and e.iloc[i] < 0.05:
            was_rebounding = (h_prev2 < h_prev) or (h_prev > 0 and h_prev < 0.1)
            turned_down = (h_cur < h_prev) and (h_cur < 0)
            dif_dist = e.iloc[i] - d.iloc[i]
            frost_ready = (-0.02 <= dif_dist <= 0.08) and (d.iloc[i] <= d.iloc[i - 1] * 1.02)
            is_down_day = frame["close"].iloc[i] < frame["close"].iloc[i - 1]
            if was_rebounding and turned_down and frost_ready and is_down_day:
                patterns["frost_on_snow"].append(i)

        # 3. 底背离：近期新低，但 DIF 没有同步创新低。
        recent_low_idx = frame["low"].iloc[max(0, i - 10):i + 1].idxmin()
        if recent_low_idx == i:
            previous_lows = [
                j for j in range(max(0, i - 30), max(0, i - 8))
                if frame["low"].iloc[j] == frame["low"].iloc[max(0, j - 4):min(n, j + 5)].min()
            ]
            if previous_lows:
                prev_idx = previous_lows[-1]
                if frame["low"].iloc[i] < frame["low"].iloc[prev_idx] and d.iloc[i] > d.iloc[prev_idx]:
                    patterns["bullish_divergence"].append(i)

    return patterns


def detect_duck_head(df: pd.DataFrame) -> pd.DataFrame:
    """识别均线老鸭头的放量启动候选。

    规则为 MA5 > MA10 > MA60 的多头结构；MA5 最近 5 根曾回踩 MA10 附近；
    当前收盘突破前三根高点，并且成交量至少为前五根均量的 1.5 倍。
    返回布尔 ``duck_head`` 及中间均线列，供界面解释与绘制。
    """
    _require_columns(df, ("close", "high", "volume"))
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").clip(lower=0)

    ma5 = close.rolling(5, min_periods=5).mean()
    ma10 = close.rolling(10, min_periods=10).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    trend_up = (ma5 > ma10) & (ma10 > ma60)
    pullback_near_ma10 = ((close - ma10).abs() / ma10.replace(0.0, np.nan)).rolling(5, min_periods=1).min() <= 0.008
    breakout = close > high.shift(1).rolling(3, min_periods=3).max()
    volume_confirm = volume >= volume.shift(1).rolling(5, min_periods=3).mean() * 1.5
    duck_head = (trend_up & pullback_near_ma10 & breakout & volume_confirm).fillna(False)

    return pd.DataFrame({
        "ma5": ma5,
        "ma10": ma10,
        "ma60": ma60,
        "duck_head": duck_head.astype(bool),
    }, index=df.index)


def calculate_chop_filter(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算 Choppiness Index（切波指数，0~100）。

    值较低通常代表趋势行情，值较高通常代表震荡行情。返回值与 ``df`` 索引一致。
    """
    _require_columns(df, ("high", "low", "close"))
    if int(period) < 2:
        raise ValueError("period 必须大于或等于 2")

    p = int(period)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    tr_sum = true_range.rolling(p, min_periods=p).sum()
    price_range = high.rolling(p, min_periods=p).max() - low.rolling(p, min_periods=p).min()
    ratio = tr_sum / price_range.replace(0.0, np.nan)
    return (100.0 * np.log10(ratio) / np.log10(p)).rename("chop")
