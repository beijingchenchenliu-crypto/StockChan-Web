# -*- coding: utf-8 -*-
"""缠论分析编排入口（完整对接多尺度趋势结构算法）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .channel import DEFAULT_EXTEND, TrendChannel, clamp_extend, find_channel
from .fractal import find_fractals
from .indicator import (
    StrokePower,
    compute_indicators,
    compute_macd,
    compute_stroke_powers,
    market_regime,
)
from .merge import merge_inclusion
from .models import (
    Fractal,
    MergedBar,
    Pivot,
    Stroke,
    TradePoint,
    WavePoint,
)
from .pivot import find_pivots
from .signals import DIVERGENCE_RATIO, _debug_enabled, find_trade_points
from .stroke import MIN_GAP, build_strokes
from .wave import label_waves


@dataclass
class ChanResult:
    """一次完整分析的产物。"""

    df: pd.DataFrame
    dates: List[str] = field(default_factory=list)
    bars: List[MergedBar] = field(default_factory=list)
    fractals: List[Fractal] = field(default_factory=list)
    strokes: List[Stroke] = field(default_factory=list)
    pivots: List[Pivot] = field(default_factory=list)
    waves: List[WavePoint] = field(default_factory=list)
    macd: Optional[pd.DataFrame] = None
    powers: List[StrokePower] = field(default_factory=list)
    trade_points: List[TradePoint] = field(default_factory=list)
    indicators: Dict[str, object] = field(default_factory=dict)
    channel: Optional[TrendChannel] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return str(self.meta.get("symbol", ""))

    @property
    def name(self) -> str:
        return str(self.meta.get("name", ""))

    @property
    def buy_points(self) -> List[TradePoint]:
        return [t for t in self.trade_points if t.side == "buy"]

    @property
    def sell_points(self) -> List[TradePoint]:
        return [t for t in self.trade_points if t.side == "sell"]

    @property
    def confirmed_points(self) -> List[TradePoint]:
        return [t for t in self.trade_points if t.confirmed]

    @property
    def tentative_points(self) -> List[TradePoint]:
        return [t for t in self.trade_points if t.tentative]

    def _ind(self, key: str):
        return self.indicators.get(key) if self.indicators else None

    @property
    def chop(self):
        return self._ind("chop")

    @property
    def squeeze(self):
        return self._ind("squeeze")

    @property
    def supertrend(self):
        return self._ind("supertrend")

    @property
    def bull_bear(self):
        return self._ind("bull_bear")

    @property
    def latest_chop(self) -> Optional[float]:
        s = self.chop
        if s is None or len(s) == 0:
            return None
        value = float(s.iloc[-1])
        return value if np.isfinite(value) else None

    @property
    def latest_regime(self) -> str:
        return market_regime(self.latest_chop)

    @property
    def squeeze_on(self) -> bool:
        s = self.squeeze
        if s is None or len(s) == 0:
            return False
        return bool(s["on"].iloc[-1])

    def _level_counts(self, side: str) -> str:
        pts = [t for t in self.trade_points if t.side == side]
        if not pts:
            return "0"
        parts = []
        for lv in (1, 2, 3):
            c = sum(1 for t in pts if t.level == lv)
            if c:
                parts.append(f"{lv}{'B' if side == 'buy' else 'S'}:{c}")
        return f"{len(pts)}（{' '.join(parts)}）"

    def summary(self) -> Dict[str, object]:
        last_close = float(self.df["close"].iloc[-1]) if len(self.df) else float("nan")
        return {
            "标的": f"{self.name or ''} {self.symbol}".strip(),
            "原始K线": len(self.df),
            "合并K线": len(self.bars),
            "分型数": len(self.fractals),
            "笔数": len(self.strokes),
            "中枢数": len(self.pivots),
            "买点": self._level_counts("buy"),
            "卖点": self._level_counts("sell"),
            "波浪标记": len(self.waves),
            "市场状态": self.latest_regime,
            "最新收盘": round(last_close, 2) if last_close == last_close else None,
        }

    def describe(self) -> str:
        lines = ["===== 缠论与形态分析结果 ====="]
        for k, v in self.summary().items():
            lines.append(f"{k:>8}: {v}")

        if self.channel is not None:
            lines.append(f"{'主结构':>8}: {self.channel.direction}（{self.channel.status}）")
            if self.channel.primary_structure:
                p = self.channel.primary_structure
                lines.append(f"{'':>8}  跨度 {p.fit_span} 根，触点 {p.touch_count} 次，置信度 {p.confidence:.2f}")
            if self.channel.secondary_structure:
                s = self.channel.secondary_structure
                lines.append(f"{'局部结构':>8}: {s.label}（{s.source_window}窗口，距最新 {s.last_touch_distance} 根）")

        return "\n".join(lines)


class ChanAnalyzer:
    """缠论与通用多尺度形态分析引擎。"""

    def __init__(
        self,
        df: pd.DataFrame,
        min_gap: int = MIN_GAP,
        divergence_ratio: float = DIVERGENCE_RATIO,
        strict_wave: bool = False,
        channel_points: int = 18,
        channel_extend: int = DEFAULT_EXTEND,
        period_key: str = "D",
        min_period: Optional[str] = None,
        lookback_mode: str = "深度",
        meta: Optional[Dict[str, object]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True) if df is not None else pd.DataFrame()
        self.min_gap = min_gap
        self.divergence_ratio = divergence_ratio
        self.strict_wave = strict_wave
        self.channel_points = channel_points
        self.channel_extend = clamp_extend(channel_extend)
        self.period_key = period_key
        self.min_period = min_period
        self.lookback_mode = lookback_mode
        self.meta = dict(meta or {})
        self.result: Optional[ChanResult] = None

    def _dates(self) -> List[str]:
        if self.df.empty or "date" not in self.df.columns:
            return []
        return [d.strftime("%Y-%m-%d") for d in pd.to_datetime(self.df["date"])]

    def run(self) -> ChanResult:
        dates = self._dates()
        bars = merge_inclusion(self.df)
        fractals = find_fractals(bars, dates)
        strokes = build_strokes(fractals, min_gap=self.min_gap)
        pivots = find_pivots(strokes)

        indicators: Dict[str, object] = {}
        macd = None
        powers: List[StrokePower] = []
        if not self.df.empty and "close" in self.df.columns:
            indicators = compute_indicators(self.df)
            macd = indicators.get("macd")
            powers = compute_stroke_powers(strokes, macd)

        trade_points = find_trade_points(
            strokes, pivots, powers,
            ratio=self.divergence_ratio,
            df=self.df,
            debug_dump=_debug_enabled(),
        )

        latest_close = (
            float(self.df["close"].iloc[-1])
            if not self.df.empty and "close" in self.df.columns
            else None
        )

        # 多尺度通道与形态扫描
        channel = find_channel(
            strokes=strokes,
            total_bars=len(self.df),
            n_points=self.channel_points,
            extend=self.channel_extend,
            latest_close=latest_close,
            df=self.df,
            level="stroke",
            period_key=self.period_key,
            min_period=self.min_period,
            lookback_mode=self.lookback_mode,
        )

        waves = label_waves(strokes, strict=self.strict_wave)

        self.result = ChanResult(
            df=self.df,
            dates=dates,
            bars=bars,
            fractals=fractals,
            strokes=strokes,
            pivots=pivots,
            waves=waves,
            macd=macd,
            powers=powers,
            trade_points=trade_points,
            indicators=indicators,
            channel=channel,
            meta=self.meta,
        )
        return self.result

    @staticmethod
    def analyze(df: pd.DataFrame, **kwargs) -> ChanResult:
        return ChanAnalyzer(df, **kwargs).run()
