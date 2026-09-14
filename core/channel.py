# -*- coding: utf-8 -*-
"""通用多尺度趋势结构识别引擎（聚焦2026年中期下降切线与局部结构）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import BOTTOM, TOP, Stroke

Rail = Tuple[float, float, float, float]
DEFAULT_EXTEND = 12
EXTEND_RANGE = (10, 15)


def clamp_extend(extend: int) -> int:
    lo, hi = EXTEND_RANGE
    return int(min(max(int(extend), lo), hi))


def line_value(line: Dict[str, Any], raw_index: int) -> float:
    """以原始K线绝对索引计算趋势线价格：Y = k * X + b。"""
    if not isinstance(line, dict):
        raise TypeError("line must be a mapping")
    return float(line["slope"]) * int(raw_index) + float(line["intercept"])


@dataclass
class StructureItem:
    pattern_type: str = "horizontal_rectangle"
    label: str = "普通整理"
    status: str = "candidate"
    status_reason: str = ""
    confidence: float = 0.50
    level: str = "stroke"
    source_window: str = "medium"
    score: float = 0.0

    fit_span: int = 0
    total_span: int = 0
    last_touch_distance: int = 0
    alternation_ratio: float = 0.0

    actual_latest_close: float = 0.0
    touch_tolerance: float = 0.0
    width_start: float = 0.0
    width_end: float = 0.0
    current_upper: float = 0.0
    current_lower: float = 0.0
    dist_to_upper_pct: float = 0.0
    dist_to_lower_pct: float = 0.0
    breakout_up_level: float = 0.0
    breakdown_level: float = 0.0

    upper_line: Dict[str, Any] = field(default_factory=dict)
    lower_line: Dict[str, Any] = field(default_factory=dict)
    touch_count: int = 0
    break_count: int = 0
    points_detail: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    support_rail: Optional[Rail] = None
    resistance_rail: Optional[Rail] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "type": self.pattern_type,
            "label": self.label,
            "status": self.status,
            "status_reason": self.status_reason,
            "confidence": self.confidence,
            "level": self.level,
            "source_window": self.source_window,
            "score": round(self.score, 2),
            "fit_span": self.fit_span,
            "total_span": self.total_span,
            "last_touch_distance": self.last_touch_distance,
            "alternation_ratio": round(self.alternation_ratio, 2),
            "actual_latest_close": self.actual_latest_close,
            "touch_tolerance": self.touch_tolerance,
            "width_start": self.width_start,
            "width_end": self.width_end,
            "current_upper": self.current_upper,
            "current_lower": self.current_lower,
            "dist_to_upper_pct": self.dist_to_upper_pct,
            "dist_to_lower_pct": self.dist_to_lower_pct,
            "breakout_up_level": self.breakout_up_level,
            "breakdown_level": self.breakdown_level,
            "upper_line": self.upper_line,
            "lower_line": self.lower_line,
            "touch_count": self.touch_count,
            "break_count": self.break_count,
            "points_detail": self.points_detail,
            "reason": self.reason,
            "support": self.support_rail,
            "resistance": self.resistance_rail,
        }


@dataclass
class TrendChannel:
    support: Optional[Rail] = None
    resistance: Optional[Rail] = None
    support_points: List[Tuple[int, float]] = field(default_factory=list)
    resistance_points: List[Tuple[int, float]] = field(default_factory=list)
    slope_support: float = 0.0
    slope_resistance: float = 0.0
    extend: int = DEFAULT_EXTEND
    rejected: List[str] = field(default_factory=list)

    primary_structure: Optional[StructureItem] = None
    secondary_structure: Optional[StructureItem] = None
    pattern_type: str = "horizontal_rectangle"
    label: str = "普通整理"
    status: str = "candidate"
    confidence: float = 0.50
    level: str = "stroke"
    actual_latest_close: float = 0.0
    reason: str = ""

    @property
    def valid(self) -> bool:
        return self.support is not None or self.resistance is not None

    @property
    def direction(self) -> str:
        return self.label

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "valid": self.valid,
            "direction": self.direction,
            "type": self.pattern_type,
            "label": self.label,
            "status": self.status,
            "confidence": self.confidence,
            "level": self.level,
            "actual_latest_close": self.actual_latest_close,
            "support": self.support,
            "resistance": self.resistance,
            "slope_support": self.slope_support,
            "slope_resistance": self.slope_resistance,
            "extend": self.extend,
            "reason": self.reason,
            "primary": self.primary_structure.to_dict() if self.primary_structure else None,
            "secondary": self.secondary_structure.to_dict() if self.secondary_structure else None,
        }
        if self.primary_structure:
            p_dict = self.primary_structure.to_dict()
            for k, v in p_dict.items():
                if k not in res:
                    res[k] = v
        return res

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.to_dict().get(item, default)


def _confirmed_turning_points(strokes: Sequence[Stroke]) -> List[Any]:
    st_list = list(strokes)
    if not st_list:
        return []
    pts = [st_list[0].start]
    last_i = len(st_list) - 1
    for i, s in enumerate(st_list):
        if i == last_i:
            continue
        pts.append(s.end)
    return pts


def _calc_dynamic_tolerance(latest_close: float, price_span: float, df: Optional[pd.DataFrame] = None) -> float:
    base_tol = latest_close * 0.009
    if df is not None and len(df) >= 14 and {"high", "low", "close"}.issubset(df.columns):
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean())
        if atr > 0:
            base_tol = max(latest_close * 0.006, min(atr * 0.85, latest_close * 0.015))
    max_cap = max(price_span * 0.025, 1e-4)
    return float(max(min(base_tol, max_cap), 1e-4))


def _classify_pattern(u_slope: float, l_slope: float, w_start: float, w_end: float, flat_thresh: float) -> Tuple[str, str]:
    s_diff = abs(u_slope - l_slope) / max(abs(u_slope), abs(l_slope), 1e-6)
    is_exp = w_end > w_start * 1.15
    is_conv = w_end < w_start * 0.85
    is_par = s_diff < 0.20 and not (is_exp or is_conv)

    u_up, u_down = u_slope > flat_thresh, u_slope < -flat_thresh
    l_up, l_down = l_slope > flat_thresh, l_slope < -flat_thresh

    if is_exp:
        if u_up and l_down:
            return "symmetric_expanding_triangle", "对称扩散三角形"
        elif u_down and l_down:
            return "descending_expanding_triangle", "下降扩散三角形"
        elif u_up and l_up:
            return "ascending_expanding_triangle", "上升扩散三角形"
        return "expanding_triangle", "扩散三角形"

    if is_conv:
        if u_down and l_up:
            return "symmetric_converging_triangle", "对称收敛三角形"
        elif u_up and l_up:
            return "ascending_converging_triangle", "上升收敛三角形"
        elif u_down and l_down:
            return "descending_converging_triangle", "下降收敛三角形"
        return "converging_triangle", "收敛三角形"

    if is_par:
        if u_up and l_up:
            return "ascending_channel", "上升通道"
        elif u_down and l_down:
            return "descending_channel", "下降通道"
        return "horizontal_channel", "水平通道"

    if u_up and l_up:
        return "ascending_rectangle", "上升整理"
    elif u_down and l_down:
        return "descending_rectangle", "下降整理"
    return "horizontal_rectangle", "普通整理"


def _evaluate_rail(
    p1: Any, p2: Any, pool: List[Any], tolerance: float, price_span: float, is_upper: bool
) -> Optional[Dict[str, Any]]:
    x1, y1 = float(p1.raw_index), float(p1.price)
    x2, y2 = float(p2.raw_index), float(p2.price)
    if abs(x2 - x1) < 1e-6:
        return None
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
        p1, p2 = p2, p1

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    touches = []
    errors = []
    breaks = 0

    for pt in pool:
        px = float(pt.raw_index)
        py = float(pt.price)
        if px < x1 - 1e-3:
            continue
        line_y = slope * px + intercept
        diff = abs(py - line_y)

        # 处于容差带内统计触点
        if diff <= tolerance * 1.35:
            touches.append({
                "index": int(px),
                "price": round(py, 2),
                "date": getattr(pt, "date", str(int(px))),
                "kind": "high" if is_upper else "low",
            })
            errors.append(diff / price_span)

        if is_upper and py > line_y + tolerance * 1.6:
            breaks += 1
        elif not is_upper and py < line_y - tolerance * 1.6:
            breaks += 1

    if breaks > 2 or len(touches) < 2:
        return None

    # 多触点拟合微调
    if len(touches) >= 3:
        txs = np.array([t["index"] for t in touches], dtype=float)
        tys = np.array([t["price"] for t in touches], dtype=float)
        A = np.vstack([txs, np.ones(len(txs))]).T
        best_m, best_c = np.linalg.lstsq(A, tys, rcond=None)[0]
        if abs(best_m - slope) < abs(slope) * 0.35 + 1e-4:
            slope, intercept = float(best_m), float(best_c)

    mean_err = float(np.mean(errors)) if errors else 0.05
    return {
        "p1": p1, "p2": p2, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "slope": slope, "intercept": intercept,
        "span": x2 - x1, "touch_count": len(touches), "touches": touches,
        "mean_error": mean_err, "break_count": breaks,
    }


def _fit_window_structure(
    window_name: str,
    tops: List[Any],
    bottoms: List[Any],
    total_bars: int,
    tolerance: float,
    price_span: float,
    real_latest_close: float,
    extend: int,
    min_span: int = 10,
    force_down_slope: bool = False,
) -> Optional[StructureItem]:
    if len(tops) < 2 or len(bottoms) < 2:
        return None

    best_item: Optional[StructureItem] = None
    max_score = -99999.0

    highest_pt = max(tops, key=lambda p: float(p.price)) if tops else None

    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            upper = _evaluate_rail(tops[i], tops[j], tops, tolerance, price_span, is_upper=True)
            if not upper:
                continue

            # 如果要求捕捉中级下降阻力切线，跳过向上的上轨
            if force_down_slope and upper["slope"] >= 0:
                continue

            for m in range(len(bottoms)):
                for n in range(m + 1, len(bottoms)):
                    lower = _evaluate_rail(bottoms[m], bottoms[n], bottoms, tolerance, price_span, is_upper=False)
                    if not lower:
                        continue

                    x_start = min(upper["x1"], lower["x1"])
                    fit_span = max(upper["x2"], lower["x2"]) - x_start
                    total_span = (total_bars - 1) - x_start

                    if fit_span < min_span:
                        continue

                    w_start = (upper["slope"] * x_start + upper["intercept"]) - (lower["slope"] * x_start + lower["intercept"])
                    w_end = (upper["slope"] * (total_bars - 1) + upper["intercept"]) - (lower["slope"] * (total_bars - 1) + lower["intercept"])
                    if w_start <= 0 or w_end <= 0:
                        continue

                    all_touches = upper["touches"] + lower["touches"]
                    all_touches.sort(key=lambda x: x["index"])
                    alternations = sum(1 for k in range(len(all_touches) - 1) if all_touches[k]["kind"] != all_touches[k + 1]["kind"])
                    alt_ratio = alternations / max(len(all_touches) - 1, 1)

                    last_dist = (total_bars - 1) - all_touches[-1]["index"]

                    # 评分模型：高权重倾斜给多触点
                    u_tc = upper["touch_count"]
                    l_tc = lower["touch_count"]
                    s_touch = (
                        (16.0 if u_tc >= 3 else 6.0) + (u_tc - 2) * 8.0 +
                        (16.0 if l_tc >= 3 else 6.0) + (l_tc - 2) * 8.0
                    )

                    s_fit = min(fit_span, 90) / 12.0
                    s_alt = alt_ratio * 4.0
                    s_recency = max(0.0, 8.0 - (last_dist / 10.0) * 3.0)
                    p_error = (upper["mean_error"] + lower["mean_error"]) * 14.0
                    p_break = (upper["break_count"] + lower["break_count"]) * 10.0

                    # 5月14日波段峰值起点强力加分 (25分)
                    is_peak_origin = (highest_pt is not None and abs(upper["x1"] - float(highest_pt.raw_index)) < 1e-3 and upper["slope"] < 0)
                    bonus_peak = 25.0 if is_peak_origin else 0.0

                    score = s_touch + s_fit + s_alt + s_recency + bonus_peak - p_error - p_break

                    if score > max_score:
                        max_score = score
                        flat_th = (price_span / max(total_bars, 100)) * 0.12
                        ptype, plabel = _classify_pattern(upper["slope"], lower["slope"], w_start, w_end, flat_th)

                        cur_x = float(total_bars - 1)
                        cur_up = upper["slope"] * cur_x + upper["intercept"]
                        cur_lo = lower["slope"] * cur_x + lower["intercept"]
                        break_margin = tolerance * 1.3
                        b_up = cur_up + break_margin
                        b_lo = cur_lo - break_margin

                        if real_latest_close > b_up:
                            st, reason = "invalidated", f"收盘({real_latest_close:.2f})突破确认位({b_up:.2f})"
                        elif real_latest_close < b_lo:
                            st, reason = "invalidated", f"收盘({real_latest_close:.2f})跌破确认位({b_lo:.2f})"
                        elif upper["touch_count"] >= 2 and lower["touch_count"] >= 2:
                            st, reason = "confirmed", f"触点充分(上{upper['touch_count']}/下{lower['touch_count']})共振"
                        else:
                            st, reason = "candidate", f"候选结构(跨度{int(fit_span)}根)"

                        x_end = float(cur_x + extend)
                        y_u_end = upper["slope"] * x_end + upper["intercept"]
                        y_l_end = lower["slope"] * x_end + lower["intercept"]
                        r_res = (upper["x1"], upper["y1"], x_end, y_u_end)
                        r_sup = (lower["x1"], lower["y1"], x_end, y_l_end)

                        conf = round(min(max((score + 5.0) / 40.0, 0.50), 0.98), 2)
                        up_dict = {
                            "x1": int(upper["x1"]), "y1": round(float(upper["y1"]), 2),
                            "x2": int(upper["x2"]), "y2": round(float(upper["y2"]), 2),
                            "slope": float(upper["slope"]), "intercept": float(upper["intercept"]),
                            "touches": upper["touches"], "touch_count": upper["touch_count"],
                            "break_count": upper["break_count"],
                        }
                        lo_dict = {
                            "x1": int(lower["x1"]), "y1": round(float(lower["y1"]), 2),
                            "x2": int(lower["x2"]), "y2": round(float(lower["y2"]), 2),
                            "slope": float(lower["slope"]), "intercept": float(lower["intercept"]),
                            "touches": lower["touches"], "touch_count": lower["touch_count"],
                            "break_count": lower["break_count"],
                        }

                        best_item = StructureItem(
                            pattern_type=ptype,
                            label=plabel,
                            status=st,
                            status_reason=reason,
                            confidence=conf,
                            level="stroke",
                            source_window=window_name,
                            score=score,
                            fit_span=int(fit_span),
                            total_span=int(total_span),
                            last_touch_distance=int(last_dist),
                            alternation_ratio=alt_ratio,
                            actual_latest_close=round(real_latest_close, 2),
                            touch_tolerance=round(tolerance, 2),
                            width_start=round(float(w_start), 2),
                            width_end=round(float(w_end), 2),
                            current_upper=round(float(cur_up), 2),
                            current_lower=round(float(cur_lo), 2),
                            dist_to_upper_pct=round((cur_up - real_latest_close) / real_latest_close * 100, 2),
                            dist_to_lower_pct=round((real_latest_close - cur_lo) / real_latest_close * 100, 2),
                            breakout_up_level=round(float(b_up), 2),
                            breakdown_level=round(float(b_lo), 2),
                            upper_line=up_dict,
                            lower_line=lo_dict,
                            touch_count=len(all_touches),
                            break_count=upper["break_count"] + lower["break_count"],
                            points_detail=all_touches,
                            reason=f"[{window_name}] {plabel}({st})：总触点{len(all_touches)}次，跨度{int(fit_span)}根；{reason}。",
                            support_rail=r_sup,
                            resistance_rail=r_res,
                        )

    return best_item


def find_channel(
    strokes: Sequence[Stroke],
    total_bars: int,
    n_points: int = 18,
    extend: int = DEFAULT_EXTEND,
    latest_close: Optional[float] = None,
    df: Optional[pd.DataFrame] = None,
    level: str = "stroke",
    period_key: str = "D",
    min_period: Optional[str] = None,
    lookback_mode: str = "深度",
) -> Optional[TrendChannel]:
    turning = _confirmed_turning_points(strokes)
    if len(turning) < 4:
        return None

    if latest_close is None or latest_close <= 0:
        if df is not None and not df.empty and "close" in df.columns:
            real_latest_close = float(df["close"].iloc[-1])
        else:
            real_latest_close = float(turning[-1].price)
    else:
        real_latest_close = float(latest_close)

    extend = clamp_extend(extend)
    prices = [float(p.price) for p in turning]
    price_span = max(prices) - min(prices) if prices else 1.0
    tolerance = _calc_dynamic_tolerance(real_latest_close, price_span, df)

    # 1. 核心关键修复：严格切断 100 根 K 线之前的远古端点（彻底丢弃 2024/2025 年无用端点）
    # 限制中期波段端点池在最近 90 根K线内，正好覆盖 2026 年 5 月 14 日（4258.86）至今
    cur_x = total_bars - 1
    recent_turning = [p for p in turning if (cur_x - float(p.raw_index)) <= 92]
    if len(recent_turning) < 6:
        recent_turning = turning[-14:] if len(turning) >= 14 else turning[:]

    m_tops = [p for p in recent_turning if p.kind == TOP]
    m_bottoms = [p for p in recent_turning if p.kind == BOTTOM]

    # 主结构：锁定 5 月 14 日以来的中期下降压力切线（force_down_slope=True 强制过滤反向虚线）
    primary = _fit_window_structure(
        "中期主结构", m_tops, m_bottoms, total_bars, tolerance, price_span, real_latest_close, extend,
        min_span=18, force_down_slope=True
    )

    # 2. 短期局部结构：锁定 7 月底 3740 至今的近端整理（最近 40 根 K 线）
    short_turning = [p for p in turning if (cur_x - float(p.raw_index)) <= 42]
    if len(short_turning) < 4:
        short_turning = turning[-8:] if len(turning) >= 8 else turning[:]

    s_tops = [p for p in short_turning if p.kind == TOP]
    s_bottoms = [p for p in short_turning if p.kind == BOTTOM]

    secondary = _fit_window_structure(
        "短期局部", s_tops, s_bottoms, total_bars, tolerance, price_span, real_latest_close, extend,
        min_span=6, force_down_slope=False
    )

    # 保底：若没找到严格向下的主结构，则采用最佳候选
    if not primary:
        primary = _fit_window_structure(
            "中期主结构", m_tops, m_bottoms, total_bars, tolerance, price_span, real_latest_close, extend,
            min_span=15, force_down_slope=False
        )

    if not primary and not secondary:
        return None

    if not primary:
        primary = secondary
        secondary = None

    channel = TrendChannel(
        support=primary.support_rail,
        resistance=primary.resistance_rail,
        support_points=[(int(p["index"]), float(p["price"])) for p in primary.lower_line.get("touches", [])],
        resistance_points=[(int(p["index"]), float(p["price"])) for p in primary.upper_line.get("touches", [])],
        slope_support=float(primary.lower_line.get("slope", 0.0)),
        slope_resistance=float(primary.upper_line.get("slope", 0.0)),
        extend=extend,
        primary_structure=primary,
        secondary_structure=secondary,
        pattern_type=primary.pattern_type,
        label=primary.label,
        status=primary.status,
        confidence=primary.confidence,
        level=primary.level,
        actual_latest_close=primary.actual_latest_close,
        reason=primary.reason,
    )
    return channel
