"""画笔（缠论第三步）。

标准规则
--------
1. 笔的两端必须是**一顶一底**（方向相反）。
2. 两个分型之间（不含分型自身占据的K线）**至少要有 1 根独立K线**，
   等价于两个分型中心在合并K线序列上的间隔 ``>= 4``。
3. 顶必须高于底，且中途价格不能被反向分型破坏。

实现采用"贪心转折点"算法：
- 遇到同类型分型 → 保留更极端的那个（顶取更高、底取更低）；
- 遇到反类型分型且间隔达标 → 确认为新转折点，成笔；
- 间隔不达标 → 丢弃该分型（视为噪声）。
"""

from __future__ import annotations

from typing import List, Sequence

from .models import BOTTOM, TOP, Fractal, Stroke

#: 两分型中心的最小合并K线间隔（>=4 表示中间至少隔 1 根独立K线）
MIN_GAP = 4


def _more_extreme(candidate: Fractal, current: Fractal) -> bool:
    """同类型分型中，candidate 是否比 current 更极端。"""
    if candidate.kind == TOP:
        return candidate.price > current.price
    return candidate.price < current.price


def build_strokes(
    fractals: Sequence[Fractal],
    min_gap: int = MIN_GAP,
) -> List[Stroke]:
    """由分型序列构造笔序列。"""
    if not fractals:
        return []

    # ---------- 第一步：筛出有效转折点 ----------
    turning: List[Fractal] = [fractals[0]]
    for frac in fractals[1:]:
        last = turning[-1]

        if frac.kind == last.kind:
            # 同类型，保留更极端的
            if _more_extreme(frac, last):
                turning[-1] = frac
            continue

        # 反类型，检查间隔是否达标
        if frac.index - last.index >= min_gap:
            turning.append(frac)
        # 间隔不足则丢弃该分型

    # ---------- 第二步：相邻转折点两两成笔 ----------
    strokes: List[Stroke] = []
    for i in range(len(turning) - 1):
        a, b = turning[i], turning[i + 1]
        if a.kind == b.kind:
            continue  # 理论上不会发生，防御性跳过
        direction = "up" if b.price > a.price else "down"
        strokes.append(Stroke(start=a, end=b, direction=direction, index=len(strokes)))

    return strokes


def strokes_to_points(strokes: Sequence[Stroke]) -> List[tuple]:
    """把笔序列展开成折线坐标点 ``[(raw_index, price), ...]``，供绘图使用。"""
    pts: List[tuple] = []
    for s in strokes:
        if not pts:
            pts.append((s.start.raw_index, s.start.price))
        pts.append((s.end.raw_index, s.end.price))
    return pts
