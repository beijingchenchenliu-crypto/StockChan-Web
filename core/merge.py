"""包含关系合并（缠论第一步）。

规则
----
相邻两根K线，若一根的最高最低完全覆盖另一根，则存在**包含关系**，需要合并：

- 合并方向由**前一根合并K线**决定（向上/向下）。
- 向上处理：新高取两者最高价的**较大值**，新低取两者最低价的**较大值**。
- 向下处理：新高取两者最高价的**较小值**，新低取两者最低价的**较小值**。

合并后产生的新K线可能与更前面的一根再次构成包含关系，因此这里用
``while`` 反复回退合并，直到最后两根不再包含为止（比常见的一次性前向
扫描更严谨）。
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .models import MergedBar


def _has_inclusion(a: MergedBar, b: MergedBar) -> bool:
    """a、b 之间是否存在包含关系。"""
    return (a.high >= b.high and a.low <= b.low) or (a.high <= b.high and a.low >= b.low)


def _merge_two(a: MergedBar, b: MergedBar, direction: int) -> MergedBar:
    """按 ``direction`` 把 a、b 合并成一根新K线。"""
    if direction >= 0:  # 向上：高高取高，低低取高
        new_high, new_high_idx = (a.high, a.high_idx) if a.high >= b.high else (b.high, b.high_idx)
        new_low, new_low_idx = (a.low, a.low_idx) if a.low >= b.low else (b.low, b.low_idx)
    else:               # 向下：高高取低，低低取低
        new_high, new_high_idx = (a.high, a.high_idx) if a.high <= b.high else (b.high, b.high_idx)
        new_low, new_low_idx = (a.low, a.low_idx) if a.low <= b.low else (b.low, b.low_idx)

    return MergedBar(
        index=a.index,
        high=float(new_high),
        low=float(new_low),
        start=min(a.start, b.start),
        end=max(a.end, b.end),
        high_idx=int(new_high_idx),
        low_idx=int(new_low_idx),
        direction=direction,
    )


def _direction_at(bars: List[MergedBar], pos: int) -> int:
    """判断 ``bars[pos]`` 相对前一根的方向；没有前一根时默认向上。"""
    if pos <= 0:
        return 1
    return 1 if bars[pos].high > bars[pos - 1].high else -1


def merge_inclusion(df: pd.DataFrame) -> List[MergedBar]:
    """对原始日线做包含关系合并。

    :param df: 至少包含 ``high`` / ``low`` 两列的 DataFrame，按时间升序
    :return:   合并后的 :class:`MergedBar` 列表
    """
    if df is None or len(df) == 0:
        return []

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    bars: List[MergedBar] = []
    for p in range(len(highs)):
        bars.append(MergedBar(
            index=len(bars),
            high=float(highs[p]),
            low=float(lows[p]),
            start=p, end=p, high_idx=p, low_idx=p, direction=0,
        ))

        # 新K线可能与前一根包含，合并后还可能再包含，需循环处理
        while len(bars) >= 2 and _has_inclusion(bars[-2], bars[-1]):
            prev, cur = bars[-2], bars[-1]
            direction = _direction_at(bars, len(bars) - 2)
            merged = _merge_two(prev, cur, direction)
            bars.pop()
            bars.pop()
            bars.append(merged)

    # 统一重新编号并补全方向
    for i, bar in enumerate(bars):
        bar.index = i
        bar.direction = 1 if i == 0 else (1 if bar.high > bars[i - 1].high else -1)

    return bars
