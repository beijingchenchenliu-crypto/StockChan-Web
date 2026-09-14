"""中枢识别（缠论第四步）。

定义
----
连续 **三笔** 的价格区间存在公共重叠部分，即构成中枢：

    ZG = min(三笔各自的最高点)
    ZD = max(三笔各自的最低点)
    要求 ZG > ZD

延伸
----
中枢区间 ``[ZD, ZG]`` 由**前三笔**一次性确定，之后**不再收缩**。
后续每一笔若其**终点**仍落在 ``[ZD, ZG]`` 之内，说明还在中枢震荡 → 并入中枢；
一旦某笔的终点跑出区间（向上突破 ZG 或向下跌破 ZD），该笔即为"离开笔"，
中枢到此结束。

> 关键点：离开笔**不属于**中枢。否则突破行情会被吞进中枢、把 ZG/ZD 压扁，
> 后续的三类买卖点就永远判不出来。

这是"初步"实现：不做次级别递归、不做中枢扩张/新生区分，但足以支撑
可视化与后续迭代。
"""

from __future__ import annotations

from typing import List, Sequence

from .models import Pivot, Stroke


def find_pivots(strokes: Sequence[Stroke], min_strokes: int = 3) -> List[Pivot]:
    """从笔序列中识别中枢。"""
    pivots: List[Pivot] = []
    n = len(strokes)
    if n < min_strokes:
        return pivots

    i = 0
    while i + min_strokes - 1 < n:
        group = list(strokes[i:i + min_strokes])
        zg = min(s.high for s in group)
        zd = max(s.low for s in group)

        if zg <= zd:
            # 三笔无公共重叠，起点右移一笔继续找
            i += 1
            continue

        # ---------- 延伸：终点仍落在区间内才算震荡 ----------
        j = i + min_strokes
        while j < n and zd <= strokes[j].end.price <= zg:
            group.append(strokes[j])
            j += 1
        # 此处 strokes[j]（若存在）即"离开笔"，不属于本中枢

        pivots.append(Pivot(
            zg=float(zg),
            zd=float(zd),
            start_index=int(group[0].start.index),
            end_index=int(group[-1].end.index),
            raw_start=int(group[0].start.raw_index),
            raw_end=int(group[-1].end.raw_index),
            strokes=group,
        ))
        i = j

    return pivots


def strokes_in_pivot(pivot: Pivot) -> int:
    """中枢包含的笔数量。"""
    return len(pivot.strokes)
