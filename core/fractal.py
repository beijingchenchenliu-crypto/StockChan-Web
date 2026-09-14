"""分型识别（缠论第二步）。

在**合并后**的K线序列上，连续三根 K1/K2/K3：

- **顶分型**：K2 的高点最高 且 K2 的低点也最高
- **底分型**：K2 的低点最低 且 K2 的高点也最低

由于合并后已不存在包含关系，"高点最高"与"低点最高"等价，但这里仍然
两个条件都校验，保证逻辑自洽、可读。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .models import BOTTOM, TOP, Fractal, MergedBar


def _date_at(dates: Optional[Sequence[str]], raw_index: int) -> str:
    if dates is None:
        return ""
    if 0 <= raw_index < len(dates):
        return str(dates[raw_index])
    return ""


def find_fractals(
    bars: Sequence[MergedBar],
    dates: Optional[Sequence[str]] = None,
) -> List[Fractal]:
    """在合并K线序列上找出所有分型。

    :param bars:  合并后的K线列表
    :param dates: 原始日线日期序列（与 ``raw_index`` 对应），用于标注
    """
    fractals: List[Fractal] = []
    n = len(bars)
    if n < 3:
        return fractals

    for i in range(1, n - 1):
        left, mid, right = bars[i - 1], bars[i], bars[i + 1]

        is_top = (mid.high > left.high and mid.high > right.high
                  and mid.low > left.low and mid.low > right.low)
        is_bottom = (mid.low < left.low and mid.low < right.low
                     and mid.high < left.high and mid.high < right.high)

        if is_top:
            fractals.append(Fractal(
                index=i, kind=TOP, price=float(mid.high),
                raw_index=int(mid.high_idx), date=_date_at(dates, mid.high_idx),
            ))
        elif is_bottom:
            fractals.append(Fractal(
                index=i, kind=BOTTOM, price=float(mid.low),
                raw_index=int(mid.low_idx), date=_date_at(dates, mid.low_idx),
            ))

    return fractals
