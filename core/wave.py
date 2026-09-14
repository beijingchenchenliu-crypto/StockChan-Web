"""波浪形态初步标记（基于笔序列的启发式）。

思路
----
缠论的"笔"天然就是一段段推动/调整，直接把笔的转折点当作波浪的摆动点，
在摆动点序列上扫描符合**推动浪（1-2-3-4-5）**基本规则的窗口，命中后
把随后的三笔标为**调整浪（A-B-C）**。

初步实现遵守的规则
------------------
1. 2 浪不跌破 1 浪起点（向上）/ 不升破 1 浪起点（向下）
2. 3 浪不是 1、3、5 中最短的一浪
3. 5 浪创新高（向上）/ 创新低（向下）
4. （可选，``strict=True``）4 浪不与 1 浪价格区间重叠

由于笔的级别远小于经典波浪级别，规则 4 在实盘中经常被破坏，因此默认
**不启用**严格模式，以保证标记的可用性。

⚠️ 这里输出的是"初步标记"，用于辅助观察，不构成交易建议。
"""

from __future__ import annotations

from typing import List, Sequence

from .models import Stroke, WavePoint

#: 波浪标签顺序
IMPULSE_LABELS = ["1", "2", "3", "4", "5"]
CORRECTION_LABELS = ["A", "B", "C"]


def _turning_points(strokes: Sequence[Stroke]):
    """把笔序列展开为转折点列表。"""
    pts = []
    if not strokes:
        return pts
    pts.append(strokes[0].start)
    for s in strokes:
        pts.append(s.end)
    return pts


def _is_impulse(p, strict: bool = False) -> bool:
    """判断 6 个转折点（5 段）是否构成一个推动浪。"""
    if len(p) < 6:
        return False

    prices = [x.price for x in p]
    p0, p1, p2, p3, p4, p5 = prices
    up = p1 > p0

    if up:
        if not (p1 > p0 and p2 > p0 and p3 > p1 and p4 > p1 and p5 > p3):
            return False
        if strict and p4 <= p1:          # 4 浪不得进入 1 浪区间
            return False
    else:
        if not (p1 < p0 and p2 < p0 and p3 < p1 and p4 < p1 and p5 < p3):
            return False
        if strict and p4 >= p1:
            return False

    w1 = abs(p1 - p0)
    w3 = abs(p3 - p2)
    w5 = abs(p5 - p4)
    if w3 <= min(w1, w5):                 # 3 浪不能是最短
        return False

    return True


def _is_correction(impulse_up: bool, a, b, c) -> bool:
    """对 A-B-C 做一个宽松的形态校验。"""
    if impulse_up:
        # 下跌调整：A 下、B 反弹但不过 5 浪高点、C 破 A 低点
        return a.price < b.price and c.price < b.price and b.price <= max(a.price, c.price) * 1.15
    return a.price > b.price and c.price > b.price and b.price >= min(a.price, c.price) * 0.85


def label_waves(strokes: Sequence[Stroke], strict: bool = False) -> List[WavePoint]:
    """在笔序列上做波浪初步标记。

    :param strokes: 笔序列
    :param strict:  是否启用严格的"4 浪不重叠"规则
    :return:        :class:`WavePoint` 列表（按时间升序）
    """
    pts = _turning_points(strokes)
    n = len(pts)
    if n < 6:
        return []

    waves: List[WavePoint] = []
    i = 0
    while i + 5 < n:
        window = pts[i:i + 6]
        if not _is_impulse(window, strict=strict):
            i += 1
            continue

        # ---------- 标记 1~5 浪 ----------
        for k, label in enumerate(IMPULSE_LABELS):
            pt = window[k + 1]
            waves.append(WavePoint(
                index=pt.index, raw_index=pt.raw_index, price=pt.price,
                label=label, kind=pt.kind, date=pt.date,
            ))

        impulse_up = window[1].price > window[0].price
        consumed = 6

        # ---------- 尝试标记 A-B-C ----------
        if i + 8 < n:
            a, b, c = pts[i + 6], pts[i + 7], pts[i + 8]
            if _is_correction(impulse_up, a, b, c):
                for label, pt in zip(CORRECTION_LABELS, (a, b, c)):
                    waves.append(WavePoint(
                        index=pt.index, raw_index=pt.raw_index, price=pt.price,
                        label=label, kind=pt.kind, date=pt.date, degree="corrective",
                    ))
                consumed = 9

        i += consumed

    return waves


def wave_segments(waves: Sequence[WavePoint]) -> List[tuple]:
    """把波浪点转成绘图用的折线点 ``[(raw_index, price), ...]``。"""
    return [(w.raw_index, w.price) for w in waves]
