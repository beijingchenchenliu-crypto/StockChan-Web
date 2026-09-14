"""缠论买卖点判定（第一 / 二 / 三类）。

判定规则
--------

**一类买点 1B**（底背驰）
    下跌笔跌破前一个中枢下沿 ``ZD``（或无中枢时创出比前一同向下跌笔更低的低点），
    但该笔的 MACD 绿柱面积显著小于前一同向下跌笔 → 力度衰减 = 背驰。
    卖点 1S 完全对称（突破 ``ZG`` / 创新高 + 顶背驰）。

**二类买点 2B**
    一类买点之后，第一次次级别回抽（向下笔）**不创新低**，即回抽低点高于 1B 低点。
    卖点 2S 对称（反弹不创新高）。

**三类买点 3B**
    一笔强势向上离开中枢（笔终点 > ``ZG``），随后一笔回抽**不跌破 ZG** → 三买。
    卖点 3S 对称（跌破 ``ZD`` 后反抽不涨破 ``ZD``）。

确定态 / 观察态
--------------
所有买卖点都由某一笔的终点产生。**最后一笔**尚未被后续行情确认（可能被更极端的
分型延长、也可能被反向分型改写），因此由最后一笔产生的买卖点标记为
``tentative=True``（观察态）；其余为确定态。

参数
----
``ratio``：背驰阈值。后一笔力度 < 前一笔力度 × ratio 才认为背驰，默认 0.85。
调大（如 0.95）→ 信号更多更灵敏；调小（如 0.7）→ 只保留显著背驰。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .indicator import StrokePower
from .models import Pivot, Stroke, TradePoint

#: 默认背驰力度衰减阈值
DIVERGENCE_RATIO = 0.85


def _pivot_before(pivots: Sequence[Pivot], stroke: Stroke) -> Optional[Pivot]:
    """返回该笔开始之前最近的一个中枢。"""
    found: Optional[Pivot] = None
    for pv in pivots:
        if pv.raw_end <= stroke.raw_start:
            found = pv
        else:
            break
    return found


def find_trade_points(
    strokes: Sequence[Stroke],
    pivots: Sequence[Pivot],
    powers: Sequence[StrokePower],
    ratio: float = DIVERGENCE_RATIO,
) -> List[TradePoint]:
    """识别一/二/三类买卖点。

    :param strokes: 笔序列
    :param pivots:  中枢序列
    :param powers:  与 ``strokes`` 等长的 MACD 力度序列
    :param ratio:   背驰力度衰减阈值
    :return:        按时间升序排列的买卖点列表
    """
    strokes = list(strokes)
    pivots = list(pivots)
    powers = list(powers)

    n = len(strokes)
    if n == 0 or len(powers) != n:
        return []

    last_idx = n - 1
    points: List[TradePoint] = []

    def add(stroke: Stroke, side: str, level: int, reason: str) -> None:
        points.append(TradePoint(
            index=stroke.end.index,
            raw_index=stroke.end.raw_index,
            price=float(stroke.end.price),
            date=stroke.end.date,
            side=side,
            level=level,
            tentative=(stroke.index == last_idx),
            reason=reason,
            stroke_index=stroke.index,
        ))

    # ------------------------------------------------------------------ #
    # 1. 一类买卖点：MACD 面积背驰
    # ------------------------------------------------------------------ #
    prev_down: Optional[tuple] = None  # (Stroke, StrokePower)
    prev_up: Optional[tuple] = None

    for s in strokes:
        p = powers[s.index]

        if s.direction == "down":
            if prev_down is not None:
                prev_s, prev_p = prev_down
                pv = _pivot_before(pivots, s)

                broke_pivot = pv is not None and s.end.price < pv.zd
                new_low = s.end.price < prev_s.end.price

                if ((broke_pivot or new_low) and p.area > 0 and prev_p.area > 0
                        and p.area < prev_p.area * ratio):
                    where = f"跌破中枢ZD={pv.zd:.2f}" if broke_pivot else "创出新的低点"
                    add(s, "buy", 1,
                        f"底背驰：{where}，MACD绿柱面积 {p.area:.1f} < 前笔 {prev_p.area:.1f}"
                        f"（{p.area / prev_p.area:.0%}）")
            prev_down = (s, p)

        else:  # up
            if prev_up is not None:
                prev_s, prev_p = prev_up
                pv = _pivot_before(pivots, s)

                broke_pivot = pv is not None and s.end.price > pv.zg
                new_high = s.end.price > prev_s.end.price

                if ((broke_pivot or new_high) and p.area > 0 and prev_p.area > 0
                        and p.area < prev_p.area * ratio):
                    where = f"突破中枢ZG={pv.zg:.2f}" if broke_pivot else "创出新的高点"
                    add(s, "sell", 1,
                        f"顶背驰：{where}，MACD红柱面积 {p.area:.1f} < 前笔 {prev_p.area:.1f}"
                        f"（{p.area / prev_p.area:.0%}）")
            prev_up = (s, p)

    # ------------------------------------------------------------------ #
    # 1.5 最新笔观察态预警
    # ------------------------------------------------------------------ #
    # 最后一笔尚未经过下一笔反向确认时，严格的一类条件可能尚未完全成立。
    # 对“创新高/低 + 力度衰减”的最新笔给出观察态，避免界面只剩历史确定信号。
    # 该点明确标记 tentative=True，不会冒充已确认买卖点。
    if n >= 2:
        latest = strokes[-1]
        latest_power = powers[latest.index]
        previous_same = next((s for s in reversed(strokes[:-1])
                              if s.direction == latest.direction), None)
        if previous_same is not None:
            previous_power = powers[previous_same.index]
            weak = (latest_power.area > 0 and previous_power.area > 0
                    and latest_power.area <= previous_power.area * min(1.10, ratio * 1.30))
            if weak:
                if latest.direction == "down" and latest.end.price < previous_same.end.price:
                    add(latest, "buy", 1,
                        f"观察预警：最新下跌笔创新低，MACD力度衰减（{latest_power.area / previous_power.area:.0%}），等待后续反向笔确认")
                elif latest.direction == "up" and latest.end.price > previous_same.end.price:
                    add(latest, "sell", 1,
                        f"观察预警：最新上涨笔创新高，MACD力度衰减（{latest_power.area / previous_power.area:.0%}），等待后续反向笔确认")

    # ------------------------------------------------------------------ #
    # 2. 二类买卖点：一买/一卖后的回抽不创新低/新高
    # ------------------------------------------------------------------ #
    for tp in list(points):
        if tp.level != 1:
            continue
        k = tp.stroke_index
        if k + 2 >= n:
            continue

        back = strokes[k + 2]  # 一买后：涨一笔 → 回调一笔
        if tp.side == "buy" and back.direction == "down" and back.end.price > tp.price:
            add(back, "buy", 2,
                f"一买后回抽不创新低（{back.end.price:.2f} > 1B {tp.price:.2f}）")
        elif tp.side == "sell" and back.direction == "up" and back.end.price < tp.price:
            add(back, "sell", 2,
                f"一卖后反弹不创新高（{back.end.price:.2f} < 1S {tp.price:.2f}）")

    # ------------------------------------------------------------------ #
    # 3. 三类买卖点：离开中枢后回抽不回中枢
    #
    #    中枢的 ZG/ZD 由前三笔固定，中枢在"终点跑出区间"的那一笔处结束，
    #    因此 strokes[k+1] 就是离开笔，strokes[k+2] 是随后的回抽。
    # ------------------------------------------------------------------ #
    for pv in pivots:
        if not pv.strokes:
            continue
        k = pv.strokes[-1].index
        if k + 2 >= n:
            continue

        out = strokes[k + 1]    # 离开中枢的那一笔
        back = strokes[k + 2]   # 随后的回抽

        if (out.direction == "up" and out.end.price > pv.zg
                and back.direction == "down" and back.end.price > pv.zg):
            add(back, "buy", 3,
                f"强势突破中枢后回抽不破 ZG={pv.zg:.2f}（回抽低点 {back.end.price:.2f}）")

        elif (out.direction == "down" and out.end.price < pv.zd
                and back.direction == "up" and back.end.price < pv.zd):
            add(back, "sell", 3,
                f"强势跌破中枢后反抽不破 ZD={pv.zd:.2f}（反抽高点 {back.end.price:.2f}）")

    # ------------------------------------------------------------------ #
    # 去重 + 排序
    # ------------------------------------------------------------------ #
    best: dict = {}
    for tp in points:
        key = (tp.raw_index, tp.side)
        cur = best.get(key)
        # 同一根K线同一方向只保留一个，级别数字小的更优先（1B > 2B > 3B）
        if cur is None or tp.level < cur.level:
            best[key] = tp

    return sorted(best.values(), key=lambda t: (t.raw_index, t.side))
