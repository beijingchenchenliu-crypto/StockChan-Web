"""缠论引擎的数据结构。

术语约定
--------
- ``index``       : 合并K线序列中的序号（缠论分析的坐标轴）
- ``raw_index``   : 原始日线 DataFrame 中的行号（绘图的坐标轴）
- ``kind``        : ``"top"`` 顶分型 / ``"bottom"`` 底分型
- ``direction``   : ``1`` 向上 / ``-1`` 向下
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

#: 顶分型
TOP = "top"
#: 底分型
BOTTOM = "bottom"


@dataclass
class MergedBar:
    """经过包含关系处理后的"干净"K线。"""

    index: int          #: 合并后在序列中的序号
    high: float
    low: float
    start: int          #: 覆盖的原始K线起始行号
    end: int            #: 覆盖的原始K线结束行号
    high_idx: int       #: 最高价所在的原始行号
    low_idx: int        #: 最低价所在的原始行号
    direction: int = 0  #: 相对前一根的方向，1 向上 / -1 向下

    @property
    def span(self) -> int:
        """该合并K线覆盖了多少根原始K线。"""
        return self.end - self.start + 1

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (f"MergedBar(#{self.index} H={self.high:.2f} L={self.low:.2f} "
                f"raw=[{self.start},{self.end}] dir={self.direction})")


@dataclass
class Fractal:
    """分型。"""

    index: int          #: 分型中间那根合并K线的序号
    kind: str           #: TOP / BOTTOM
    price: float        #: 分型的极值价（顶=最高价，底=最低价）
    raw_index: int      #: 极值所在的原始行号
    date: str = ""      #: 极值所在日期（字符串，便于直接标注）

    def __repr__(self) -> str:  # pragma: no cover
        return f"Fractal({self.kind} @{self.date} {self.price:.2f} idx={self.index})"


@dataclass
class Stroke:
    """笔：由一顶一底两个分型连成。"""

    start: Fractal
    end: Fractal
    direction: str  #: "up" / "down"
    index: int = 0  #: 在笔序列中的序号

    @property
    def high(self) -> float:
        return max(self.start.price, self.end.price)

    @property
    def low(self) -> float:
        return min(self.start.price, self.end.price)

    @property
    def amplitude(self) -> float:
        return abs(self.end.price - self.start.price)

    @property
    def raw_start(self) -> int:
        return self.start.raw_index

    @property
    def raw_end(self) -> int:
        return self.end.raw_index

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Stroke(#{self.index} {self.direction} "
                f"{self.start.price:.2f}->{self.end.price:.2f})")


@dataclass
class Pivot:
    """中枢：至少三笔重叠构成的区间 [zd, zg]。"""

    zg: float               #: 中枢上沿
    zd: float               #: 中枢下沿
    start_index: int        #: 起始合并K线序号
    end_index: int          #: 结束合并K线序号
    raw_start: int          #: 起始原始行号
    raw_end: int            #: 结束原始行号
    strokes: List[Stroke] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.zg - self.zd

    @property
    def mid(self) -> float:
        return (self.zg + self.zd) / 2.0

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Pivot([{self.zd:.2f}, {self.zg:.2f}] "
                f"raw=[{self.raw_start},{self.raw_end}] n={len(self.strokes)})")


@dataclass
class TradePoint:
    """缠论买卖点（一/二/三类）。"""

    index: int              #: 合并K线序号
    raw_index: int          #: 原始行号
    price: float
    date: str
    side: str               #: "buy" / "sell"
    level: int              #: 1 / 2 / 3
    tentative: bool = False  #: True = 观察态（由最后一笔产生，可能被后续行情改写）
    reason: str = ""        #: 判定依据，便于人工复核
    stroke_index: int = -1  #: 所属笔在笔序列中的序号

    @property
    def label(self) -> str:
        """图上标注文字，如 ``1b`` / ``2s``（小写，TradingView 风格）。"""
        return f"{self.level}{'b' if self.side == 'buy' else 's'}"

    @property
    def confirmed(self) -> bool:
        """是否处于确定态。"""
        return not self.tentative

    @property
    def display(self) -> str:
        """带状态后缀的标注，观察态加 ``?``。"""
        return self.label if self.confirmed else f"{self.label}?"

    def __repr__(self) -> str:  # pragma: no cover
        state = "确定" if self.confirmed else "观察"
        return f"TradePoint({self.display} @{self.date} {self.price:.2f} {state})"


@dataclass
class WavePoint:
    """波浪标记点（初步）。"""

    index: int          #: 合并K线序号
    raw_index: int      #: 原始行号
    price: float
    label: str          #: "1".."5" / "A" / "B" / "C"
    kind: str = ""      #: TOP / BOTTOM
    date: str = ""
    degree: str = "primary"  #: 波浪级别，暂统一为 primary

    def __repr__(self) -> str:  # pragma: no cover
        return f"WavePoint({self.label} @{self.date} {self.price:.2f})"
