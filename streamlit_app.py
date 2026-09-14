# -*- coding: utf-8 -*-
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from core.channel import line_value
from .style import chart_theme, resolve_mode

#: 以下配色随主题（明 / 暗）变化，因此是**模块级可变全局**，
#: 由 :func:`set_theme_mode` 统一更新。
#: 主窗口必须通过 ``kline.SIGNAL_BUY_COLOR`` 这种方式读取，不能
#: ``from .kline_widget import SIGNAL_BUY_COLOR`` —— 后者会把值在 import
#: 时绑定死，主题切换后表格颜色不会跟着变。
SIGNAL_BUY_COLOR = "#FF453A"    # 买点鲜红
SIGNAL_SELL_COLOR = "#30D158"   # 卖点森林绿

#: 观察态（潜在买卖点）在表格里的高亮底色与文字色。
LIVE_ROW_BG = QtGui.QColor(255, 214, 10, 40)
OBSERVE_TEXT_COLOR = "#FFD60A"

#: 图表底板与坐标轴配色
CHART_BG = "#18181B"
AXIS_PEN_COLOR = "#5A5A5F"
AXIS_TEXT_COLOR = "#E5E5EA"
CROSSHAIR_COLOR = "#8E8E93"
STROKE_COLOR = "#A0A0A6"
ZERO_LINE_COLOR = "#5A5A5F"
GRID_ALPHA = 0.06

#: 蜡烛实体色
CANDLE_UP_COLOR = "#FF453A"
CANDLE_DOWN_COLOR = "#26D0B8"

#: 当前生效的模式（``dark`` / ``light``）
_THEME_MODE = "dark"


def set_theme_mode(mode: str) -> str:
    """切换图表模块级配色，返回实际生效的模式。

    pyqtgraph 画布不认 QSS，所以主题切换时除了 :meth:`KLineChart.apply_theme`
    改背景 / 网格 / 坐标轴，还需要把买卖点、蜡烛、观察态高亮这些散落在各处
    的色值一并换掉 —— 全部集中在这里，避免遗漏。
    """
    global SIGNAL_BUY_COLOR, SIGNAL_SELL_COLOR, LIVE_ROW_BG, OBSERVE_TEXT_COLOR
    global CHART_BG, AXIS_PEN_COLOR, AXIS_TEXT_COLOR, CROSSHAIR_COLOR
    global STROKE_COLOR, ZERO_LINE_COLOR, GRID_ALPHA
    global CANDLE_UP_COLOR, CANDLE_DOWN_COLOR, _THEME_MODE

    resolved = resolve_mode(mode)
    colors = chart_theme(resolved)

    _THEME_MODE = resolved
    CHART_BG = str(colors["chart_bg"])
    GRID_ALPHA = float(colors["grid_alpha"])
    AXIS_PEN_COLOR = str(colors["axis_pen"])
    AXIS_TEXT_COLOR = str(colors["axis_text"])
    CROSSHAIR_COLOR = str(colors["crosshair"])
    STROKE_COLOR = str(colors["stroke"])
    ZERO_LINE_COLOR = str(colors["zero_line"])
    CANDLE_UP_COLOR = str(colors["candle_up"])
    CANDLE_DOWN_COLOR = str(colors["candle_down"])
    SIGNAL_BUY_COLOR = str(colors["signal_buy"])
    SIGNAL_SELL_COLOR = str(colors["signal_sell"])
    OBSERVE_TEXT_COLOR = str(colors["observe_text"])
    r, g, b, a = colors["live_row_bg"]
    LIVE_ROW_BG = QtGui.QColor(int(r), int(g), int(b), int(a))
    return resolved


def current_theme_mode() -> str:
    """当前生效的图表模式。"""
    return _THEME_MODE


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data: np.ndarray,
                 up_color: Optional[str] = None,
                 down_color: Optional[str] = None):
        super().__init__()
        self.data = data
        self.up_color = up_color or CANDLE_UP_COLOR
        self.down_color = down_color or CANDLE_DOWN_COLOR
        self.picture = QtGui.QPicture()
        self._generate_picture()

    def _generate_picture(self) -> None:
        """批量绘制蜡烛。

        旧实现逐根 ``for row in self.data`` 并用 ``row[:5]`` 切片 —— numpy
        切片每次都会新建数组，8000 根就要建 4 万个小数组，实测单次重绘 1.3 秒。
        改成先把各列取成连续数组、再按涨跌分两组用 ``drawLines`` / ``drawRects``
        批量提交，同样 8000 根降到几十毫秒。
        """
        p = QtGui.QPainter(self.picture)
        w = 0.35
        data = self.data
        if data is None or len(data) == 0:
            p.end()
            return

        x = data[:, 0]
        o = data[:, 1]
        c = data[:, 2]
        low = data[:, 3]
        high = data[:, 4]
        up_mask = c >= o

        def draw(mask, color: str) -> None:
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return
            pen = pg.mkPen(color, width=1.2)
            p.setPen(pen)
            p.drawLines([QtCore.QLineF(x[i], low[i], x[i], high[i]) for i in idx])
            p.setBrush(pg.mkBrush(color))
            p.drawRects([
                QtCore.QRectF(x[i] - w, min(o[i], c[i]), w * 2,
                              max(abs(c[i] - o[i]), 1e-4))
                for i in idx
            ])

        draw(up_mask, self.up_color)
        draw(~up_mask, self.down_color)
        p.end()

    def paint(self, p, *args) -> None:
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.picture.boundingRect())


class KLineChart(QtWidgets.QWidget):
    crosshairMoved = QtCore.pyqtSignal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.indicators: Dict[str, Any] = {}
        self.overlay_items: List[Any] = []
        self.channel_items: List[Any] = []
        self._candle: Optional[CandlestickItem] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        self.plot_main = self.win.addPlot(row=0, col=0)
        self.plot_main.setMouseEnabled(x=True, y=True)

        self.plot_sub = self.win.addPlot(row=1, col=0)
        self.plot_sub.setMaximumHeight(180)
        self.plot_sub.setXLink(self.plot_main)

        # 十字光标
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.plot_main.addItem(self.v_line, ignoreBounds=True)
        self.plot_main.addItem(self.h_line, ignoreBounds=True)

        # 底板 / 网格 / 坐标轴 / 光标统一由 apply_theme 铺设，
        # 避免两处各写一套配色后失去同步。
        self.apply_theme(_THEME_MODE)

        self.proxy = pg.SignalProxy(self.plot_main.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def apply_theme(self, mode: str) -> str:
        """同步图表配色（pyqtgraph 画布不认 QSS，必须单独设置）。

        深色 → ``#18181A`` 底板 + 网格 alpha 0.08；
        浅色 → ``#FFFFFF`` 底板 + 网格 alpha 0.15。
        """
        prev_candles = (CANDLE_UP_COLOR, CANDLE_DOWN_COLOR)
        resolved = set_theme_mode(mode)
        colors = chart_theme(resolved)

        self.win.setBackground(str(colors["chart_bg"]))

        for plot in (self.plot_main, self.plot_sub):
            plot.showGrid(x=True, y=True, alpha=float(colors["grid_alpha"]))
            for side in ("bottom", "left"):
                axis = plot.getAxis(side)
                axis.setPen(pg.mkPen(str(colors["axis_pen"]), width=1))
                axis.setTextPen(pg.mkPen(str(colors["axis_text"])))

        dash_pen = pg.mkPen(str(colors["crosshair"]), width=1,
                            style=QtCore.Qt.PenStyle.DashLine)
        self.v_line.setPen(dash_pen)
        self.h_line.setPen(dash_pen)

        # 蜡烛颜色是画进位图（QPicture）里的，颜色变了就必须重画；
        # 两个主题的蜡烛色故意保持一致，所以常态下这一步会被跳过（省 ~350ms）。
        if (CANDLE_UP_COLOR, CANDLE_DOWN_COLOR) != prev_candles:
            if self.df is not None and not self.df.empty:
                self._redraw_candles()
        return resolved

    def _redraw_candles(self) -> None:
        if self._candle is not None:
            try:
                self.plot_main.removeItem(self._candle)
            except Exception:
                pass
            self._candle = None
        self._draw_candles()

    def _draw_candles(self) -> None:
        if self.df is None or self.df.empty:
            return
        n = len(self.df)
        # 用 to_numpy 一次性取列，避免 iterrows()（8000 行要 ~0.9 秒）
        data = np.column_stack([
            np.arange(n, dtype=float),
            self.df["open"].to_numpy(dtype=float),
            self.df["close"].to_numpy(dtype=float),
            self.df["low"].to_numpy(dtype=float),
            self.df["high"].to_numpy(dtype=float),
        ])
        self._candle = CandlestickItem(data)
        self.plot_main.addItem(self._candle)

    def set_data(self, df: pd.DataFrame) -> None:
        self.df = df.copy().reset_index(drop=True)
        self.clear_overlays()
        self.plot_main.clear()
        self.plot_sub.clear()
        self._candle = None

        self.plot_main.addItem(self.v_line, ignoreBounds=True)
        self.plot_main.addItem(self.h_line, ignoreBounds=True)

        self._draw_candles()
        self.reset_view()

    def set_indicators(self, indicators: Dict[str, Any]) -> None:
        self.indicators = indicators or {}

    def plot_squeeze_momentum_with_badges(
        self, squeeze_df: pd.DataFrame, macd_patterns: dict = None,
    ) -> None:
        """渲染高对比度 Squeeze 释放三角与形态徽章。

        兼容当前内部列名 ``release`` / ``momentum`` 与外部常用列名
        ``fired`` / ``val``；本方法只向副图添加图元，不改动指标数据。
        """
        if self.plot_sub is None or squeeze_df is None or squeeze_df.empty:
            return

        fired_col = "fired" if "fired" in squeeze_df.columns else "release"
        value_col = "val" if "val" in squeeze_df.columns else "momentum"
        if fired_col not in squeeze_df.columns or value_col not in squeeze_df.columns:
            return

        values = pd.to_numeric(squeeze_df[value_col], errors="coerce")
        fired = squeeze_df[fired_col].fillna(False).astype(bool)
        x = np.arange(len(squeeze_df), dtype=float)
        valid = fired.to_numpy() & np.isfinite(values.to_numpy(dtype=float))
        positive = valid & (values.to_numpy(dtype=float) >= 0)
        negative = valid & (values.to_numpy(dtype=float) < 0)
        zero_y_pos = np.zeros(int(positive.sum()))
        zero_y_neg = np.zeros(int(negative.sum()))

        if positive.any():
            self.plot_sub.plot(
                x[positive], zero_y_pos, pen=None, symbol="t1", symbolSize=16,
                symbolBrush=pg.mkBrush("#FF9500"),
                symbolPen=pg.mkPen("#FFFFFF", width=1.5),
            )
        if negative.any():
            self.plot_sub.plot(
                x[negative], zero_y_neg, pen=None, symbol="t", symbolSize=16,
                symbolBrush=pg.mkBrush("#00F0FF"),
                symbolPen=pg.mkPen("#FFFFFF", width=1.5),
            )

        if not isinstance(macd_patterns, Mapping):
            return
        font = QtGui.QFont("Segoe UI Emoji", 12, QtGui.QFont.Weight.Bold)
        finite_values = values.to_numpy(dtype=float)
        amplitude = float(np.nanmax(np.abs(finite_values))) if np.isfinite(finite_values).any() else 0.0
        # 动态偏移避免在分钟线等小数动量场景中用固定 1.2 拉坏副图坐标范围。
        offset = max(amplitude * 0.12, 1e-9)
        for i in macd_patterns.get("air_refuel", []):
            if 0 <= int(i) < len(squeeze_df) and np.isfinite(finite_values[int(i)]):
                txt = pg.TextItem("⚡加油", color="#FFD60A", anchor=(0.5, 0))
                txt.setFont(font)
                txt.setToolTip("MACD 空中加油：水上洗盘结束后的多头中继观察")
                txt.setPos(float(i), float(max(finite_values[int(i)], 0.0) + offset))
                self.plot_sub.addItem(txt)
        for i in macd_patterns.get("frost_on_snow", []):
            if 0 <= int(i) < len(squeeze_df) and np.isfinite(finite_values[int(i)]):
                txt = pg.TextItem("💣雪上加霜", color="#FF453A", anchor=(0.5, 1))
                txt.setFont(font)
                txt.setToolTip("MACD 雪上加霜：水下弱反弹夭折后的空头加速观察")
                txt.setPos(float(i), float(min(finite_values[int(i)], 0.0) - offset))
                self.plot_sub.addItem(txt)

    def set_subplot(self, indicator_name: str) -> None:
        self.plot_sub.clear()
        if self.df is None or self.df.empty:
            return

        x = np.arange(len(self.df), dtype=float)
        kind = str(indicator_name or "").lower()
        if "squeeze" in kind or "挤压" in str(indicator_name):
            squeeze = self.indicators.get("squeeze") if self.indicators else None
            if isinstance(squeeze, pd.DataFrame) and "momentum" in squeeze.columns:
                momentum = pd.to_numeric(squeeze["momentum"], errors="coerce").to_numpy(dtype=float)
                on = squeeze.get("on", pd.Series(False, index=range(len(momentum))))
                on = np.asarray(on, dtype=bool)
                release = np.asarray(squeeze.get("release", np.zeros(len(momentum), dtype=bool)), dtype=bool)
            else:
                momentum = np.zeros(len(self.df), dtype=float)
                on = np.zeros(len(self.df), dtype=bool)
                release = np.zeros(len(self.df), dtype=bool)
            n = min(len(x), len(momentum))
            x2, m = x[:n], momentum[:n]
            finite = np.isfinite(m)
            if finite.any():
                pos = finite & (m >= 0)
                neg = finite & (m < 0)
                if pos.any():
                    self.plot_sub.addItem(pg.BarGraphItem(x=x2[pos], height=m[pos], width=0.7,
                                                          brush=pg.mkBrush("#FF453A"), pen=pg.mkPen(None)))
                if neg.any():
                    self.plot_sub.addItem(pg.BarGraphItem(x=x2[neg], height=m[neg], width=0.7,
                                                          brush=pg.mkBrush("#26D0B8"), pen=pg.mkPen(None)))
                self.plot_sub.addItem(pg.InfiniteLine(pos=0, angle=0,
                                                      pen=pg.mkPen(ZERO_LINE_COLOR, width=1)))
                squeeze_x = x2[:len(on)][on[:n]] if len(on) else np.array([])
                squeeze_y = np.zeros(len(squeeze_x))
                if len(squeeze_x):
                    self.plot_sub.addItem(pg.ScatterPlotItem(squeeze_x, squeeze_y, symbol="o", size=7,
                                                             brush=pg.mkBrush("#9A9A9F"), pen=None))
                patterns = self.indicators.get("macd_patterns") if self.indicators else None
                self.plot_squeeze_momentum_with_badges(squeeze, patterns)
                self.plot_sub.setLabel("left", "Squeeze动量")
            return

        # 当前 UI 仅保留 MACD 和 Squeeze；保留显式分支，避免显示名称
        # （如“MACD 副图”）改变时误落入错误的绘制路径。
        if "macd" not in kind:
            return

        macd_df = self.indicators.get("macd") if self.indicators else None

        if isinstance(macd_df, pd.DataFrame) and {"dif", "dea", "macd"}.issubset(macd_df.columns):
            dif = macd_df["dif"].to_numpy(dtype=float)
            dea = macd_df["dea"].to_numpy(dtype=float)
            macd = macd_df["macd"].to_numpy(dtype=float)
        else:
            c = self.df["close"].astype(float)
            dif_s = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
            dea_s = dif_s.ewm(span=9, adjust=False).mean()
            macd_s = 2 * (dif_s - dea_s)
            dif, dea, macd = dif_s.to_numpy(), dea_s.to_numpy(), macd_s.to_numpy()

        pos_mask = macd >= 0
        neg_mask = macd < 0

        if np.any(pos_mask):
            bg_pos = pg.BarGraphItem(
                x=x[pos_mask], height=macd[pos_mask], width=0.5,
                brush=pg.mkBrush("#FF453A"), pen=pg.mkPen(None)
            )
            self.plot_sub.addItem(bg_pos)

        if np.any(neg_mask):
            bg_neg = pg.BarGraphItem(
                x=x[neg_mask], height=macd[neg_mask], width=0.5,
                brush=pg.mkBrush("#26D0B8"), pen=pg.mkPen(None)
            )
            self.plot_sub.addItem(bg_neg)

        self.plot_sub.plot(x, dif, pen=pg.mkPen("#f57c00", width=1.5), name="DIF")
        self.plot_sub.plot(x, dea, pen=pg.mkPen("#0288d1", width=1.5), name="DEA")

    def plot_keltner(self, keltner: Any) -> None:
        """在主图绘制肯特纳通道中轨、上轨和下轨。"""
        if self.df is None or self.df.empty or not isinstance(keltner, pd.DataFrame):
            return
        required = {"mid", "upper", "lower"}
        if not required.issubset(keltner.columns):
            return
        n = min(len(self.df), len(keltner))
        x = np.arange(n, dtype=float)
        colors = {"mid": "#ff9800", "upper": "#8e24aa", "lower": "#8e24aa"}
        styles = {"mid": QtCore.Qt.PenStyle.SolidLine,
                  "upper": QtCore.Qt.PenStyle.DashLine,
                  "lower": QtCore.Qt.PenStyle.DashLine}
        for key in ("upper", "mid", "lower"):
            values = pd.to_numeric(keltner[key].iloc[:n], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values)
            if not mask.any():
                continue
            item = pg.PlotCurveItem(x[mask], values[mask],
                                    pen=pg.mkPen(colors[key], width=1.8 if key == "mid" else 1.3,
                                                 style=styles[key]))
            self.plot_main.addItem(item)
            self.overlay_items.append(item)

    def clear_overlays(self) -> None:
        self.remove_overlay_items(list(self.overlay_items))
        self.overlay_items.clear()

        for it in self.channel_items:
            try:
                self.plot_main.removeItem(it)
            except Exception:
                pass
        self.channel_items.clear()

    def remove_overlay_items(self, items: List[Any]) -> None:
        """按需摘除部分叠加层（供主题切换时只重画变色图层）。"""
        for it in items:
            try:
                self.plot_main.removeItem(it)
            except Exception:
                pass
            try:
                self.overlay_items.remove(it)
            except ValueError:
                pass

    def plot_fractals(self, fractals: List[Any]) -> None:
        for f in fractals:
            idx = getattr(f, "raw_index", getattr(f, "index", None))
            price = getattr(f, "price", None)
            kind = getattr(f, "kind", None)

            if idx is None and isinstance(f, dict):
                idx = f.get("index") or f.get("raw_index")
                price = f.get("price")
                kind = f.get("kind")

            if idx is None or price is None:
                continue

            is_top = str(kind).lower() in ("top", "high", "1")
            color = "#d32f2f" if is_top else "#00796b"
            scatter = pg.ScatterPlotItem(
                [int(idx)], [float(price)],
                symbol="t1" if is_top else "t",
                size=8, pen=None, brush=pg.mkBrush(color)
            )
            self.plot_main.addItem(scatter)
            self.overlay_items.append(scatter)

    def plot_strokes(self, strokes: List[Any]) -> None:
        if not strokes:
            return

        points = []
        for s in strokes:
            if hasattr(s, "start") and hasattr(s, "end"):
                st_idx = getattr(s.start, "raw_index", getattr(s.start, "index", 0))
                st_p = getattr(s.start, "price", 0.0)
                end_idx = getattr(s.end, "raw_index", getattr(s.end, "index", 0))
                end_p = getattr(s.end, "price", 0.0)
                if not points:
                    points.append((int(st_idx), float(st_p)))
                points.append((int(end_idx), float(end_p)))
            elif isinstance(s, dict):
                idx = s.get("index") or s.get("raw_index")
                p = s.get("price")
                if idx is not None and p is not None:
                    points.append((int(idx), float(p)))

        if len(points) < 2:
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        line = pg.PlotCurveItem(xs, ys, pen=pg.mkPen(STROKE_COLOR, width=1.5))
        self.plot_main.addItem(line)
        self.overlay_items.append(line)

        # 支撑阻力水平线：基于关键笔端点计算并向右绘制
        self._plot_support_resistance_lines(points)

    def _plot_support_resistance_lines(self, points: List[Tuple[int, float]]) -> None:
        """绘制当前价上下方各两级水平支撑/压力。

        旧逻辑按笔端点奇偶位置取 ``max/min``，在下降趋势中很容易把当前价
        上方的压力误标成支撑。这里直接从最近行情的局部高低点提取候选，
        再按当前收盘价分上下两侧筛选，并合并相近价位。
        """
        if self.df is None or self.df.empty or "close" not in self.df.columns:
            return

        total_bars = len(self.df)
        close = pd.to_numeric(self.df["close"], errors="coerce").to_numpy(dtype=float)
        highs = pd.to_numeric(self.df.get("high", self.df["close"]), errors="coerce").to_numpy(dtype=float)
        lows = pd.to_numeric(self.df.get("low", self.df["close"]), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(close)
        if not valid.any():
            return
        last = int(np.flatnonzero(valid)[-1])
        current = float(close[last])
        if not np.isfinite(current) or current <= 0:
            return

        lookback = max(40, min(total_bars, 180))
        start = max(0, last - lookback + 1)
        # 局部极值窗口随数据量略微变化，避免单根噪声成为水平位。
        radius = 2 if lookback < 90 else 3
        support_candidates = []
        resistance_candidates = []
        for i in range(start + radius, last - radius + 1):
            if not np.isfinite(highs[i]) or not np.isfinite(lows[i]):
                continue
            lo_window = lows[i - radius:i + radius + 1]
            hi_window = highs[i - radius:i + radius + 1]
            if lows[i] <= np.nanmin(lo_window):
                support_candidates.append((float(lows[i]), i))
            if highs[i] >= np.nanmax(hi_window):
                resistance_candidates.append((float(highs[i]), i))
        # 笔端点作为行情过短时的补充候选。
        for idx, price in points[-16:]:
            if not np.isfinite(price):
                continue
            if price < current:
                support_candidates.append((float(price), int(idx)))
            elif price > current:
                resistance_candidates.append((float(price), int(idx)))

        tolerance = max(current * 0.004, float(np.nanstd(close[start:last + 1])) * 0.35, 1e-6)

        def distinct(candidates):
            ordered = sorted(candidates, key=lambda item: abs(item[0] - current))
            chosen = []
            for price, idx in ordered:
                if any(abs(price - old[0]) <= tolerance for old in chosen):
                    continue
                chosen.append((price, idx))
                if len(chosen) == 2:
                    break
            return chosen

        supports = distinct([(p, i) for p, i in support_candidates if p < current * 0.9995])
        resistances = distinct([(p, i) for p, i in resistance_candidates if p > current * 1.0005])
        x_end = total_bars + 12

        def draw(levels, color, name, anchor):
            font = QtGui.QFont("Microsoft YaHei", 9, QtGui.QFont.Weight.Bold)
            for rank, (price, idx) in enumerate(levels, 1):
                line = pg.PlotCurveItem([max(0, idx), x_end], [price, price],
                                        pen=pg.mkPen(color, width=1.6, style=QtCore.Qt.PenStyle.DashLine))
                text = pg.TextItem(f"{name}{rank}: {price:.2f}", anchor=(0, anchor), color=color)
                text.setFont(font)
                text.setPos(x_end, price)
                self.plot_main.addItem(line)
                self.plot_main.addItem(text)
                self.overlay_items.extend([line, text])

        draw(resistances, "#d32f2f", "压力", 1)
        draw(supports, "#1976d2", "支撑", 0)

    def plot_pivots(self, pivots: List[Any]) -> None:
        for p in pivots:
            if hasattr(p, "start_index"):
                # 中枢索引来自合并K线；图表横轴是原始K线，优先使用 raw 索引。
                x0 = int(getattr(p, "raw_start", getattr(p, "start_index", 0)))
                x1 = int(getattr(p, "raw_end", getattr(p, "end_index", x0 + 1)))
                zd = float(getattr(p, "zd", getattr(p, "low", 0.0)))
                zg = float(getattr(p, "zg", getattr(p, "high", 0.0)))
            elif isinstance(p, dict):
                x0 = int(p.get("raw_start", p.get("start_index", 0)))
                x1 = int(p.get("raw_end", p.get("end_index", x0 + 1)))
                zd = float(p.get("zd", p.get("low", 0.0)))
                zg = float(p.get("zg", p.get("high", 0.0)))
            else:
                continue

            if not np.isfinite(zd) or not np.isfinite(zg) or zg <= zd:
                continue

            rect = QtWidgets.QGraphicsRectItem(x0 - 0.2, zd, max(x1 - x0 + 0.4, 1), zg - zd)
            rect.setPen(pg.mkPen("#f57c00", width=1.2, style=QtCore.Qt.PenStyle.DashLine))
            rect.setBrush(pg.mkBrush(QtGui.QColor(255, 152, 0, 45)))
            self.plot_main.addItem(rect)
            self.overlay_items.append(rect)

    def plot_waves(self, waves: List[Any]) -> None:
        if self.df is None or not waves:
            return
        points = []
        for wave in waves:
            idx = getattr(wave, "raw_index", getattr(wave, "index", None))
            price = getattr(wave, "price", None)
            if idx is None or price is None:
                continue
            try:
                idx, price = int(idx), float(price)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(self.df) and np.isfinite(price):
                points.append((idx, price, str(getattr(wave, "label", "")), str(getattr(wave, "kind", ""))))
        if not points:
            return
        points.sort(key=lambda item: item[0])
        line = pg.PlotCurveItem(
            [p[0] for p in points], [p[1] for p in points],
            pen=pg.mkPen("#6a1b9a", width=2.0, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.plot_main.addItem(line)
        self.overlay_items.append(line)
        span = float(self.df["high"].max() - self.df["low"].min()) or 1.0
        font = QtGui.QFont("Microsoft YaHei", 14, QtGui.QFont.Weight.Bold)
        for idx, price, label, kind in points:
            color = "#8e24aa" if kind.lower() in {"top", "高点", "peak"} else "#1565c0"
            dot = pg.ScatterPlotItem([idx], [price], symbol="o", size=16,
                                     pen=pg.mkPen("#ffffff", width=1.2), brush=pg.mkBrush(color))
            text = pg.TextItem(label or "波", anchor=(0.5, 1.2 if color == "#1565c0" else -0.2), color=color)
            text.setFont(font)
            text.setPos(idx, price + (span * 0.012 if color == "#1565c0" else -span * 0.012))
            self.plot_main.addItem(dot)
            self.plot_main.addItem(text)
            self.overlay_items.extend([dot, text])

    def plot_signals(self, signals: List[Any]) -> None:
        """买卖点大字体、加宽五角星与鲜明标注。

        同方向的星星合并成一个 ``ScatterPlotItem`` 提交（243 个信号从 243 个
        图元降到 2 个），字号 QFont 也只在循环外建一次 —— 主题切换时会整层重画，
        这些开销直接决定切换是否跟手。
        """
        if self.df is None or self.df.empty:
            return

        price_span = float(self.df["high"].max() - self.df["low"].min())
        y_offset = price_span * 0.015

        buys: List[tuple] = []
        sells: List[tuple] = []
        for s in signals:
            idx = getattr(s, "raw_index", getattr(s, "index", None))
            price = getattr(s, "price", None)
            if idx is None or price is None:
                continue
            idx, price = int(idx), float(price)
            label = str(getattr(s, "label", "")).upper()
            (buys if getattr(s, "side", "") == "buy" else sells).append((idx, price, label))

        font = QtGui.QFont("Arial", 19, QtGui.QFont.Weight.ExtraBold)

        for group, is_buy in ((buys, True), (sells, False)):
            if not group:
                continue
            color = SIGNAL_BUY_COLOR if is_buy else SIGNAL_SELL_COLOR
            # 大号醒目五角星（同色批量提交）
            scatter = pg.ScatterPlotItem(
                [g[0] for g in group], [g[1] for g in group],
                symbol="star", size=32,
                pen=pg.mkPen("#ffffff", width=2.2),
                brush=pg.mkBrush(color),
            )
            self.plot_main.addItem(scatter)
            self.overlay_items.append(scatter)

            # 大号加宽字体标签，确保缩放后仍然醒目
            for idx, price, label in group:
                text = pg.TextItem(label, anchor=(0.5, 1.4 if is_buy else -0.4), color=color)
                text.setFont(font)
                text.setPos(idx, price - y_offset if is_buy else price + y_offset)
                self.plot_main.addItem(text)
                self.overlay_items.append(text)

    def _add_gap_rect(self, x0: int, low: float, high: float, x1: int,
                      color: str, rgba: tuple) -> None:
        if high <= low:
            return
        rect = QtWidgets.QGraphicsRectItem(x0 - 0.3, low,
                                           max(x1 - x0 + 0.6, 0.6), high - low)
        rect.setPen(pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DashLine))
        rect.setBrush(pg.mkBrush(QtGui.QColor(*rgba)))
        self.plot_main.addItem(rect)
        self.overlay_items.append(rect)

    def plot_gaps(self) -> None:
        """标注未补跳空缺口（向上跳空浅红，向下跳空浅绿）。

        旧实现是双重 ``for`` 找"首次回补K线"，最坏 O(n²) —— 8724 根K线实测
        447 毫秒。改成用 numpy 一次性定位所有缺口、再对每个缺口做向量化查找，
        整体降到 O(n)。
        """
        if self.df is None or len(self.df) < 2:
            return

        total_bars = len(self.df)
        highs = self.df["high"].to_numpy(dtype=float)
        lows = self.df["low"].to_numpy(dtype=float)

        # 向上跳空缺口：今日最低 > 昨日最高
        for i in (np.flatnonzero(lows[1:] > highs[:-1]) + 1):
            gap_low, gap_high = float(highs[i - 1]), float(lows[i])
            rest = np.flatnonzero(lows[i + 1:] <= gap_low)
            fill_idx = int(rest[0]) + i + 1 if rest.size else total_bars
            self._add_gap_rect(int(i), gap_low, gap_high, fill_idx,
                               "#ef5350", (239, 83, 80, 40))

        # 向下跳空缺口：今日最高 < 昨日最低
        for i in (np.flatnonzero(highs[1:] < lows[:-1]) + 1):
            gap_high, gap_low = float(lows[i - 1]), float(highs[i])
            rest = np.flatnonzero(highs[i + 1:] >= gap_high)
            fill_idx = int(rest[0]) + i + 1 if rest.size else total_bars
            self._add_gap_rect(int(i), gap_low, gap_high, fill_idx,
                               "#26a69a", (38, 166, 154, 40))

    def plot_supertrend(self, st_df: Any) -> None:
        pass

    def plot_bull_bear_line(self, series: Any) -> None:
        """绘制牛熊分界线（默认60周期收盘价均线）。"""
        if self.df is None or series is None:
            return
        values = np.asarray(series, dtype=float).reshape(-1)
        n = min(len(values), len(self.df))
        if n <= 0:
            return
        values = values[:n]
        mask = np.isfinite(values)
        if not mask.any():
            return
        line = pg.PlotCurveItem(np.arange(n)[mask], values[mask],
                                pen=pg.mkPen("#7b1fa2", width=2.6))
        self.plot_main.addItem(line)
        self.overlay_items.append(line)
        last = int(np.flatnonzero(mask)[-1])
        tag = pg.TextItem("牛熊分界线", anchor=(1.0, 0.5), color="#7b1fa2")
        tag.setFont(QtGui.QFont("Microsoft YaHei", 11, QtGui.QFont.Weight.Bold))
        tag.setPos(last, float(values[last]))
        self.plot_main.addItem(tag)
        self.overlay_items.append(tag)

    def plot_anchored_vwap(self, series: Any) -> None:
        """绘制主图定锚 VWAP。

        ``series`` 与原始 K 线保持相同索引，锚点左侧的 NaN 会被自动跳过。
        此方法只负责显示，不参与任何缠论结构或买卖点计算。
        """
        if self.df is None or series is None:
            return
        values = np.asarray(series, dtype=float).reshape(-1)
        n = min(len(values), len(self.df))
        if n <= 0:
            return
        values = values[:n]
        mask = np.isfinite(values)
        if not mask.any():
            return
        color = "#d97706"
        line = pg.PlotCurveItem(
            np.arange(n)[mask], values[mask], pen=pg.mkPen(color, width=2.2)
        )
        self.plot_main.addItem(line)
        self.overlay_items.append(line)
        last = int(np.flatnonzero(mask)[-1])
        tag = pg.TextItem("定锚VWAP", anchor=(1.0, 0.5), color=color)
        tag.setFont(QtGui.QFont("Microsoft YaHei", 10, QtGui.QFont.Weight.Bold))
        tag.setPos(last, float(values[last]))
        self.plot_main.addItem(tag)
        self.overlay_items.append(tag)

    def plot_avwap(self, avwap_series: pd.Series) -> None:
        """以橙色虚线叠加显示 Anchored VWAP。"""
        if avwap_series is None or avwap_series.dropna().empty:
            return
        valid = avwap_series.dropna()
        pen = pg.mkPen(
            color="#FF9500", width=2,
            style=QtCore.Qt.PenStyle.DashLine,
        )
        # 当前图表类的主图和叠加项容器名称分别为 plot_main / overlay_items。
        item = self.plot_main.plot(
            valid.index.to_numpy(), valid.to_numpy(), pen=pen, name="AVWAP"
        )
        self.overlay_items.append(item)

    def plot_channel(self, channel: Any, mode: str = "自动") -> None:
        for it in self.channel_items:
            try:
                self.plot_main.removeItem(it)
            except Exception:
                pass
        self.channel_items.clear()

        if not channel or not getattr(channel, "valid", False):
            return

        items_to_draw = []
        p_struct = getattr(channel, "primary_structure", None)
        s_struct = getattr(channel, "secondary_structure", None)

        if mode == "只显示主结构":
            if p_struct:
                items_to_draw.append((p_struct, True))
        elif mode == "仅局部结构":
            if s_struct:
                items_to_draw.append((s_struct, False))
        else:
            if p_struct:
                items_to_draw.append((p_struct, True))
            if s_struct:
                if abs(s_struct.fit_span - (p_struct.fit_span if p_struct else 0)) > 6:
                    items_to_draw.append((s_struct, False))

        total_bars = len(self.df) if self.df is not None and not self.df.empty else 100

        for struct, is_primary in items_to_draw:
            st = getattr(struct, "status", "candidate")
            up = getattr(struct, "upper_line", {})
            lo = getattr(struct, "lower_line", {})
            if not up or not lo:
                continue

            if is_primary:
                color_up = "#d32f2f" if st == "confirmed" else "#e65100"
                color_lo = "#1976d2" if st == "confirmed" else "#0097a7"
                style_line = QtCore.Qt.PenStyle.SolidLine if st == "confirmed" else QtCore.Qt.PenStyle.DashLine
                width_line = 2.2
            else:
                color_up = "#7b1fa2"
                color_lo = "#00796b"
                style_line = QtCore.Qt.PenStyle.DashDotLine
                width_line = 1.5

            pen_upper = pg.mkPen(color=color_up, width=width_line, style=style_line)
            pen_lower = pg.mkPen(color=color_lo, width=width_line, style=style_line)

            span = max(up.get("x2", 0) - up.get("x1", 0), lo.get("x2", 0) - lo.get("x1", 0), 20)
            ext = int(min(span * 0.2, 15))
            x_end = min(total_bars - 1 + ext, total_bars + 15)

            y_u_start = line_value(up, up["x1"])
            y_u_end = line_value(up, x_end)
            y_l_start = line_value(lo, lo["x1"])
            y_l_end = line_value(lo, x_end)

            c_upper = pg.PlotCurveItem([up["x1"], x_end], [y_u_start, y_u_end], pen=pen_upper)
            c_lower = pg.PlotCurveItem([lo["x1"], x_end], [y_l_start, y_l_end], pen=pen_lower)
            self.plot_main.addItem(c_upper)
            self.plot_main.addItem(c_lower)
            self.channel_items.extend([c_upper, c_lower])

            for pt in up.get("touches", []):
                sp = pg.ScatterPlotItem(
                    [pt["index"]], [pt["price"]],
                    size=7 if is_primary else 5,
                    pen=pg.mkPen("#ffffff", width=1),
                    brush=pg.mkBrush(color_up)
                )
                self.plot_main.addItem(sp)
                self.channel_items.append(sp)

            for pt in lo.get("touches", []):
                sp = pg.ScatterPlotItem(
                    [pt["index"]], [pt["price"]],
                    size=7 if is_primary else 5,
                    pen=pg.mkPen("#ffffff", width=1),
                    brush=pg.mkBrush(color_lo)
                )
                self.plot_main.addItem(sp)
                self.channel_items.append(sp)

            prefix = "【主结构】" if is_primary else "【局部】"
            txt = (
                f"{prefix}{getattr(struct, 'label', '整理')}({st})\n"
                f"触点: 上{up.get('touch_count', 0)}/下{lo.get('touch_count', 0)}共{getattr(struct, 'touch_count', 0)}次 | 跨度: {getattr(struct, 'fit_span', 0)}根\n"
                f"阻力/突破位: {getattr(struct, 'breakout_up_level', 0.0):.2f} | 支撑位: {getattr(struct, 'breakdown_level', 0.0):.2f}"
            )
            t_item = pg.TextItem(txt, anchor=(0, 1 if is_primary else 0), color=color_up)
            t_item.setFont(QtGui.QFont("Microsoft YaHei", 9, QtGui.QFont.Weight.Bold))
            t_item.setPos(x_end, y_u_end)
            self.plot_main.addItem(t_item)
            self.channel_items.append(t_item)

    def highlight_at(self, raw_index: int) -> None:
        if self.df is None or raw_index < 0 or raw_index >= len(self.df):
            return
        price = float(self.df.loc[raw_index, "close"])
        self.plot_main.setXRange(max(0, raw_index - 40), min(len(self.df), raw_index + 40), padding=0)
        self.v_line.setPos(raw_index)
        self.h_line.setPos(price)

    def reset_view(self) -> None:
        if self.df is None or self.df.empty:
            return
        n = len(self.df)
        self.plot_main.setXRange(max(0, n - 120), n, padding=0.02)
        sub = self.df.iloc[max(0, n - 120) :]
        self.plot_main.setYRange(sub["low"].min() * 0.99, sub["high"].max() * 1.01)

    def _mouse_moved(self, evt: Any) -> None:
        pos = evt[0]
        if not self.plot_main.sceneBoundingRect().contains(pos) or self.df is None:
            return
        mouse_point = self.plot_main.vb.mapSceneToView(pos)
        idx = int(round(mouse_point.x()))
        if 0 <= idx < len(self.df):
            self.v_line.setPos(mouse_point.x())
            self.h_line.setPos(mouse_point.y())
            r = self.df.iloc[idx]
            chg = r["close"] - r["open"]
            pct = chg / r["open"] * 100 if r["open"] else 0
            self.crosshairMoved.emit({
                "date": str(r["date"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "change": float(chg),
                "pct": float(pct),
            })
