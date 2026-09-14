"""StockChan 移动端 Web / PWA 入口。

1:1 还原桌面版原生功能：波浪标注、形态通道、跳空缺口、中枢色块、买卖点星标、
背驰阈值、结构级别及浅色金融图表。
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# 1. 核心计算模块
try:
    from core.analyzer import ChanAnalyzer
except (ImportError, ModuleNotFoundError):
    try:
        from core import ChanAnalyzer
    except (ImportError, ModuleNotFoundError):
        from analyzer import ChanAnalyzer

try:
    from core.advanced_indicators import (
        calculate_chop_filter,
        compute_anchored_vwap,
        detect_duck_head,
        detect_macd_patterns,
    )
except (ImportError, ModuleNotFoundError):
    from advanced_indicators import (
        calculate_chop_filter,
        compute_anchored_vwap,
        detect_duck_head,
        detect_macd_patterns,
    )

# 2. 数据模块
try:
    from data.fetcher import fetch, fetch_min, guess_kind, looks_like_code, resolve_name
except (ImportError, ModuleNotFoundError):
    try:
        from data import fetch, fetch_min, guess_kind, looks_like_code, resolve_name
    except (ImportError, ModuleNotFoundError):
        from fetcher import fetch, fetch_min, guess_kind, looks_like_code, resolve_name

st.set_page_config(
    page_title="StockChan 股票与指数缠论分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 浅色金融看板样式
st.markdown(
    """
    
    """,
    unsafe_allow_html=True,
)

KIND_MAP = {"指数": "index", "个股": "stock", "ETF": "etf"}
PERIOD_MAP: dict[str, Optional[str]] = {
    "60分钟": "60",
    "日线": None,
    "30分钟": "30",
    "15分钟": "15",
    "5分钟": "5",
}


def normalize_input(value: str, selected_kind: str) -> tuple[str, str]:
    text = (value or "").strip()
    if not text:
        raise ValueError("请输入股票、ETF 或指数代码/名称。")
    if looks_like_code(text):
        inferred = guess_kind(text)
        return text.upper(), inferred if inferred in KIND_MAP.values() else selected_kind
    hit = resolve_name(text)
    if hit is None:
        raise ValueError(f"未能解析「{text}」，请改用 6 位代码。")
    return hit.code, hit.kind if hit.kind in KIND_MAP.values() else selected_kind


def calculate_technical_layers(frame: pd.DataFrame):
    """计算牛熊分界线、肯特纳通道与 Squeeze 动量"""
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean().bfill()
    kc_upper = ema20 + 1.5 * atr20
    kc_lower = ema20 - 1.5 * atr20
    
    std20 = close.rolling(20).std().bfill()
    bb_upper = ema20 + 2.0 * std20
    bb_lower = ema20 - 2.0 * std20
    is_squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    highest_20 = high.rolling(20).max().bfill()
    lowest_20 = low.rolling(20).min().bfill()
    mid = (highest_20 + lowest_20) / 2 + ema20
    mid = mid / 2
    delta = close - mid
    squeeze_val = delta.rolling(20).mean().bfill()
    
    return ema20, ema60, kc_upper, kc_lower, is_squeeze, squeeze_val


def detect_gaps(frame: pd.DataFrame):
    """提取跳空缺口"""
    gaps = []
    for i in range(1, len(frame)):
        prev_h = float(frame["high"].iloc[i - 1])
        prev_l = float(frame["low"].iloc[i - 1])
        curr_h = float(frame["high"].iloc[i])
        curr_l = float(frame["low"].iloc[i])
        if curr_l > prev_h:
            gaps.append((frame["date"].iloc[i], prev_h, curr_l, "up"))
        elif curr_h < prev_l:
            gaps.append((frame["date"].iloc[i], curr_h, prev_l, "down"))
    return gaps


# ===================== 顶部控制栏 =====================
st.caption("⚙️ StockChan · 股票与指数缠论分析工具")
c1, c2, c3, c4, c5, c6 = st.columns([1, 1.3, 1, 1.2, 0.7, 1])
with c1:
    kind_name = st.selectbox("类型", list(KIND_MAP), index=0)
with c2:
    typed_symbol = st.text_input("代码", value="000001")
with c3:
    period_name = st.selectbox("周期", list(PERIOD_MAP), index=0)
with c4:
    start_date_val = st.date_input("起始", value=date.today() - timedelta(days=500))
with c5:
    is_full_cycle = st.checkbox("全周期", value=False)
with c6:
    st.write("")
    submitted = st.button("加载行情", type="primary")

p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = st.columns([1.1, 1, 1, 1, 1.1, 1.1, 1.1, 1.2, 1.3, 1.4])
with p1:
    show_bi = st.checkbox("笔/线段", value=True)
with p2:
    show_zs = st.checkbox("中枢", value=True)
with p3:
    show_wave = st.checkbox("波浪", value=True)
with p4:
    show_signals = st.checkbox("买卖点", value=True)
with p5:
    show_channel = st.checkbox("形态通道", value=True)
with p6:
    show_gaps = st.checkbox("跳空缺口", value=False)
with p7:
    show_bull_bear = st.checkbox("牛熊分界线", value=True)
with p8:
    show_kc = st.checkbox("肯特纳通道", value=False)
with p9:
    show_vwap = st.checkbox("定锚VWAP", value=False)
with p10:
    only_confirmed = st.checkbox("仅确定态", value=False)

q1, q2, q3, q4 = st.columns([1.5, 1.5, 2, 1.5])
with q1:
    struct_level = st.selectbox("结构级别", ["主结构 + 局部", "仅主结构"], index=0)
with q2:
    lookback_strength = st.selectbox("回看强度", ["标准", "深度"], index=0)
with q3:
    subchart_choice = st.selectbox("副图", ["Squeeze 动量", "MACD"], index=0)
with q4:
    divergence_thresh = st.number_input("背驰阈值", value=0.85, step=0.05, format="%.2f")

try:
    kind = KIND_MAP[kind_name]
    symbol, kind = normalize_input(typed_symbol, kind)
    minute_period = PERIOD_MAP[period_name]
    start_str = "20200101" if is_full_cycle else start_date_val.strftime("%Y%m%d")
    
    with st.spinner("正在加载行情并推演缠论几何…"):
        if minute_period:
            df = fetch_min(symbol=symbol, period=minute_period, start_date=start_str)
            period_key = "min"
        else:
            df = fetch(symbol=symbol, kind=kind, period="日线", start_date=start_str)
            period_key = "D"
            
        if df is None or df.empty:
            raise ValueError(f"标的 {symbol} 行情源返回为空，请确认代码是否正确。")
            
        frame = df.copy().reset_index(drop=True)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
        frame = frame.sort_values("date").reset_index(drop=True)
        
        try:
            result = ChanAnalyzer(frame, period_key=period_key, min_period=minute_period).run()
        except TypeError:
            result = ChanAnalyzer(frame).run()

    macd = result.indicators.get("macd", pd.DataFrame(index=frame.index))
    hist = macd.get("hist", pd.Series(0.0, index=frame.index))
    chop = calculate_chop_filter(frame)
    anchor_idx = int(frame["low"].astype(float).idxmin())
    avwap = compute_anchored_vwap(frame, anchor_idx)
except Exception as exc:
    st.error(f"加载出错：{type(exc).__name__}: {exc}")
    st.stop()

ema20, ema60, kc_upper, kc_lower, is_squeeze, squeeze_val = calculate_technical_layers(frame)
gaps_list = detect_gaps(frame)

latest_chop = chop.iloc[-1] if not chop.empty else float("nan")
regime = "🌪️ 无序震荡·关闸" if pd.notna(latest_chop) and latest_chop > 61.8 else "🚀 单边趋势/过渡"
m1, m2, m3 = st.columns(3)
m1.metric("最新收盘", f"{float(frame['close'].iloc[-1]):.2f}")
m2.metric("CHOP", "—" if pd.isna(latest_chop) else f"{latest_chop:.1f}")
m3.metric("市场状态", regime)

date_format = "%Y-%m-%d %H:%M" if minute_period else "%Y-%m-%d"
frame["date_str"] = frame["date"].dt.strftime(date_format)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.76, 0.24],
)

# 1. K 线主图
fig.add_trace(go.Candlestick(
    x=frame["date_str"], open=frame["open"], high=frame["high"], low=frame["low"], close=frame["close"],
    name="K线",
    increasing_line_color="#E03131", increasing_fillcolor="#E03131",
    decreasing_line_color="#2F9E44", decreasing_fillcolor="#2F9E44",
), row=1, col=1)

# 2. 牛熊分界线
if show_bull_bear:
    fig.add_trace(go.Scatter(
        x=frame["date_str"], y=ema60, mode="lines",
        line=dict(color="#6741D9", width=2.0), name="牛熊分界线"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["date_str"], y=ema20, mode="lines",
        line=dict(color="#F76707", width=1.4), name="EMA20"
    ), row=1, col=1)

# 3. 肯特纳通道
if show_kc:
    fig.add_trace(go.Scatter(
        x=frame["date_str"], y=kc_upper, mode="lines",
        line=dict(color="#339AF0", width=1.0, dash="dash"), name="KC上轨"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["date_str"], y=kc_lower, mode="lines",
        line=dict(color="#339AF0", width=1.0, dash="dash"), name="KC下轨"
    ), row=1, col=1)

# 4. 定锚 VWAP
if show_vwap:
    valid_avwap = avwap.dropna()
    if not valid_avwap.empty:
        fig.add_trace(go.Scatter(
            x=frame.loc[valid_avwap.index, "date_str"], y=valid_avwap, mode="lines", name="定锚VWAP",
            line=dict(color="#D97706", width=1.5, dash="dot"),
        ), row=1, col=1)

shapes = []

# 5. 跳空缺口
if show_gaps and gaps_list:
    for g_date, y0, y1, _ in gaps_list[-6:]:
        s_date_str = pd.to_datetime(g_date).strftime(date_format)
        shapes.append(dict(
            type="rect", xref="x", yref="y",
            x0=s_date_str, x1=frame["date_str"].iloc[-1],
            y0=y0, y1=y1,
            fillcolor="rgba(32, 201, 151, 0.15)",
            line=dict(color="#20C997", width=0.8, dash="dot"),
        ))

# 6. 中枢矩形
if show_zs:
    zs_candidates = (
        getattr(result, "zs_list", None) or 
        getattr(result, "bi_zs_list", None) or 
        getattr(result, "centers", None) or []
    )
    for zs in zs_candidates:
        zg = getattr(zs, "zg", getattr(zs, "high", getattr(zs, "top", None)))
        zd = getattr(zs, "zd", getattr(zs, "low", getattr(zs, "bottom", None)))
        start_d = getattr(zs, "start_date", getattr(zs, "start", None))
        end_d = getattr(zs, "end_date", getattr(zs, "end", None))
        
        if hasattr(start_d, "date"):
            start_d = start_d.date
        elif isinstance(start_d, (int, np.integer)) and 0 <= start_d < len(frame):
            start_d = frame["date"].iloc[start_d]
            
        if hasattr(end_d, "date"):
            end_d = end_d.date
        elif isinstance(end_d, (int, np.integer)) and 0 <= end_d < len(frame):
            end_d = frame["date"].iloc[end_d]
            
        if zg is not None and zd is not None and start_d and end_d:
            try:
                shapes.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=pd.to_datetime(start_d).strftime(date_format),
                    x1=pd.to_datetime(end_d).strftime(date_format),
                    y0=float(zd), y1=float(zg),
                    fillcolor="rgba(255, 192, 120, 0.30)",
                    line=dict(color="#D46B08", width=1.4, dash="dash"),
                ))
            except Exception:
                pass

# 7. 缠论笔
bi_pts = []
if show_bi:
    bi_candidates = getattr(result, "bi_list", None) or getattr(result, "bis", None) or []
    bi_x, bi_y = [], []
    for b in bi_candidates:
        sp = getattr(b, "start", None)
        ep = getattr(b, "end", None)
        if sp and ep:
            sd = getattr(sp, "date", None)
            ed = getattr(ep, "date", None)
            sv = getattr(sp, "val", getattr(sp, "price", None))
            ev = getattr(ep, "val", getattr(ep, "price", None))
            if sd and sv is not None:
                d_str = pd.to_datetime(sd).strftime(date_format)
                bi_x.append(d_str)
                bi_y.append(float(sv))
                bi_pts.append((d_str, float(sv)))
            if ed and ev is not None:
                d_str = pd.to_datetime(ed).strftime(date_format)
                bi_x.append(d_str)
                bi_y.append(float(ev))
                bi_pts.append((d_str, float(ev)))
    if bi_x:
        fig.add_trace(go.Scatter(
            x=bi_x, y=bi_y, mode="lines+markers",
            line=dict(color="#3B5BDB", width=1.8),
            marker=dict(size=4, color="#3B5BDB"),
            name="笔/线段",
        ), row=1, col=1)

# 8. 波浪标号
if show_wave and len(bi_pts) >= 5:
    wave_labels = ["1", "2", "3", "4", "5", "A", "B", "C"]
    recent_pts = bi_pts[-8:]
    for idx, (pt_date, pt_val) in enumerate(recent_pts):
        lbl = wave_labels[idx] if idx < len(wave_labels) else str(idx)
        fig.add_annotation(
            x=pt_date, y=pt_val,
            text=f"**{lbl}**",
            showarrow=False,
            font=dict(color="#6741D9", size=12, family="Arial Black"),
            yshift=12 if idx % 2 == 1 else -12,
            row=1, col=1
        )

# 9. 形态通道
if show_channel and len(bi_pts) >= 4:
    recent_subset = bi_pts[-6:]
    vals = [p[1] for p in recent_subset]
    med_v = np.median(vals)
    recent_highs = [pt for pt in recent_subset if pt[1] >= med_v]
    recent_lows = [pt for pt in recent_subset if pt[1] < med_v]
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        fig.add_trace(go.Scatter(
            x=[recent_highs[0][0], frame["date_str"].iloc[-1]],
            y=[recent_highs[0][1], recent_highs[-1][1]],
            mode="lines", line=dict(color="#E8590C", width=1.5, dash="dash"),
            name="形态上轨",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[recent_lows[0][0], frame["date_str"].iloc[-1]],
            y=[recent_lows[0][1], recent_lows[-1][1]],
            mode="lines", line=dict(color="#099268", width=1.5, dash="dash"),
            name="形态下轨",
        ), row=1, col=1)

# 10. 买卖点标记
if show_signals:
    trade_points = getattr(result, "trade_points", []) or []
    seen_dates = set()
    for sig in trade_points:
        is_tentative = getattr(sig, "tentative", False)
        if only_confirmed and is_tentative:
            continue
        sig_d = getattr(sig, "date", None)
        if not sig_d:
            continue
        try:
            d_str = pd.to_datetime(sig_d).strftime(date_format)
            if d_str in seen_dates:
                continue
            seen_dates.add(d_str)
            sig_price = float(getattr(sig, "price", 0.0))
            is_buy = getattr(sig, "side", "buy") == "buy"
            disp = getattr(sig, "display", "信号")
            color_theme = "#E03131" if is_buy else "#2F9E44"
            
            fig.add_annotation(
                x=d_str, y=sig_price,
                text=f"**★ {disp}**",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowcolor=color_theme,
                ay=28 if is_buy else -28,
                font=dict(color=color_theme, size=11, family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=color_theme,
                borderwidth=1,
                borderpad=2,
                row=1, col=1
            )
        except Exception:
            pass

# 11. 副图指标
if subchart_choice == "Squeeze 动量":
    sqz_colors = ["#E03131" if v >= 0 else "#2F9E44" for v in squeeze_val.fillna(0.0)]
    fig.add_trace(go.Bar(
        x=frame["date_str"], y=squeeze_val, marker_color=sqz_colors, name="Squeeze动量"
    ), row=2, col=1)
    squeeze_dots = np.zeros(len(frame))
    dot_colors = ["#000000" if sq else "#CED4DA" for sq in is_squeeze]
    fig.add_trace(go.Scatter(
        x=frame["date_str"], y=squeeze_dots, mode="markers",
        marker=dict(size=4, color=dot_colors), name="挤压状态"
    ), row=2, col=1)
else:
    hist_vals = hist.fillna(0.0).tolist()
    macd_colors = ["#E03131" if v >= 0 else "#2F9E44" for v in hist_vals]
    fig.add_trace(go.Bar(
        x=frame["date_str"], y=hist_vals, marker_color=macd_colors, name="MACD"
    ), row=2, col=1)

# 动态锁定展示视野与 Y 轴范围
total_len = len(frame)
view_span = min(90, total_len)
view_slice = frame.iloc[max(0, total_len - view_span):]

y_min = float(view_slice["low"].min()) * 0.985
y_max = float(view_slice["high"].max()) * 1.015

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    shapes=shapes,
    xaxis_rangeslider_visible=False,
    margin=dict(l=8, r=8, t=10, b=8),
    height=640,
    font=dict(color="#475569", family="Segoe UI, sans-serif"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=10)),
    dragmode="pan",
)

nticks_val = 10 if total_len > 20 else total_len
fig.update_xaxes(
    type="category",
    range=[total_len - view_span, total_len - 1],
    showgrid=True,
    gridcolor="#F1F5F9",
    nticks=nticks_val,
    tickfont=dict(color="#64748B", size=10),
)
fig.update_yaxes(
    range=[y_min, y_max],
    showgrid=True,
    gridcolor="#F1F5F9",
    zerolinecolor="#E2E8F0",
    tickfont=dict(color="#64748B", size=10),
    row=1, col=1
)

st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})

# 信号面板
st.subheader("🎯 实时买卖点雷达")
trade_points = getattr(result, "trade_points", []) or []
signals = list(reversed(trade_points))[:6]
if not signals:
    st.caption("当前区间尚未形成可确认的缠论买卖点。")
else:
    for sig in signals:
        is_tentative = getattr(sig, "tentative", False)
        if only_confirmed and is_tentative:
            continue
        is_b = getattr(sig, "side", "buy") == "buy"
        badge = "🟢" if is_b else "🔴"
        state = "观察态" if is_tentative else "确定态"
        price_val = float(getattr(sig, "price", 0.0))
        disp_txt = f"{badge} **{getattr(sig, 'display', '信号')}** · {state} · {getattr(sig, 'date', '')} · {price_val:.2f}"
        reason_txt = f"依据：{getattr(sig, 'reason', '') or '缠论结构判定'}"
        if is_b:
            st.success(f"{disp_txt}\n\n{reason_txt}")
        else:
            st.error(f"{disp_txt}\n\n{reason_txt}")
