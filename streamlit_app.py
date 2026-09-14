"""StockChan 移动端 Web / PWA 入口。

完整呈现缠论中枢、笔/线段、买卖点标记及动量几何指标。
"""

from __future__ import annotations

import os
import sys
from datetime import date
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# 兼容模块导入
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

try:
    from data import fetch, fetch_min, guess_kind, looks_like_code, resolve_name
except (ImportError, ModuleNotFoundError):
    try:
        from data.fetcher import fetch, fetch_min, guess_kind, looks_like_code, resolve_name
    except (ImportError, ModuleNotFoundError):
        from fetcher import fetch, fetch_min, guess_kind, looks_like_code, resolve_name


st.set_page_config(
    page_title="StockChan 移动量化",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "pwa_installed" not in st.session_state:
    components.html(
        r"""
        
        """,
        height=0,
    )
    st.session_state["pwa_installed"] = True

st.markdown(
    """
    
    """,
    unsafe_allow_html=True,
)

KIND_MAP = {"个股": "stock", "指数": "index", "ETF": "etf"}
PERIOD_MAP: dict[str, Optional[str]] = {
    "日线": None,
    "60分钟": "60",
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


@st.cache_data(ttl=30, show_spinner=False)
def get_analysis_data(symbol: str, kind: str, minute_period: Optional[str], start: str):
    if minute_period:
        df = fetch_min(symbol=symbol, period=minute_period, start_date=start)
        period_key = "min"
    else:
        df = fetch(symbol=symbol, kind=kind, period="日线", start_date=start)
        period_key = "D"
    if df is None or df.empty:
        raise ValueError("行情源返回为空。")
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"行情数据缺少必要列：{', '.join(sorted(missing))}")

    frame = df.copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    if frame.empty:
        raise ValueError("行情日期或价格字段无效。")

    result = ChanAnalyzer(frame, period_key=period_key, min_period=minute_period).run()
    macd = result.indicators.get("macd", pd.DataFrame(index=frame.index))
    dif = macd.get("dif", pd.Series(0.0, index=frame.index))
    dea = macd.get("dea", pd.Series(0.0, index=frame.index))
    hist = macd.get("hist", pd.Series(0.0, index=frame.index))
    patterns = detect_macd_patterns(frame, dif, dea, hist)
    chop = calculate_chop_filter(frame)
    duck = detect_duck_head(frame)
    anchor_idx = int(frame["low"].astype(float).idxmin())
    avwap = compute_anchored_vwap(frame, anchor_idx)
    return frame, result, pd.Series(hist), patterns, chop, duck, avwap


st.title("📈 StockChan · 缠论与动量几何")

# 1. 顶部查询栏
with st.form("quote_query", border=False):
    col1, col2, col3, col4 = st.columns([2.1, 1, 1, 0.8])
    with col1:
        typed_symbol = st.text_input("代码或名称", value="002181", placeholder="例如：粤传媒、000001、510410")
    with col2:
        kind_name = st.selectbox("类型", list(KIND_MAP), index=0)
    with col3:
        period_name = st.selectbox("周期", list(PERIOD_MAP), index=0)
    with col4:
        submitted = st.form_submit_button("加载行情", use_container_width=True)

# 2. 还原桌面版的【叠加显示】控制开关
with st.expander("🛠️ 叠加显示与图层控制", expanded=True):
    layer1, layer2, layer3, layer4, layer5 = st.columns(5)
    with layer1:
        show_bi = st.checkbox("笔 / 线段", value=True)
    with layer2:
        show_zs = st.checkbox("中枢矩形", value=True)
    with layer3:
        show_signals = st.checkbox("买卖点标记", value=True)
    with layer4:
        show_vwap = st.checkbox("定锚 VWAP", value=False)
    with layer5:
        show_patterns = st.checkbox("MACD形态标记", value=True)

if submitted:
    st.cache_data.clear()

try:
    kind = KIND_MAP[kind_name]
    symbol, kind = normalize_input(typed_symbol, kind)
    minute_period = PERIOD_MAP[period_name]
    start = f"{date.today().year - 3}{date.today():%m%d}"
    with st.spinner("正在计算缠论结构与指标…"):
        df, result, hist, patterns, chop, duck, avwap = get_analysis_data(
            symbol, kind, minute_period, start
        )
except Exception as exc:
    st.error(f"行情解析失败或连接超时：{type(exc).__name__}: {exc}")
    st.stop()

latest_chop = chop.iloc[-1] if not chop.empty else float("nan")
regime = "🌪️ 无序震荡·关闸" if pd.notna(latest_chop) and latest_chop > 61.8 else "🚀 单边趋势/过渡"
metric1, metric2, metric3 = st.columns(3)
metric1.metric("最新收盘", f"{float(df['close'].iloc[-1]):.2f}")
metric2.metric("CHOP", "—" if pd.isna(latest_chop) else f"{latest_chop:.1f}")
metric3.metric("市场状态", regime)

# 格式化连续日期
date_format = "%Y-%m-%d %H:%M" if minute_period else "%Y-%m-%d"
df["date_str"] = df["date"].dt.strftime(date_format)
date_to_idx = {d: i for i, d in enumerate(df["date_str"])}

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
    row_heights=[0.72, 0.28],
)

# 1. 主图 K 线（中国标准配色：红涨绿跌）
fig.add_trace(go.Candlestick(
    x=df["date_str"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="K线",
    increasing_line_color="#FF3B30", increasing_fillcolor="#FF3B30",
    decreasing_line_color="#34C759", decreasing_fillcolor="#34C759",
), row=1, col=1)

# 2. 缠论中枢渲染（矩形色块）
shapes = []
if show_zs:
    zs_list = getattr(result, "zs_list", []) or []
    for zs in zs_list:
        zg = getattr(zs, "zg", getattr(zs, "high", None))
        zd = getattr(zs, "zd", getattr(zs, "low", None))
        start_date = getattr(zs, "start_date", None)
        end_date = getattr(zs, "end_date", None)
        if zg is not None and zd is not None and start_date and end_date:
            s_str = pd.to_datetime(start_date).strftime(date_format)
            e_str = pd.to_datetime(end_date).strftime(date_format)
            shapes.append(dict(
                type="rect",
                xref="x", yref="y",
                x0=s_str, x1=e_str,
                y0=float(zd), y1=float(zg),
                fillcolor="rgba(100, 149, 237, 0.22)",
                line=dict(color="#4A90E2", width=1.2, dash="dash"),
            ))

# 3. 缠论笔 / 线段可视化
if show_bi:
    bi_list = getattr(result, "bi_list", []) or []
    bi_x, bi_y = [], []
    for b in bi_list:
        start_pt = getattr(b, "start", None)
        end_pt = getattr(b, "end", None)
        if start_pt and end_pt:
            s_d = getattr(start_pt, "date", None)
            e_d = getattr(end_pt, "date", None)
            if s_d:
                bi_x.append(pd.to_datetime(s_d).strftime(date_format))
                bi_y.append(float(getattr(start_pt, "val", getattr(start_pt, "price", 0.0))))
            if e_d:
                bi_x.append(pd.to_datetime(e_d).strftime(date_format))
                bi_y.append(float(getattr(end_pt, "val", getattr(end_pt, "price", 0.0))))
    if bi_x:
        fig.add_trace(go.Scatter(
            x=bi_x, y=bi_y, mode="lines+markers",
            line=dict(color="#FFD60A", width=2),
            marker=dict(size=4, color="#FFD60A"),
            name="笔/线段",
        ), row=1, col=1)

# 4. K 线上买卖点图标标注
if show_signals:
    trade_points = getattr(result, "trade_points", []) or []
    for sig in trade_points:
        sig_d = getattr(sig, "date", None)
        if not sig_d:
            continue
        d_str = pd.to_datetime(sig_d).strftime(date_format)
        sig_price = float(getattr(sig, "price", 0.0))
        is_buy = getattr(sig, "side", "buy") == "buy"
        disp = getattr(sig, "display", "信号")
        
        fig.add_annotation(
            x=d_str, y=sig_price,
            text=f"{'▲' if is_buy else '▼'} {disp}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#34C759" if is_buy else "#FF3B30",
            arrowsize=1,
            arrowwidth=1.5,
            ay=28 if is_buy else -28,
            font=dict(color="#FFFFFF", size=11),
            bgcolor="rgba(40, 167, 69, 0.8)" if is_buy else "rgba(220, 53, 69, 0.8)",
            bordercolor="#FFFFFF",
            borderwidth=1,
            row=1, col=1
        )

# 5. 定锚 VWAP
if show_vwap:
    valid_avwap = avwap.dropna()
    if not valid_avwap.empty:
        fig.add_trace(go.Scatter(
            x=df.loc[valid_avwap.index, "date_str"], y=valid_avwap, mode="lines", name="定锚 VWAP",
            line=dict(color="#FF9F0A", width=1.5, dash="dash"),
        ), row=1, col=1)

# 6. MACD 动量副图
hist_vals = hist.fillna(0.0).tolist()
macd_colors = ["#FF3B30" if v >= 0 else "#34C759" for v in hist_vals]
fig.add_trace(go.Bar(x=df["date_str"], y=hist_vals, marker_color=macd_colors, name="MACD"), row=2, col=1)

# 空中加油 / 雪上加霜
if show_patterns and isinstance(patterns, dict):
    for idx in patterns.get("air_refuel", []):
        if 0 <= idx < len(df):
            fig.add_annotation(
                x=df["date_str"].iloc[idx], y=hist_vals[idx], text="⚡加油", showarrow=True,
                arrowhead=1, arrowcolor="#FFD60A", font=dict(color="#FFD60A", size=11), row=2, col=1
            )
    for idx in patterns.get("frost_on_snow", []):
        if 0 <= idx < len(df):
            fig.add_annotation(
                x=df["date_str"].iloc[idx], y=hist_vals[idx], text="💣雪上加霜", showarrow=True,
                arrowhead=1, arrowcolor="#34C759", font=dict(color="#34C759", size=11), row=2, col=1
            )

# 默认视野定位到最近 90 根 K 线
total_len = len(df)
default_span = min(90, total_len)

fig.update_layout(
    template="plotly_dark", paper_bgcolor="#121214", plot_bgcolor="#121214",
    shapes=shapes,
    xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=10, b=8), height=620,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    dragmode="pan",
)
fig.update_xaxes(
    type="category",
    range=[total_len - default_span, total_len - 1],
    showgrid=False,
)
fig.update_yaxes(gridcolor="#222226", zerolinecolor="#44444a")

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

# 买卖点雷达面板
st.subheader("🎯 实时买卖点雷达")
trade_points = getattr(result, "trade_points", []) or []
signals = list(reversed(trade_points))[:6]
if not signals:
    st.caption("当前区间尚未形成可确认的缠论买卖点。")
else:
    for sig in signals:
        badge = "🟢" if getattr(sig, "side", "buy") == "buy" else "🔴"
        state = "观察态" if getattr(sig, "tentative", False) else "确定态"
        price_val = float(getattr(sig, "price", 0.0))
        st.info(
            f"{badge} **{getattr(sig, 'display', '信号')}** · {state} · {getattr(sig, 'date', '')} · {price_val:.2f}\n\n"
            f"依据：{getattr(sig, 'reason', '') or '缠论结构判定'}"
        )
