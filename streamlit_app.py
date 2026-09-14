"""StockChan 移动端 Web / PWA 入口。

专业浅色模式 (Light Theme)：高对比度护眼白底，清晰呈现缠论中枢、笔线段、买卖点及动量副图。
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
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# 1. 核心缠论模块兼容导入
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

# 2. 数据获取模块兼容导入
try:
    from data.fetcher import fetch, fetch_min, guess_kind, looks_like_code, resolve_name
except (ImportError, ModuleNotFoundError):
    try:
        from data import fetch, fetch_min, guess_kind, looks_like_code, resolve_name
    except (ImportError, ModuleNotFoundError):
        from fetcher import fetch, fetch_min, guess_kind, looks_like_code, resolve_name

st.set_page_config(
    page_title="StockChan 移动量化",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 注入浅色 PWA 桥接与优雅浅色 CSS
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


def calculate_indicators(frame: pd.DataFrame):
    """计算牛熊分界线、肯特纳通道与 Squeeze 动量"""
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    
    # 均线牛熊线 (EMA20 & EMA60)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    
    # 肯特纳通道 (KC: 20周期 EMA +/- 1.5倍 ATR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean()
    kc_upper = ema20 + 1.5 * atr20
    kc_lower = ema20 - 1.5 * atr20
    
    # 布林带挤压状态
    std20 = close.rolling(20).std()
    bb_upper = ema20 + 2.0 * std20
    bb_lower = ema20 - 2.0 * std20
    is_squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # Squeeze 动量
    highest_20 = high.rolling(20).max()
    lowest_20 = low.rolling(20).min()
    mid = (highest_20 + lowest_20) / 2 + ema20
    mid = mid / 2
    delta = close - mid
    squeeze_val = delta.rolling(20).mean()
    
    return ema20, ema60, kc_upper, kc_lower, is_squeeze, squeeze_val


@st.cache_data(ttl=60, show_spinner=False)
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

# 2. 叠加显示控制
with st.expander("🛠️ 叠加显示与指标配置", expanded=False):
    r1_1, r1_2, r1_3, r1_4, r1_5 = st.columns(5)
    with r1_1:
        show_bi = st.checkbox("笔 / 线段", value=True)
    with r1_2:
        show_zs = st.checkbox("中枢 (ZG/ZD)", value=True)
    with r1_3:
        show_signals = st.checkbox("买卖点标记", value=True)
    with r1_4:
        show_bull_bear = st.checkbox("牛熊分界线", value=True)
    with r1_5:
        show_kc = st.checkbox("肯特纳通道", value=False)
        
    r2_1, r2_2 = st.columns([1, 1])
    with r2_1:
        show_vwap = st.checkbox("定锚 VWAP", value=False)
    with r2_2:
        subchart_type = st.selectbox("副图指标", ["MACD", "Squeeze 动量"], index=0)

if submitted:
    st.cache_data.clear()

try:
    kind = KIND_MAP[kind_name]
    symbol, kind = normalize_input(typed_symbol, kind)
    minute_period = PERIOD_MAP[period_name]
    start_dt = date.today() - timedelta(days=400)
    start = f"{start_dt:%Y%m%d}"
    with st.spinner("正在计算缠论中枢与动量指标…"):
        df, result, hist, patterns, chop, duck, avwap = get_analysis_data(
            symbol, kind, minute_period, start
        )
except Exception as exc:
    st.error(f"行情解析失败：{type(exc).__name__}: {exc}")
    st.stop()

# 计算指标
ema20, ema60, kc_upper, kc_lower, is_squeeze, squeeze_val = calculate_indicators(df)

latest_chop = chop.iloc[-1] if not chop.empty else float("nan")
regime = "🌪️ 无序震荡·关闸" if pd.notna(latest_chop) and latest_chop > 61.8 else "🚀 单边趋势/过渡"
metric1, metric2, metric3 = st.columns(3)
metric1.metric("最新收盘", f"{float(df['close'].iloc[-1]):.2f}")
metric2.metric("CHOP", "—" if pd.isna(latest_chop) else f"{latest_chop:.1f}")
metric3.metric("市场状态", regime)

# 日期格式化
date_format = "%Y-%m-%d %H:%M" if minute_period else "%Y-%m-%d"
df["date_str"] = df["date"].dt.strftime(date_format)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.74, 0.26],
)

# 1. K线图（白底浅色方案：纯正 A 股红涨绿跌）
fig.add_trace(go.Candlestick(
    x=df["date_str"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="K线",
    increasing_line_color="#EB4436", increasing_fillcolor="#EB4436",
    decreasing_line_color="#0FA958", decreasing_fillcolor="#0FA958",
), row=1, col=1)

# 2. 牛熊分界线 (白底下醒目的橙/紫线条)
if show_bull_bear:
    fig.add_trace(go.Scatter(
        x=df["date_str"], y=ema20, mode="lines",
        line=dict(color="#EA580C", width=1.4), name="EMA20 快线"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["date_str"], y=ema60, mode="lines",
        line=dict(color="#7C3AED", width=1.6), name="EMA60 牛熊线"
    ), row=1, col=1)

# 3. 肯特纳通道
if show_kc:
    fig.add_trace(go.Scatter(
        x=df["date_str"], y=kc_upper, mode="lines",
        line=dict(color="rgba(14, 165, 233, 0.7)", width=1, dash="dot"), name="KC上轨"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["date_str"], y=kc_lower, mode="lines",
        line=dict(color="rgba(14, 165, 233, 0.7)", width=1, dash="dot"), name="KC下轨"
    ), row=1, col=1)

# 4. 缠论中枢矩形（浅天蓝半透明框）
shapes = []
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
        elif isinstance(start_d, (int, np.integer)) and 0 <= start_d < len(df):
            start_d = df["date"].iloc[start_d]
            
        if hasattr(end_d, "date"):
            end_d = end_d.date
        elif isinstance(end_d, (int, np.integer)) and 0 <= end_d < len(df):
            end_d = df["date"].iloc[end_d]
            
        if zg is not None and zd is not None and start_d and end_d:
            try:
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=pd.to_datetime(start_d).strftime(date_format),
                    x1=pd.to_datetime(end_d).strftime(date_format),
                    y0=float(zd), y1=float(zg),
                    fillcolor="rgba(37, 99, 235, 0.12)",
                    line=dict(color="#2563EB", width=1.5, dash="dash"),
                ))
            except Exception:
                pass

# 5. 缠论笔（浅底下采用深金黄 / 琥珀色，对比度极佳）
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
                bi_x.append(pd.to_datetime(sd).strftime(date_format))
                bi_y.append(float(sv))
            if ed and ev is not None:
                bi_x.append(pd.to_datetime(ed).strftime(date_format))
                bi_y.append(float(ev))
    if bi_x:
        fig.add_trace(go.Scatter(
            x=bi_x, y=bi_y, mode="lines+markers",
            line=dict(color="#D97706", width=2.2),
            marker=dict(size=4, color="#D97706"),
            name="缠论笔",
        ), row=1, col=1)

# 6. 买卖点标牌
if show_signals:
    trade_points = getattr(result, "trade_points", []) or []
    for sig in trade_points:
        sig_d = getattr(sig, "date", None)
        if not sig_d:
            continue
        try:
            d_str = pd.to_datetime(sig_d).strftime(date_format)
            sig_price = float(getattr(sig, "price", 0.0))
            is_buy = getattr(sig, "side", "buy") == "buy"
            disp = getattr(sig, "display", "信号")
            fig.add_annotation(
                x=d_str, y=sig_price,
                text=f"{'▲' if is_buy else '▼'}{disp}",
                showarrow=True, arrowhead=1,
                arrowcolor="#0FA958" if is_buy else "#EB4436",
                ay=24 if is_buy else -24,
                font=dict(color="#FFFFFF", size=11, family="Arial Black"),
                bgcolor="#0FA958" if is_buy else "#EB4436",
                borderpad=3,
                row=1, col=1
            )
        except Exception:
            pass

# 7. 定锚 VWAP
if show_vwap:
    valid_avwap = avwap.dropna()
    if not valid_avwap.empty:
        fig.add_trace(go.Scatter(
            x=df.loc[valid_avwap.index, "date_str"], y=valid_avwap, mode="lines", name="定锚 VWAP",
            line=dict(color="#D97706", width=1.5, dash="dash"),
        ), row=1, col=1)

# 8. 副图指标
if subchart_type == "Squeeze 动量":
    sqz_colors = ["#EB4436" if v >= 0 else "#0FA958" for v in squeeze_val.fillna(0.0)]
    fig.add_trace(go.Bar(
        x=df["date_str"], y=squeeze_val, marker_color=sqz_colors, name="Squeeze动量"
    ), row=2, col=1)
    squeeze_dots = np.zeros(len(df))
    dot_colors = ["#111827" if sq else "#9CA3AF" for sq in is_squeeze]
    fig.add_trace(go.Scatter(
        x=df["date_str"], y=squeeze_dots, mode="markers",
        marker=dict(size=4, color=dot_colors), name="挤压状态"
    ), row=2, col=1)
else:
    hist_vals = hist.fillna(0.0).tolist()
    macd_colors = ["#EB4436" if v >= 0 else "#0FA958" for v in hist_vals]
    fig.add_trace(go.Bar(
        x=df["date_str"], y=hist_vals, marker_color=macd_colors, name="MACD"
    ), row=2, col=1)

total_len = len(df)
view_span = min(80, total_len)

# 纯净浅色专业金融画布 (Light Clean Layout)
fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    shapes=shapes,
    xaxis_rangeslider_visible=False,
    margin=dict(l=8, r=8, t=10, b=8),
    height=600,
    font=dict(color="#374151", family="sans-serif"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(color="#4b5563")),
    dragmode="pan",
)

nticks_val = 10 if total_len > 20 else total_len
fig.update_xaxes(
    type="category",
    range=[total_len - view_span, total_len - 1],
    showgrid=True,
    gridcolor="#f3f4f6",
    nticks=nticks_val,
    tickfont=dict(color="#6b7280"),
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="#f3f4f6",
    zerolinecolor="#e5e7eb",
    tickfont=dict(color="#6b7280"),
)

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

# 信号雷达面板（浅色卡片化呈现）
st.subheader("🎯 实时买卖点雷达")
trade_points = getattr(result, "trade_points", []) or []
signals = list(reversed(trade_points))[:6]
if not signals:
    st.caption("当前区间尚未形成可确认的缠论买卖点。")
else:
    for sig in signals:
        is_b = getattr(sig, "side", "buy") == "buy"
        badge = "🟢" if is_b else "🔴"
        state = "观察态" if getattr(sig, "tentative", False) else "确定态"
        
        box_bg = "#f0fdf4" if is_b else "#fef2f2"
        box_border = "#bbf7d0" if is_b else "#fecaca"
        title_color = "#15803d" if is_b else "#b91c1c"
        
        st.markdown(
            f"""
