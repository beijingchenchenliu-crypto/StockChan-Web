"""StockChan 移动端 Web / PWA 看板（移动端极致优化 + 防拷贝截断版）。

1:1 还原桌面版原生功能，同时针对 iOS Safari 严重卡顿发热问题进行底层优化：
- 渲染层截断：仅渲染最近 180 根 K 线，解放手机 GPU
- 杜绝语法崩溃：100% 消除所有硬编码 HTML 尖括号，杜绝剪贴板/浏览器翻译插件破坏代码
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# 核心计算层导入
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
    from core.channel import line_value
except (ImportError, ModuleNotFoundError):
    def line_value(line_dict: dict, x_val: float) -> float:
        x1 = line_dict.get("x1", 0)
        y1 = line_dict.get("y1", 0.0)
        slope = line_dict.get("slope", 0.0)
        return float(y1 + slope * (x_val - x1))

# 数据接口导入
try:
    from data.fetcher import fetch, fetch_min, guess_kind, looks_like_code, resolve_name
except (ImportError, ModuleNotFoundError):
    try:
        from data import fetch, fetch_min, guess_kind, looks_like_code, resolve_name
    except (ImportError, ModuleNotFoundError):
        from fetcher import fetch, fetch_min, guess_kind, looks_like_code, resolve_name

# 全局安全 HTML 标签生成（绝不使用硬编码尖括号，彻底避免复制时被转义截断）
HTML_BR = chr(60) + "br" + chr(62)
HTML_B = chr(60) + "b" + chr(62)
HTML_B_END = chr(60) + "/b" + chr(62)

st.set_page_config(
    page_title="StockChan 股票与指数缠论分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 浅色金融看板样式
st.markdown(
    chr(60) + "style" + chr(62) + """
      .stApp { background: #FFFFFF; color: #1E293B; }
      [data-testid="stHeader"] { background: rgba(255, 255, 255, 0.95); }
      .block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1560px; }
      
      div[data-testid="stMetric"] { 
        background: #F8FAFC; 
        border: 1px solid #E2E8F0;
        border-radius: 8px; 
        padding: 6px 12px;
      }
      div[data-testid="stMetricLabel"] p { color: #64748B !important; font-size: 0.8rem; }
      div[data-testid="stMetricValue"] div { color: #0F172A !important; font-size: 1.25rem !important; font-weight: 700; }
      
      .stTextInput>div>div>input { background-color: #F8FAFC !important; color: #0F172A !important; border-color: #CBD5E1 !important; }
      .stSelectbox>div>div { background-color: #F8FAFC !important; }
      
      @media (max-width: 640px) {
        .block-container { padding: 0.3rem; }
        h1 { font-size: 1.15rem !important; }
      }
    """ + chr(60) + "/style" + chr(62),
    unsafe_allow_html=True,
)

KIND_MAP = {"指数": "index", "个股": "stock", "ETF": "etf"}

PERIOD_MAP: dict[str, dict] = {
    "60分钟": {"type": "min", "val": "60", "key": "min"},
    "30分钟": {"type": "min", "val": "30", "key": "min"},
    "15分钟": {"type": "min", "val": "15", "key": "min"},
    "5分钟": {"type": "min", "val": "5", "key": "min"},
    "日线": {"type": "kline", "val": "日线", "key": "D"},
    "周线": {"type": "kline", "val": "周线", "key": "W"},
    "月线": {"type": "kline", "val": "月线", "key": "M"},
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
        raise ValueError("未能解析该代码。")
    return hit.code, hit.kind if hit.kind in KIND_MAP.values() else selected_kind


# ===================== 顶部控制栏 =====================
st.caption("⚙️ StockChan · 股票与指数缠论分析工具")
c1, c2, c3, c4, c5, c6 = st.columns([1, 1.3, 1, 1.2, 0.7, 1])
with c1:
    kind_name = st.selectbox("类型", list(KIND_MAP), index=0)
with c2:
    typed_symbol = st.text_input("代码", value="000001")
with c3:
    period_name = st.selectbox("周期", list(PERIOD_MAP), index=4)
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
    show_gaps = st.checkbox("跳空缺口", value=True)
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
    struct_level = st.selectbox("结构级别", ["主结构 + 局部", "仅主结构", "仅局部结构"], index=0)
with q2:
    lookback_strength = st.selectbox("回看强度", ["标准", "深度"], index=0)
with q3:
    subchart_choice = st.selectbox("副图", ["Squeeze 动量", "MACD"], index=0)
with q4:
    divergence_thresh = st.number_input("背驰阈值", value=0.85, step=0.05, format="%.2f")

# 行情抓取与计算
try:
    kind = KIND_MAP[kind_name]
    symbol, kind = normalize_input(typed_symbol, kind)
    p_info = PERIOD_MAP[period_name]
    start_str = "20180101" if is_full_cycle else start_date_val.strftime("%Y%m%d")
    
    with st.spinner("正在加载多周期行情并推演缠论几何…"):
        if p_info["type"] == "min":
            df = fetch_min(symbol=symbol, period=p_info["val"], start_date=start_str)
            period_key = p_info["key"]
            min_p = p_info["val"]
        else:
            df = fetch(symbol=symbol, kind=kind, period=p_info["val"], start_date=start_str)
            period_key = p_info["key"]
            min_p = None
            
        if df is None or df.empty:
            raise ValueError("未能获取到数据，请核对代码。")
            
        frame = df.copy().reset_index(drop=True)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
        frame = frame.sort_values("date").reset_index(drop=True)
        
        try:
            result = ChanAnalyzer(frame, period_key=period_key, min_period=min_p).run()
        except TypeError:
            result = ChanAnalyzer(frame).run()

    indicators = getattr(result, "indicators", {}) or {}
    chop = calculate_chop_filter(frame)
    anchor_idx = int(frame["low"].astype(float).idxmin())
    avwap = compute_anchored_vwap(frame, anchor_idx)
except Exception as exc:
    st.error(f"加载出错：{type(exc).__name__}: {exc}")
    st.stop()

# 状态指标栏
latest_chop = chop.iloc[-1] if not chop.empty else float("nan")
regime = "🌪️ 无序震荡·关闸" if pd.notna(latest_chop) and latest_chop > 61.8 else "🚀 单边趋势/过渡"
m1, m2, m3 = st.columns(3)
m1.metric("最新收盘", f"{float(frame['close'].iloc[-1]):.2f}")
m2.metric("CHOP", "—" if pd.isna(latest_chop) else f"{latest_chop:.1f}")
m3.metric("市场状态", regime)

total_bars = len(frame)

# ================= 渲染层性能极速优化：仅截取绘制最近 180 根 K 线 =================
RENDER_LIMIT = 180
r_start = max(0, total_bars - RENDER_LIMIT)

x_indices = np.arange(total_bars)
date_format = "%Y-%m-%d %H:%M" if p_info["type"] == "min" else "%Y-%m-%d"
date_labels = frame["date"].dt.strftime(date_format).tolist()

x_render = x_indices[r_start:]
frame_render = frame.iloc[r_start:]

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.76, 0.24],
)

# 1. 蜡烛图
fig.add_trace(go.Candlestick(
    x=x_render, open=frame_render["open"], high=frame_render["high"], 
    low=frame_render["low"], close=frame_render["close"], name="K线",
    increasing_line_color="#FF453A", increasing_fillcolor="#FF453A",
    decreasing_line_color="#26D0B8", decreasing_fillcolor="#26D0B8",
), row=1, col=1)

# 2. 牛熊分界线 (EMA60 & EMA20)
if show_bull_bear:
    c_series = frame["close"].astype(float)
    bull_bear_series = c_series.ewm(span=60, adjust=False).mean().iloc[r_start:]
    fig.add_trace(go.Scatter(
        x=x_render, y=bull_bear_series, mode="lines",
        line=dict(color="#7B1FA2", width=2.2), name="牛熊分界线", hoverinfo="skip"
    ), row=1, col=1)
    
    ema20_series = c_series.ewm(span=20, adjust=False).mean().iloc[r_start:]
    fig.add_trace(go.Scatter(
        x=x_render, y=ema20_series, mode="lines",
        line=dict(color="#F76707", width=1.4), name="EMA20", hoverinfo="skip"
    ), row=1, col=1)

# 3. 肯特纳通道 (KC)
if show_kc:
    keltner_df = indicators.get("keltner")
    if isinstance(keltner_df, pd.DataFrame) and {"mid", "upper", "lower"}.issubset(keltner_df.columns):
        fig.add_trace(go.Scatter(x=x_render, y=keltner_df["mid"].iloc[r_start:], mode="lines", line=dict(color="#FF9800", width=1.5), name="KC中轨", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_render, y=keltner_df["upper"].iloc[r_start:], mode="lines", line=dict(color="#8E24AA", width=1.2, dash="dash"), name="KC上轨", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_render, y=keltner_df["lower"].iloc[r_start:], mode="lines", line=dict(color="#8E24AA", width=1.2, dash="dash"), name="KC下轨", hoverinfo="skip"), row=1, col=1)

# 4. 定锚 VWAP
if show_vwap:
    valid_avwap = avwap.dropna()
    valid_avwap = valid_avwap[valid_avwap.index >= r_start]
    if not valid_avwap.empty:
        fig.add_trace(go.Scatter(
            x=valid_avwap.index.to_numpy(), y=valid_avwap.to_numpy(),
            mode="lines", line=dict(color="#FF9500", width=1.8, dash="dash"), name="定锚VWAP", hoverinfo="skip"
        ), row=1, col=1)

shapes = []

# 5. 跳空缺口 (仅渲染可视范围内)
if show_gaps and total_bars >= 2:
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    for i in (np.flatnonzero(lows[1:] > highs[:-1]) + 1):
        if i < r_start: continue
        gap_low, gap_high = float(highs[i - 1]), float(lows[i])
        rest = np.flatnonzero(lows[i + 1:] <= gap_low)
        fill_idx = int(rest[0]) + i + 1 if rest.size else total_bars
        shapes.append(dict(
            type="rect", xref="x", yref="y", x0=i - 0.3, x1=fill_idx, y0=gap_low, y1=gap_high,
            fillcolor="rgba(239, 83, 80, 0.18)", line=dict(color="#EF5350", width=1, dash="dash"),
        ))
    for i in (np.flatnonzero(highs[1:] < lows[:-1]) + 1):
        if i < r_start: continue
        gap_high, gap_low = float(lows[i - 1]), float(highs[i])
        rest = np.flatnonzero(highs[i + 1:] >= gap_high)
        fill_idx = int(rest[0]) + i + 1 if rest.size else total_bars
        shapes.append(dict(
            type="rect", xref="x", yref="y", x0=i - 0.3, x1=fill_idx, y0=gap_low, y1=gap_high,
            fillcolor="rgba(38, 166, 154, 0.18)", line=dict(color="#26A69A", width=1, dash="dash"),
        ))

# 6. 缠论中枢矩形
if show_zs:
    pivots = getattr(result, "zs_list", None) or getattr(result, "pivots", None) or getattr(result, "bi_zs_list", None) or []
    for p in pivots:
        if hasattr(p, "start_index"):
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
            
        if x1 < r_start: continue
        
        if np.isfinite(zd) and np.isfinite(zg) and zg > zd:
            shapes.append(dict(
                type="rect", xref="x", yref="y", x0=x0 - 0.2, x1=x1 + 0.2, y0=zd, y1=zg,
                fillcolor="rgba(255, 152, 0, 0.25)", line=dict(color="#F57C00", width=1.4, dash="dash"),
            ))

# 7. 缠论笔
strokes = getattr(result, "strokes", None) or getattr(result, "bi_list", None) or []
bi_points = []
if strokes:
    for s in strokes:
        if hasattr(s, "start") and hasattr(s, "end"):
            st_idx = getattr(s.start, "raw_index", getattr(s.start, "index", 0))
            st_p = getattr(s.start, "price", getattr(s.start, "val", 0.0))
            end_idx = getattr(s.end, "raw_index", getattr(s.end, "index", 0))
            end_p = getattr(s.end, "price", getattr(s.end, "val", 0.0))
            if not bi_points:
                bi_points.append((int(st_idx), float(st_p)))
            bi_points.append((int(end_idx), float(end_p)))
        elif isinstance(s, dict):
            idx = s.get("index") or s.get("raw_index")
            p = s.get("price") or s.get("val")
            if idx is not None and p is not None:
                bi_points.append((int(idx), float(p)))

if show_bi and len(bi_points) >= 2:
    bp_render = [p for p in bi_points if p[0] >= r_start - 10]
    if bp_render:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in bp_render], y=[p[1] for p in bp_render],
            mode="lines+markers", line=dict(color="#A0A0A6", width=1.6),
            marker=dict(size=4, color="#A0A0A6"), name="笔", hoverinfo="skip"
        ), row=1, col=1)

# 支撑阻力位算法
close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
highs_arr = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
lows_arr = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
last_valid_idx = total_bars - 1
curr_price = float(close_arr[last_valid_idx])

lookback_len = max(40, min(total_bars, 180))
start_p_idx = max(0, last_valid_idx - lookback_len + 1)
rad = 2 if lookback_len < 90 else 3
sup_cands, res_cands = [], []

for i in range(start_p_idx + rad, last_valid_idx - rad + 1):
    lo_w = lows_arr[i - rad:i + rad + 1]
    hi_w = highs_arr[i - rad:i + rad + 1]
    if lows_arr[i] <= np.nanmin(lo_w):
        sup_cands.append((float(lows_arr[i]), i))
    if highs_arr[i] >= np.nanmax(hi_w):
        res_cands.append((float(highs_arr[i]), i))

for idx, price in bi_points[-16:]:
    if price < curr_price:
        sup_cands.append((float(price), int(idx)))
    elif price > curr_price:
        res_cands.append((float(price), int(idx)))

tol = max(curr_price * 0.004, float(np.nanstd(close_arr[start_p_idx:last_valid_idx + 1])) * 0.35, 1e-6)

def pick_distinct(cands, reverse=False):
    ordered = sorted(cands, key=lambda it: abs(it[0] - curr_price))
    chosen = []
    for p, i in ordered:
        if any(abs(p - o[0]) <= tol for o in chosen):
            continue
        chosen.append((p, i))
        if len(chosen) == 2:
            break
    return sorted(chosen, key=lambda it: it[0], reverse=reverse)

sups = pick_distinct([(p, i) for p, i in sup_cands if p < curr_price * 0.9995], reverse=True)
ress = pick_distinct([(p, i) for p, i in res_cands if p > curr_price * 1.0005], reverse=False)
x_ext_end = total_bars + 15

# 绘制压力虚线
for rank, (p_val, i_idx) in enumerate(ress, 1):
    fig.add_trace(go.Scatter(
        x=[max(r_start, i_idx), x_ext_end], y=[p_val, p_val], mode="lines", 
        line=dict(color="#D32F2F", width=1.5, dash="dash"), showlegend=False, hoverinfo="skip"
    ), row=1, col=1)
    
    line_text = HTML_B + "压力" + str(rank) + ": " + f"{p_val:.2f}" + HTML_B_END
    fig.add_annotation(
        x=x_ext_end, y=p_val, text=line_text,
        showarrow=False, font=dict(color="#D32F2F", size=11, family="Arial Black"),
        xanchor="left", yanchor="bottom", row=1, col=1
    )

# 绘制支撑虚线
for rank, (p_val, i_idx) in enumerate(sups, 1):
    fig.add_trace(go.Scatter(
        x=[max(r_start, i_idx), x_ext_end], y=[p_val, p_val], mode="lines", 
        line=dict(color="#1976D2", width=1.5, dash="dash"), showlegend=False, hoverinfo="skip"
    ), row=1, col=1)
    
    line_text = HTML_B + "支撑" + str(rank) + ": " + f"{p_val:.2f}" + HTML_B_END
    fig.add_annotation(
        x=x_ext_end, y=p_val, text=line_text,
        showarrow=False, font=dict(color="#1976D2", size=11, family="Arial Black"),
        xanchor="left", yanchor="top", row=1, col=1
    )

# 8. 波浪标号
waves = getattr(result, "waves", None) or []
if show_wave and waves:
    wave_points = []
    for w in waves:
        w_idx = getattr(w, "raw_index", getattr(w, "index", None))
        w_price = getattr(w, "price", None)
        if w_idx is not None and w_price is not None and int(w_idx) >= r_start:
            wave_points.append((int(w_idx), float(w_price), str(getattr(w, "label", "")), str(getattr(w, "kind", ""))))
    if wave_points:
        wave_points.sort(key=lambda item: item[0])
        fig.add_trace(go.Scatter(
            x=[wp[0] for wp in wave_points], y=[wp[1] for wp in wave_points],
            mode="lines", line=dict(color="#6A1B9A", width=1.8, dash="dash"), name="波浪", hoverinfo="skip"
        ), row=1, col=1)
        for w_idx, w_price, w_label, w_kind in wave_points:
            is_top = w_kind.lower() in {"top", "高点", "peak"}
            
            wave_text = HTML_B + str(w_label) + HTML_B_END
            fig.add_annotation(
                x=w_idx, y=w_price, text=wave_text, showarrow=False,
                font=dict(color="#8E24AA" if is_top else "#1565C0", size=13, family="Arial Black"),
                yshift=14 if is_top else -14, row=1, col=1
            )

# 9. 形态通道 (采用纯加法拼接，绝对不使用任何引发断行的语法)
channel = getattr(result, "channel", None)
if show_channel and channel and getattr(channel, "valid", False):
    items_to_draw = []
    p_struct = getattr(channel, "primary_structure", None)
    s_struct = getattr(channel, "secondary_structure", None)
    
    if struct_level == "仅主结构" and p_struct: items_to_draw.append((p_struct, True))
    elif struct_level == "仅局部结构" and s_struct: items_to_draw.append((s_struct, False))
    else:
        if p_struct: items_to_draw.append((p_struct, True))
        if s_struct: items_to_draw.append((s_struct, False))
            
    for struct, is_primary in items_to_draw:
        st_state = str(getattr(struct, "status", "candidate"))
        up = getattr(struct, "upper_line", {})
        lo = getattr(struct, "lower_line", {})
        if up and lo:
            color_up = "#D32F2F" if st_state == "confirmed" else "#E65100"
            color_lo = "#1976D2" if st_state == "confirmed" else "#0097A7"
            span = max(up.get("x2", 0) - up.get("x1", 0), lo.get("x2", 0) - lo.get("x1", 0), 20)
            ext = int(min(span * 0.2, 15))
            x_end = min(total_bars - 1 + ext, total_bars + 15)
            
            y_u_start = line_value(up, max(r_start, up["x1"]))
            y_u_end = line_value(up, x_end)
            y_l_start = line_value(lo, max(r_start, lo["x1"]))
            y_l_end = line_value(lo, x_end)
            
            fig.add_trace(go.Scatter(
                x=[max(r_start, up["x1"]), x_end], y=[y_u_start, y_u_end],
                mode="lines", line=dict(color=color_up, width=2.0 if is_primary else 1.4, dash="dash"), showlegend=False, hoverinfo="skip"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[max(r_start, lo["x1"]), x_end], y=[y_l_start, y_l_end],
                mode="lines", line=dict(color=color_lo, width=2.0 if is_primary else 1.4, dash="dash"), showlegend=False, hoverinfo="skip"
            ), row=1, col=1)
            
            prefix_str = "【主】" if is_primary else "【局】"
            lbl_str = str(getattr(struct, "label", "整理"))
            b_up = float(getattr(struct, "breakout_up_level", 0.0))
            b_down = float(getattr(struct, "breakdown_level", 0.0))
            
            # 使用最基础的字符串相加，并使用预定义的 HTML_BR 变量
            line1 = prefix_str + " " + lbl_str
            line2 = "阻力: %.2f 支撑: %.2f" % (b_up, b_down)
            
            safe_channel_text = line1 + HTML_BR + line2
            
            fig.add_annotation(
                x=x_end, y=y_u_end, text=safe_channel_text,
                showarrow=False, font=dict(color=color_up, size=9),
                xanchor="left", yanchor="bottom" if is_primary else "top",
                align="left", row=1, col=1
            )

# 10. 买卖点大号五角星
if show_signals:
    signals = getattr(result, "signals", None) or getattr(result, "trade_points", None) or []
    price_span = float(frame_render["high"].max() - frame_render["low"].min())
    y_offset = price_span * 0.02
    
    for s in signals:
        is_tentative = getattr(s, "tentative", False)
        if only_confirmed and is_tentative: continue
        idx = getattr(s, "raw_index", getattr(s, "index", None))
        price = getattr(s, "price", None)
        if idx is None or price is None: continue
        try:
            idx = int(idx)
            if idx < r_start: continue
            price = float(price)
            is_buy = getattr(s, "side", "") == "buy"
            label = str(getattr(s, "label", getattr(s, "display", ""))).upper()
            color = "#FF453A" if is_buy else "#30D158"
            
            fig.add_trace(go.Scatter(
                x=[idx], y=[price], mode="markers",
                marker=dict(size=14, symbol="star", color=color, line=dict(color="#FFFFFF", width=1.5)), showlegend=False, hoverinfo="skip"
            ), row=1, col=1)
            
            sig_text = HTML_B + str(label) + HTML_B_END
            fig.add_annotation(
                x=idx, y=price - y_offset if is_buy else price + y_offset,
                text=sig_text, showarrow=False, font=dict(color=color, size=13, family="Arial Black"), row=1, col=1
            )
        except Exception:
            pass

# 11. Squeeze / MACD 副图
if subchart_choice == "Squeeze 动量":
    squeeze = indicators.get("squeeze")
    if isinstance(squeeze, pd.DataFrame) and "momentum" in squeeze.columns:
        m = pd.to_numeric(squeeze["momentum"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        on = np.asarray(squeeze.get("on", np.zeros(len(m), dtype=bool)), dtype=bool)
    else:
        c = frame["close"].astype(float)
        ema20 = c.ewm(span=20, adjust=False).mean()
        highs = frame["high"].astype(float)
        lows = frame["low"].astype(float)
        mid = (highs.rolling(20).max() + lows.rolling(20).min()) / 2 + ema20
        mid = mid / 2
        m = (c - mid).rolling(20).mean().fillna(0.0).to_numpy(dtype=float)
        on = np.zeros(len(m), dtype=bool)

    m_sub = m[r_start:]
    sqz_colors = ["#FF453A" if val >= 0 else "#26D0B8" for val in m_sub]
    
    fig.add_trace(go.Bar(
        x=x_render, y=m_sub, marker_color=sqz_colors, name="Squeeze动量"
    ), row=2, col=1)
    
    on_sub = on[r_start:]
    if on_sub.any():
        fig.add_trace(go.Scatter(
            x=x_render[on_sub], y=np.zeros(int(on_sub.sum())),
            mode="markers", marker=dict(size=5, color="#9A9A9F"), name="挤压状态", hoverinfo="skip"
        ), row=2, col=1)
        
    patterns = indicators.get("macd_patterns")
    if isinstance(patterns, dict):
        for idx in patterns.get("air_refuel", []):
            if r_start <= int(idx) < total_bars:
                fig.add_annotation(
                    x=int(idx), y=float(m[int(idx)]), text="⚡加油",
                    showarrow=True, arrowhead=1, arrowcolor="#FFD60A", font=dict(color="#FFD60A", size=11), row=2, col=1
                )
        for idx in patterns.get("frost_on_snow", []):
            if r_start <= int(idx) < total_bars:
                fig.add_annotation(
                    x=int(idx), y=float(m[int(idx)]), text="💣雪上加霜",
                    showarrow=True, arrowhead=1, arrowcolor="#FF453A", font=dict(color="#FF453A", size=11), row=2, col=1
                )
else:
    macd_df = indicators.get("macd")
    if isinstance(macd_df, pd.DataFrame) and "macd" in macd_df.columns:
        hist_vals = macd_df["macd"].fillna(0.0).tolist()
    else:
        c = frame["close"].astype(float)
        dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        dea = dif.ewm(span=9, adjust=False).mean()
        hist_vals = (2 * (dif - dea)).tolist()
        
    m_sub = hist_vals[r_start:]
    macd_colors = ["#FF453A" if v >= 0 else "#26D0B8" for v in m_sub]
    fig.add_trace(go.Bar(
        x=x_render, y=m_sub, marker_color=macd_colors, name="MACD"
    ), row=2, col=1)

# 动态视野设置
view_span = min(110, len(x_render))
y_min = float(frame_render.iloc[-view_span:]["low"].min()) * 0.99
y_max = float(frame_render.iloc[-view_span:]["high"].max()) * 1.01

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    shapes=shapes,
    xaxis_rangeslider_visible=False,
    margin=dict(l=8, r=8, t=10, b=8),
    height=660,
    font=dict(color="#475569", family="Segoe UI, sans-serif"),
    showlegend=False,
    hovermode="x",
    dragmode="pan",
)

tick_step = max(1, len(x_render) // 8)
tick_vals = list(range(r_start, total_bars, tick_step))
if (total_bars - 1) not in tick_vals:
    tick_vals.append(total_bars - 1)
tick_texts = [date_labels[i] for i in tick_vals]

fig.update_xaxes(
    tickvals=tick_vals,
    ticktext=tick_texts,
    range=[total_bars - view_span, total_bars + 18],
    showgrid=True,
    gridcolor="#F1F5F9",
)
fig.update_yaxes(
    range=[y_min, y_max],
    showgrid=True,
    gridcolor="#F1F5F9",
    zerolinecolor="#E2E8F0",
    row=1, col=1
)

st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})

# 信号雷达面板
st.subheader("🎯 实时买卖点雷达")
trade_points = getattr(result, "signals", None) or getattr(result, "trade_points", []) or []
signals_list = list(reversed(trade_points))[:6]
if not signals_list:
    st.caption("当前区间尚未形成可确认的缠论买卖点。")
else:
    for sig in signals_list:
        is_tentative = getattr(sig, "tentative", False)
        if only_confirmed and is_tentative:
            continue
        is_b = getattr(sig, "side", "buy") == "buy"
        badge = "🟢" if is_b else "🔴"
        state = "观察态" if is_tentative else "确定态"
        price_val = float(getattr(sig, "price", 0.0))
        sig_label = getattr(sig, "label", getattr(sig, "display", "信号"))
        disp_txt = f"{badge} **{sig_label}** · {state} · {getattr(sig, 'date', '')} · {price_val:.2f}"
        reason_txt = f"依据：{getattr(sig, 'reason', '') or '缠论结构判定'}"
        if is_b:
            st.success(f"{disp_txt}\n\n{reason_txt}")
        else:
            st.error(f"{disp_txt}\n\n{reason_txt}")
