"""StockChan 移动端 Web / PWA 入口。

本文件只编排既有 ``data`` 与 ``core`` 模块，不修改任何缠论计算或行情
抓取逻辑。部署至 HTTPS 云服务后可作为 iPhone「添加到主屏幕」的 Web App。
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from core import ChanAnalyzer
from core.advanced_indicators import (
    calculate_chop_filter,
    compute_anchored_vwap,
    detect_duck_head,
    detect_macd_patterns,
)
from data import fetch, fetch_min, guess_kind, looks_like_code, resolve_name


st.set_page_config(
    page_title="StockChan 移动量化",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def install_pwa_bridge() -> None:
    """把 manifest、Apple 图标和 service worker 注册到 Streamlit 外层页面。"""
    components.html(
        r"""
        <script>
        (() => {
          const host = window.parent;
          const base = host.location.pathname.replace(/\/?$/, "/");
          const ensureLink = (rel, href) => {
            if (!host.document.querySelector(`link[rel="${rel}"]`)) {
              const link = host.document.createElement("link");
              link.rel = rel; link.href = href; host.document.head.appendChild(link);
            }
          };
          ensureLink("manifest", base + "manifest.webmanifest");
          ensureLink("apple-touch-icon", base + "stockchan-icon.svg");
          host.document.documentElement.style.background = "#121214";
          if ("serviceWorker" in host.navigator) {
            host.navigator.serviceWorker.register(base + "sw.js", {scope: base}).catch(() => {});
          }
        })();
        </script>
        """,
        height=0,
    )


install_pwa_bridge()

st.markdown(
    """
    <style>
      .stApp { background: #121214; color: #f5f5f7; }
      [data-testid="stHeader"] { background: rgba(18,18,20,.92); }
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
      div[data-testid="stMetric"] { background:#1c1c1f; border:1px solid #303036;
        border-radius:14px; padding:10px 14px; }
      @media (max-width: 640px) {
        .block-container { padding: .7rem .7rem 1.5rem; }
        h1 { font-size: 1.35rem !important; }
      }
    </style>
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
    """支持代码、中文名与拼音联想；最终始终交给现有数据层处理。"""
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
    """30 秒缓存：减小云端请求频率，计算仍复用原有纯 Python 模块。"""
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
with st.form("quote_query", border=False):
    col1, col2, col3, col4 = st.columns([2.1, 1, 1, 0.8])
    with col1:
        typed_symbol = st.text_input("代码或名称", value="601899", placeholder="例如：贵州茅台、510300")
    with col2:
        kind_name = st.selectbox("类型", list(KIND_MAP), index=0)
    with col3:
        period_name = st.selectbox("周期", list(PERIOD_MAP), index=0)
    with col4:
        submitted = st.form_submit_button("加载行情", use_container_width=True)

if submitted:
    st.cache_data.clear()

try:
    kind = KIND_MAP[kind_name]
    symbol, kind = normalize_input(typed_symbol, kind)
    minute_period = PERIOD_MAP[period_name]
    # 日线三年起；分钟接口的既有实现自己限制为近期窗口。
    start = f"{date.today().year - 3}{date.today():%m%d}"
    with st.spinner("正在同步行情与指标…"):
        df, result, hist, patterns, chop, duck, avwap = get_analysis_data(
            symbol, kind, minute_period, start
        )
except Exception as exc:  # 数据源错误以页面提示呈现，不让 Web 服务崩溃
    st.error(f"行情解析失败或连接超时：{type(exc).__name__}: {exc}")
    st.stop()

latest_chop = chop.iloc[-1] if not chop.empty else float("nan")
regime = "🌪️ 无序震荡·关闸" if pd.notna(latest_chop) and latest_chop > 61.8 else "🚀 单边趋势/过渡"
metric1, metric2, metric3 = st.columns(3)
metric1.metric("最新收盘", f"{float(df['close'].iloc[-1]):.2f}")
metric2.metric("CHOP", "—" if pd.isna(latest_chop) else f"{latest_chop:.1f}")
metric3.metric("市场状态", regime)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
    row_heights=[0.74, 0.26],
)
fig.add_trace(go.Candlestick(
    x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="K线", increasing_line_color="#FF453A", decreasing_line_color="#30D158",
), row=1, col=1)
valid_avwap = avwap.dropna()
if not valid_avwap.empty:
    fig.add_trace(go.Scatter(
        x=df.loc[valid_avwap.index, "date"], y=valid_avwap, mode="lines", name="定锚 VWAP",
        line=dict(color="#FF9500", width=1.5, dash="dash"),
    ), row=1, col=1)

colors = ["#FF453A" if value >= 0 else "#00F0FF" for value in hist]
fig.add_trace(go.Bar(x=df["date"], y=hist, marker_color=colors, name="MACD 动量"), row=2, col=1)
for index in patterns.get("air_refuel", []):
    if index < len(df):
        fig.add_annotation(x=df["date"].iloc[index], y=hist.iloc[index], text="⚡加油", showarrow=True,
                           arrowhead=1, arrowcolor="#FFD60A", font=dict(color="#FFD60A", size=12), row=2, col=1)
for index in patterns.get("frost_on_snow", []):
    if index < len(df):
        fig.add_annotation(x=df["date"].iloc[index], y=hist.iloc[index], text="💣雪上加霜", showarrow=True,
                           arrowhead=1, arrowcolor="#FF453A", font=dict(color="#FF453A", size=12), row=2, col=1)

fig.update_layout(
    template="plotly_dark", paper_bgcolor="#121214", plot_bgcolor="#121214",
    xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=12, b=8), height=610,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    dragmode="pan",
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="#2a2a2e", zerolinecolor="#55555c")
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

st.subheader("🎯 实时买卖点雷达")
signals = list(reversed(result.trade_points))[:5]
if not signals:
    st.caption("当前区间尚未形成可确认的缠论买卖点。")
for signal in signals:
    badge = "🟢" if getattr(signal, "side", "buy") == "buy" else "🔴"
    state = "观察态" if getattr(signal, "tentative", False) else "确定态"
    st.info(
        f"{badge} **{signal.display}** · {state} · {signal.date} · {float(signal.price):.2f}\n\n"
        f"依据：{signal.reason or '缠论结构判定'}"
    )

duck_count = int(duck.get("duck_head", pd.Series(dtype=bool)).sum())
st.caption(f"{symbol} · {period_name} · 空中加油 {len(patterns['air_refuel'])} 次 · 雪上加霜 {len(patterns['frost_on_snow'])} 次 · 老鸭头 {duck_count} 次")
