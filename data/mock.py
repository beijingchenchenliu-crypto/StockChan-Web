"""离线演示数据。

用途：在无网络 / akshare 接口临时不可用时，仍然能启动界面并验证缠论引擎
与绘图链路是否正常。生成一条带趋势、带震荡段的合成价格序列。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fetcher import COLUMNS


def make_mock_daily(n: int = 600, seed: int = 20260913, start: str = "2023-01-03") -> pd.DataFrame:
    """生成 ``n`` 根合成日线，列结构与真实行情完全一致。"""
    rng = np.random.default_rng(seed)

    # 用几段不同斜率的趋势 + 噪声拼出一条"像样"的走势
    drift = np.concatenate([
        np.linspace(0.0012, 0.0022, n // 5),
        np.linspace(-0.0018, -0.0026, n // 5),
        np.linspace(0.0004, 0.0009, n // 5),
        np.linspace(0.0026, 0.0010, n // 5),
        np.linspace(-0.0008, 0.0016, n - 4 * (n // 5)),
    ])
    noise = rng.normal(0.0, 0.011, n)
    ret = drift + noise

    close = 3000.0 * np.exp(np.cumsum(ret))
    open_ = np.empty_like(close)
    open_[0] = close[0] * (1 - rng.normal(0, 0.003))
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.002, n - 1))

    span = np.abs(rng.normal(0, 0.006, n)) + 0.001
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    volume = rng.lognormal(mean=17.0, sigma=0.35, size=n)
    amount = volume * (high + low) / 2.0

    dates = pd.bdate_range(start=start, periods=n)
    df = pd.DataFrame({
        "date": dates,
        "open": open_.round(2),
        "high": high.round(2),
        "low": low.round(2),
        "close": close.round(2),
        "volume": volume.round(0),
        "amount": amount.round(0),
    })
    return df[COLUMNS]
