"""本地行情缓存。

以 CSV 形式落在 ``data/_cache/`` 目录下，文件名由 cache key 的 md5 生成，
避免中文/特殊字符导致的路径问题。

设计原则：缓存失败永远不能让主流程崩溃，所有 IO 异常都被吞掉并降级为
"没有缓存"。
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Optional

import pandas as pd


def _cache_base() -> str:
    """缓存根目录。

    * 源码运行：放在 ``data/`` 旁边，开发期一眼能看到缓存文件。
    * PyInstaller 冻结后：``__file__`` 落在只读的安装目录里（装到
      ``C:\\Program Files`` 下时连 ``makedirs`` 都会抛 ``PermissionError``），
      因此改落到用户主目录 ``~/.stockchan``。
    """
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.expanduser("~"), ".stockchan")
    return os.path.dirname(os.path.abspath(__file__))


#: 缓存目录（绝对路径）
CACHE_DIR = os.path.join(_cache_base(), "_cache")

#: 默认缓存有效期（小时）。日线数据当天内不重复拉取。
DEFAULT_MAX_AGE_HOURS = 6.0


def _ensure_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _path_for(key: str) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{digest}.csv")


def load_cache(key: str, max_age_hours: Optional[float] = DEFAULT_MAX_AGE_HOURS) -> Optional[pd.DataFrame]:
    """读取缓存；不存在、过期或损坏时返回 ``None``。"""
    path = _path_for(key)
    if not os.path.exists(path):
        return None

    if max_age_hours is not None:
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_hours > max_age_hours:
            return None

    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if df.empty:
            return None
        df.attrs["from_cache"] = True
        df.attrs["cache_key"] = key
        return df
    except Exception:
        return None


def save_cache(key: str, df: pd.DataFrame) -> None:
    """写入缓存，失败静默忽略。"""
    if df is None or df.empty:
        return
    try:
        _ensure_dir()
        df.to_csv(_path_for(key), index=False, encoding="utf-8-sig")
    except Exception:
        pass


def clear_cache() -> int:
    """清空缓存目录，返回删除的文件数量。"""
    if not os.path.isdir(CACHE_DIR):
        return 0
    removed = 0
    for name in os.listdir(CACHE_DIR):
        if not name.endswith(".csv"):
            continue
        try:
            os.remove(os.path.join(CACHE_DIR, name))
            removed += 1
        except Exception:
            pass
    return removed
