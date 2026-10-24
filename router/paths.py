# -*- coding: utf-8 -*-
"""跨平台路径常量（Windows / macOS）。

三种运行形态下的文件位置：

- 源码运行：所有文件都在仓库的 router/ 目录里，方便调试、跑测试。
- Windows 打包版：config.json / stats.jsonl / 日志放在 exe 同目录（保持原有行为，
  绿色免安装，整个文件夹可以随便挪）。
- macOS 打包版：.app 通常被拖到 /Applications，那里普通用户不可写，而且往 .app
  里写文件会让代码签名失效，所以用户数据统一放到
  ~/Library/Application Support/LLMRouter。
"""
import os
import sys
from pathlib import Path

APP_NAME = "LLMRouter"
APP_VERSION = "2026.10.24.1"

FROZEN = bool(getattr(sys, "frozen", False))
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = os.name == "nt"

# 程序自身所在目录：源码模式 = router/ 目录；打包后 = 可执行文件所在目录
BASE_DIR = (
    Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
)

# 可执行文件名（macOS 的 .app 里没有 .exe 后缀）
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""
EXE_NAME = APP_NAME + EXE_SUFFIX
NEW_EXE_NAME = APP_NAME + ".new" + EXE_SUFFIX
PREV_EXE_NAME = APP_NAME + ".previous" + EXE_SUFFIX

# 只有 Windows 版支持「下载新版 exe 后自动替换并重启」；macOS 走 .zip 覆盖安装
SELF_UPDATE_SUPPORTED = IS_WINDOWS


def _resource_dirs() -> list:
    """打包资源的候选目录。

    - Windows onefile/onedir：资源就在 sys._MEIPASS 里。
    - macOS .app（onedir）：数据文件被放进 Contents/Resources，再软链到
      Contents/Frameworks，而 sys._MEIPASS 指向 Contents/MacOS，所以要额外找
      上一层 Resources。
    """
    dirs = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(meipass)
        dirs.append(base)
        dirs.append(base.parent / "Resources")
    dirs.append(BASE_DIR)
    return dirs


def resource_path(name: str) -> Path:
    """返回打包资源（dashboard.html、图标）的真实路径；找不到时返回首选路径。"""
    dirs = _resource_dirs()
    for directory in dirs:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return dirs[0] / name


def _default_data_dir() -> Path:
    if FROZEN and IS_MACOS:
        try:
            return Path.home() / "Library" / "Application Support" / APP_NAME
        except Exception:
            return BASE_DIR
    return BASE_DIR


def _ensure_dir(path: Path) -> Path:
    """确保可写目录存在；失败就退回程序目录，至少还能启动。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return BASE_DIR


DATA_DIR = _ensure_dir(_default_data_dir())

CONFIG_PATH = DATA_DIR / "config.json"
STATS_FILE = DATA_DIR / "stats.jsonl"
ERROR_LOG_PATH = DATA_DIR / "router-error.log"
CRASH_LOG_PATH = DATA_DIR / "crash.log"
DASHBOARD_PATH = resource_path("dashboard.html")
