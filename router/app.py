# -*- coding: utf-8 -*-
"""
LLM Router 桌面版启动器

- 后台启动网关服务（router.py）
- 用原生桌面窗口显示控制台（pywebview）
- 关闭窗口 = 隐藏到系统托盘（右下角小图标），托盘菜单可显示主界面 / 退出软件
- 如果原生窗口打不开，自动退回浏览器模式
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

# 系统托盘（pystray）；不可用时退回“关闭即最小化到任务栏”
try:
    import pystray
    from PIL import Image as _PILImage
except Exception:
    pystray = None

import paths as _paths

BASE_DIR = _paths.BASE_DIR        # 程序所在目录
DATA_DIR = _paths.DATA_DIR        # 用户数据目录（macOS 在 ~/Library/Application Support）
CONFIG_PATH = _paths.CONFIG_PATH
CRASH_LOG_PATH = _paths.CRASH_LOG_PATH

IS_MACOS = _paths.IS_MACOS

_tray_icon = None    # pystray.Icon 实例
_tray_ready = False  # 托盘是否启动成功


class AppApi:
    def __init__(self):
        self._window = None
        self._allow_close = False

    def bind(self, window):
        self._window = window

    def exit_app(self):
        self._allow_close = True
        stop_tray()
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                os._exit(0)


def _load_tray_image():
    """加载托盘图标；失败时生成占位图。"""
    # macOS 打包后图标在 Contents/Resources，Windows 在 exe 同目录，统一用 resource_path 找
    candidates = [_paths.resource_path("icon.icns"), _paths.resource_path("icon.ico")]
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", ""))
        if meipass.exists():
            candidates.append(meipass / "icon.icns")
            candidates.append(meipass / "icon.ico")
    candidates.append(BASE_DIR / "icon.icns")
    candidates.append(BASE_DIR / "icon.ico")
    for path in candidates:
        try:
            if path.exists():
                return _PILImage.open(path)
        except Exception:
            pass
    img = _PILImage.new("RGBA", (64, 64), (13, 15, 26, 255))
    return img


def start_tray(window, show_cb, quit_cb):
    """启动系统托盘图标；成功返回 True。"""
    global _tray_icon, _tray_ready
    if pystray is None:
        return False
    try:
        menu = pystray.Menu(
            pystray.MenuItem("显示主界面", lambda icon, item: show_cb(), default=True),
            pystray.MenuItem("退出软件", lambda icon, item: quit_cb()),
        )
        kwargs = {}
        if IS_MACOS:
            # macOS 的托盘图标是状态栏图标，必须绑定到已存在的 NSApplication，
            # 并且要在主线程、NSApp 跑起来之后再创建。
            import AppKit

            kwargs["darwin_nsapplication"] = AppKit.NSApplication.sharedApplication()
        _tray_icon = pystray.Icon(
            "llm-router",
            _load_tray_image(),
            "LLM Router 控制台",
            menu,
            **kwargs,
        )
        _tray_icon.run_detached()
        _tray_icon.update_menu()  # 预创建原生菜单，避免首次右键无响应
        _tray_ready = True
        return True
    except Exception:
        _tray_icon = None
        _tray_ready = False
        return False


def stop_tray():
    """停止托盘图标（退出进程前调用）。"""
    global _tray_icon, _tray_ready
    if _tray_icon is not None:
        try:
            _tray_icon.stop()
        except Exception:
            pass
        _tray_icon = None
    _tray_ready = False


if sys.stdout is None or sys.stderr is None:
    # PyInstaller --windowed 下没有控制台，stdout/stderr 为 None，uvicorn 等库会崩溃，这里兜底
    import os
    _devnull = open(os.devnull, "w")
    sys.stdout = _devnull
    sys.stderr = _devnull


def _show_error(title: str, text: str) -> None:
    """EXE 用 --windowed 打包、没有控制台，print 出来的话用户只会看到闪退。"""
    try:
        if IS_MACOS:
            # macOS 没有 MessageBoxW，用 osascript 弹系统原生对话框（不依赖 GUI 库）。
            import subprocess

            def _q(value: str) -> str:
                return value.replace("\\", "\\\\").replace('"', '\\"')

            script = (
                f'display alert "{_q(title)}" message "{_q(text)}" '
                'as critical buttons {"好"} default button "好"'
            )
            subprocess.run(
                ["osascript", "-e", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)  # MB_ICONERROR
    except Exception:
        pass


_config_error = None


def load_config() -> dict:
    global _config_error
    try:
        # utf-8-sig：兼容记事本另存为 UTF-8 时写入的 BOM。
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError:
        _config_error = None
        return {}
    except Exception as exc:
        # 配置坏了不能当成空配置启动，否则界面看起来“供应商全没了”，
        # 一保存就会把原文件覆盖掉。
        _config_error = f"{type(exc).__name__}: {exc}"
        return {}


def load_port(cfg: dict) -> int:
    try:
        return int(cfg.get("port") or 8765)
    except Exception:
        return 8765


def start_server(port: int):
    """在后台线程启动 uvicorn，返回 (server, thread)；启动失败返回 (None, None)。"""
    import uvicorn

    import router

    config = uvicorn.Config(router.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="llm-router-server")
    thread.start()
    for _ in range(50):
        if server.started:
            router.set_server(server, thread, port)
            return server, thread
        time.sleep(0.1)
    if thread.is_alive():
        server.should_exit = True
    return None, None


def _headless_requested() -> bool:
    """CI / 无图形环境用：只跑网关，不开窗口和托盘。"""
    return os.environ.get("LLMROUTER_HEADLESS", "").strip().lower() in ("1", "true", "yes")


def _selftest_requested() -> bool:
    """打包自检用：只验证依赖能不能 import，不启动网关。"""
    return os.environ.get("LLMROUTER_SELFTEST", "").strip().lower() in ("1", "true", "yes")


# 打包后最容易缺的就是这些：PyInstaller 没有 pyobjc / pystray 的现成 hook，
# 全靠 LLMRouter.spec 里的 hiddenimports 手动声明，写漏了要到用户机器上才炸。
_SELFTEST_COMMON = (
    "webview",
    "pystray",
    "PIL.Image",
)

_SELFTEST_MACOS = (
    "webview.platforms.cocoa",
    "AppKit",
    "Foundation",
    "WebKit",
    "objc",
    "PyObjCTools.AppHelper",
    "Quartz",
    "UniformTypeIdentifiers",
)

_SELFTEST_WINDOWS = (
    "webview.platforms.winforms",
    "clr",
)


def _selftest_modules():
    if IS_MACOS:
        return _SELFTEST_COMMON + _SELFTEST_MACOS
    if _paths.IS_WINDOWS:
        return _SELFTEST_COMMON + _SELFTEST_WINDOWS
    return _SELFTEST_COMMON


def _run_selftest() -> int:
    """逐个 import 关键依赖；缺哪个就报哪个。返回进程退出码。"""
    import importlib

    modules = _selftest_modules()
    failed = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(name)
            print(f"SELFTEST FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"SELFTEST OK {name}")

    print(f"SELFTEST RESULT {len(modules) - len(failed)}/{len(modules)}")
    if failed:
        print("SELFTEST FAILED")
        return 1
    print("SELFTEST PASSED")
    return 0


def main():
    import router

    if _selftest_requested():
        sys.exit(_run_selftest())

    router.migrate_config_on_startup()
    cfg = load_config()
    if _config_error:
        _show_error(
            "LLM Router 无法读取配置",
            f"配置文件解析失败：\n{CONFIG_PATH}\n\n{_config_error}\n\n"
            "请修正该文件（或先改名备份，程序会重新生成默认配置）后再启动。",
        )
        sys.exit(1)
    port = load_port(cfg)

    import stats

    stats.load()

    server, thread = start_server(port)
    if server is None:
        _show_error(
            "LLM Router 启动失败",
            f"端口 {port} 可能已被占用。\n\n"
            f"请关闭占用该端口的程序，或修改 {CONFIG_PATH} 里的 port 后重试。",
        )
        print(f"启动失败：端口 {port} 可能已被占用。请关闭占用该端口的程序，或改端口后重试。")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/"
    print(f"LLM Router 控制台：{url}")
    print("关闭窗口将隐藏到系统托盘（右下角小图标）；点托盘“显示主界面”恢复，托盘菜单“退出软件”可退出。")

    if _headless_requested():
        print("无界面模式：网关已启动，按 Ctrl+C 退出。")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        server.should_exit = True
        return

    if cfg.get("open_browser"):
        # 浏览器模式（设置里勾选了"浏览器模式"时）
        import webbrowser

        webbrowser.open(url)
        print("浏览器模式：关闭软件请按 Ctrl+C。")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    else:
        # 桌面窗口模式
        try:
            import webview

            window_api = AppApi()
            window = webview.create_window(
                "LLM Router 控制台",
                url,
                width=1500,
                height=900,
                min_size=(1100, 700),
                background_color="#0d0f1a",
                js_api=window_api,
            )
            window_api.bind(window)

            def _on_closing():
                if window_api._allow_close:
                    return True
                if _tray_ready:
                    window.hide()
                else:
                    window.minimize()
                return False

            window.events.closing += _on_closing

            def _show_window():
                try:
                    window.show()
                    window.restore()
                except Exception:
                    pass

            def _quit_from_tray():
                window_api._allow_close = True
                stop_tray()
                try:
                    window.destroy()
                except Exception:
                    os._exit(0)

            # 托盘必须在进入 GUI 主循环之前建好：pystray 的 run_detached()
            # 要求先调用它、再由别的库（pywebview）接管主循环。
            # macOS 上 pywebview 自己会把 NSApplication 设成 Regular 策略
            # （有 Dock 图标、有应用菜单、⌘Q 可用），这里不用再改。
            start_tray(window, _show_window, _quit_from_tray)
            webview.start()
        except Exception as exc:
            print(f"原生窗口打开失败（{exc}），改用浏览器打开。")
            import webbrowser

            webbrowser.open(url)
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass
    server.should_exit = True


def _write_crash_log():
    """把异常堆栈写入 crash.log，方便双击闪退时排查。"""
    try:
        import traceback
        with open(CRASH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            traceback.print_exc(file=f)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        _write_crash_log()
        print(f"程序异常退出，详情已写入 {CRASH_LOG_PATH}")
        raise
