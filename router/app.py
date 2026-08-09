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

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent  # exe 所在目录（crash.log 放这里）
else:
    BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

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
    candidates = []
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", ""))
        if meipass.exists():
            candidates.append(meipass / "icon.ico")
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
        _tray_icon = pystray.Icon(
            "llm-router",
            _load_tray_image(),
            "LLM Router 控制台",
            menu,
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


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
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


def main():
    import router

    cfg = load_config()
    port = load_port(cfg)

    import stats

    stats.load()

    server, thread = start_server(port)
    if server is None:
        print(f"启动失败：端口 {port} 可能已被占用。请关闭占用该端口的程序，或改端口后重试。")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/"
    print(f"LLM Router 控制台：{url}")
    print("关闭窗口将隐藏到系统托盘（右下角小图标）；点托盘“显示主界面”恢复，托盘菜单“退出软件”可退出。")

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
        with open(BASE_DIR / "crash.log", "a", encoding="utf-8") as f:
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
        print("程序异常退出，详情已写入 crash.log")
        raise
