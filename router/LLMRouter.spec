# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（Windows / macOS 共用）。

- Windows：单文件 LLMRouter.exe，图标资源一起塞进 exe。
- macOS：onedir + BUNDLE，产出 dist/LLMRouter.app（内含 Contents/MacOS/LLMRouter）。
  图标用仓库里预先生成好的 icon.icns。
"""
import sys

from PyInstaller.utils.hooks import collect_all

datas = [('dashboard.html', '.')]
binaries = []
hiddenimports = []

# Windows 用 icon.ico 当 exe 图标；macOS 用预先生成好的 icon.icns（BUNDLE 的 icon 参数）
datas.append(('icon.ico', '.'))
if sys.platform == 'darwin':
    # 托盘图标走 resource_path('icon.icns')，打包后要能在 Contents/Resources 里找到
    datas.append(('icon.icns', '.'))

tmp_ret = collect_all('webview')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

if sys.platform == 'darwin':
    # pywebview 的 Cocoa 后端和 pystray 的 macOS 后端都是运行时才 import，
    # PyInstaller 没有现成的 hook 去收 PyObjC 的框架，只能显式声明。
    hiddenimports += [
        'AppKit',
        'Foundation',
        'WebKit',
        'UniformTypeIdentifiers',
        'objc',
        'PyObjCTools',
        'PyObjCTools.AppHelper',
        'PyObjCTools.MachSignals',
        'Quartz',
        'CoreFoundation',
        'Security',
    ]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='LLMRouter',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='LLMRouter',
    )
    app = BUNDLE(
        coll,
        name='LLMRouter.app',
        # 注意：BUNDLE 的 icon 必须是字符串路径（EXE 的 icon 才是列表）。
        # BUNDLE 内部会 os.path.isabs(icon)，传列表直接 TypeError，整个打包失败。
        # 用预生成的 icon.icns，不走 BUNDLE 的 ico->icns 现场转换（那条路会
        # 依赖 Pillow 的 ICNS 编码器，出问题时报错很隐晦）。
        icon='icon.icns',
        bundle_identifier='com.xcchc.llmrouter',
        version='1.0.1',
        info_plist={
            'CFBundleDisplayName': 'LLM Router',
            'NSHighResolutionCapable': True,
            'LSApplicationCategoryType': 'public.app-category.developer-tools',
            'NSPrincipalClass': 'NSApplication',
            # 只有菜单栏图标 + 普通窗口，不需要沙盒/权限
            'LSMinimumSystemVersion': '11.0',
            # 界面从 http://127.0.0.1:8765 加载，WKWebView 的 ATS 会拦明文 HTTP。
            # 只写 NSAllowsArbitraryLoads 一个键：一旦同时出现
            # NSAllowsLocalNetworking / NSAllowsArbitraryLoadsInWebContent，
            # 这个键会被系统忽略（macOS 10.12+ 的规则），反而更不可控。
            'NSAppTransportSecurity': {'NSAllowsArbitraryLoads': True},
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='LLMRouter',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=['icon.ico'],
    )
