# -*- coding: utf-8 -*-
"""打包配置（LLMRouter.spec）的静态检查。

这些断言不打包，只检查 spec 里的关键参数有没有写错——尤其是那些在
Windows 上根本走不到、只有 macOS CI 才会炸的分支。
"""
import ast
import pathlib
import struct
import unittest

ROUTER_DIR = pathlib.Path(__file__).resolve().parent.parent
SPEC_PATH = ROUTER_DIR / "LLMRouter.spec"


def _spec_calls():
    """解析 spec，返回 {函数名: [关键字参数字典, ...]}。"""
    tree = ast.parse(SPEC_PATH.read_text(encoding="utf-8"))
    calls = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if not name:
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        calls.setdefault(name, []).append(kwargs)
    return calls


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


class SpecTests(unittest.TestCase):
    def setUp(self):
        if not SPEC_PATH.exists():
            self.skipTest("找不到 LLMRouter.spec")
        self.calls = _spec_calls()

    def test_bundle_icon_is_a_path_not_a_list(self):
        # BUNDLE 内部会 os.path.isabs(icon)，传列表直接 TypeError，
        # 而这个分支只在 macOS 上执行，Windows 本地测试发现不了。
        bundles = self.calls.get("BUNDLE", [])
        self.assertTrue(bundles, "spec 里没有 BUNDLE(...)")
        for kwargs in bundles:
            icon = _literal(kwargs.get("icon"))
            self.assertIsInstance(icon, str, "BUNDLE 的 icon 必须是字符串路径")

    def test_bundle_icon_file_exists(self):
        for kwargs in self.calls.get("BUNDLE", []):
            icon = _literal(kwargs.get("icon"))
            if not icon:
                continue
            self.assertTrue(
                (ROUTER_DIR / icon).is_file(), f"图标文件不存在：{icon}"
            )

    def test_bundle_icon_is_icns(self):
        for kwargs in self.calls.get("BUNDLE", []):
            icon = _literal(kwargs.get("icon"))
            if icon:
                self.assertTrue(icon.endswith(".icns"), f"BUNDLE 只接受 .icns：{icon}")

    def test_icon_icns_is_shipped_as_data(self):
        # 托盘图标运行时通过 resource_path('icon.icns') 找，必须打进包里。
        # datas 在 spec 里是边构建边 append 的，不是字面量，所以直接查源码。
        source = SPEC_PATH.read_text(encoding="utf-8")
        for entry in ("('dashboard.html', '.')", "('icon.ico', '.')", "('icon.icns', '.')"):
            self.assertIn(entry, source, f"spec 的 datas 缺少 {entry}")

    def test_bundle_identifier_is_reverse_dns(self):
        for kwargs in self.calls.get("BUNDLE", []):
            identifier = _literal(kwargs.get("bundle_identifier"))
            self.assertTrue(identifier)
            parts = identifier.split(".")
            self.assertGreaterEqual(len(parts), 2, identifier)
            self.assertNotIn("-", identifier)

    def test_ats_allows_local_http(self):
        # 界面走 http://127.0.0.1，ATS 不放行的话窗口会白屏
        for kwargs in self.calls.get("BUNDLE", []):
            plist = _literal(kwargs.get("info_plist")) or {}
            ats = plist.get("NSAppTransportSecurity") or {}
            self.assertTrue(ats.get("NSAllowsArbitraryLoads"), "缺少 ATS 例外")
            # 同时出现这几个键会让 NSAllowsArbitraryLoads 被忽略
            for conflicting in ("NSAllowsLocalNetworking", "NSAllowsArbitraryLoadsInWebContent"):
                self.assertNotIn(conflicting, ats)


class IconIcnsTests(unittest.TestCase):
    """icon.icns 必须是 macOS 认得的格式，否则 .app 图标是空白。"""

    def setUp(self):
        self.path = ROUTER_DIR / "icon.icns"
        if not self.path.exists():
            self.skipTest("找不到 icon.icns")

    def test_header_and_length(self):
        data = self.path.read_bytes()
        self.assertEqual(data[:4], b"icns")
        declared = struct.unpack(">I", data[4:8])[0]
        self.assertEqual(declared, len(data), "icns 声明的长度和实际文件大小不一致")

    def test_members_are_walkable(self):
        data = self.path.read_bytes()
        offset = 8
        seen = []
        while offset < len(data):
            self.assertGreaterEqual(len(data) - offset, 8, "成员头被截断")
            type_code = data[offset : offset + 4]
            length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
            self.assertGreaterEqual(length, 8, f"成员 {type_code!r} 长度非法")
            self.assertLessEqual(offset + length, len(data), f"成员 {type_code!r} 越界")
            seen.append(type_code.decode("ascii", "replace"))
            offset += length
        self.assertEqual(offset, len(data))
        self.assertIn("TOC ", seen)  # icns 的 TOC 类型码带一个尾随空格
        # 至少要有一个 512/1024 的大图，Dock 和 Finder 才不会糊
        self.assertTrue(
            {"ic10", "ic09", "ic14", "ic13"} & set(seen), f"缺少大尺寸图标：{seen}"
        )


if __name__ == "__main__":
    unittest.main()
