# -*- coding: utf-8 -*-
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import router


class ConfigLoadingTests(unittest.TestCase):
    def test_missing_config_creates_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "config.json"
            with patch.object(router, "CONFIG_PATH", path):
                cfg = router.load_config()
            self.assertTrue(path.exists())
            self.assertEqual(cfg["port"], 8765)
            self.assertEqual(cfg["suppliers"], [])
            # 新建的配置不能带 BOM，否则别的工具解析会出错。
            self.assertFalse(path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_config_with_utf8_bom_is_readable(self):
        # 记事本「另存为 UTF-8」会写入 BOM；旧版直接 utf-8 解码会抛错并导致双击闪退。
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "config.json"
            payload = {"port": 8899, "suppliers": [{"name": "demo"}]}
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))
            with patch.object(router, "CONFIG_PATH", path):
                cfg = router.load_config()
            self.assertEqual(cfg["port"], 8899)
            self.assertEqual(cfg["suppliers"], [{"name": "demo"}])


if __name__ == "__main__":
    unittest.main()
