# -*- coding: utf-8 -*-
import hashlib
import os
import pathlib
import subprocess
import tempfile
import unittest

import paths
import router

# 文件名随平台变化（Windows 有 .exe 后缀，macOS 没有），测试里统一从这里取
EXE_NAME = paths.EXE_NAME
NEW_EXE_NAME = paths.NEW_EXE_NAME
PREV_EXE_NAME = paths.PREV_EXE_NAME


class UpdateStatusTests(unittest.TestCase):
    def test_missing_new_exe(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base / EXE_NAME).write_bytes(b"old")
            status = router._update_file_status(base)
            self.assertTrue(status["current_exe"]["exists"])
            self.assertFalse(status["new_exe"]["exists"])

    def test_new_exe_is_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base / EXE_NAME).write_bytes(b"old")
            (base / NEW_EXE_NAME).write_bytes(b"new")
            status = router._update_file_status(base)
            self.assertTrue(status["new_exe"]["exists"])
            self.assertEqual(
                status["new_exe"]["sha256"],
                hashlib.sha256(b"new").hexdigest(),
            )

    def test_status_reports_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            status = router._update_file_status(pathlib.Path(directory))
            self.assertIn(status["platform"], ("windows", "macos"))
            self.assertEqual(status["self_update"], paths.SELF_UPDATE_SUPPORTED)
            self.assertEqual(status["exe_name"], EXE_NAME)
            self.assertEqual(status["new_exe_name"], NEW_EXE_NAME)

    def test_build_script_embeds_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            script = router._build_update_script(base)
            self.assertIn(str(base.resolve()), script)
            self.assertIn(NEW_EXE_NAME, script)
            self.assertIn("PYINSTALLER_RESET_ENVIRONMENT = '1'", script)
            self.assertIn("Name -like '_PYI_*'", script)
            self.assertIn("Start-Router", script)

    def test_build_script_waits_for_own_process_name(self):
        # 等待旧进程退出时用的进程名必须是 APP_NAME，写错会等错进程或直接不等
        with tempfile.TemporaryDirectory() as directory:
            script = router._build_update_script(pathlib.Path(directory))
            self.assertIn(f"Get-Process -Name '{paths.APP_NAME}'", script)

    def test_build_script_without_wait_has_no_process_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            script = router._build_update_script(
                pathlib.Path(directory), wait_for_process=False
            )
            self.assertNotIn("Get-Process -Name", script)

    def test_apply_rejected_when_self_update_unsupported(self):
        # macOS 版走 zip 覆盖安装，在线替换接口必须明确报错而不是静默失败。
        # 这里把开关临时改成 False，好让 Windows 上也能验证这段逻辑。
        from unittest import mock

        from fastapi.testclient import TestClient

        with mock.patch.object(router._paths, "SELF_UPDATE_SUPPORTED", False):
            client = TestClient(router.app)
            response = client.post("/api/update/apply")
        self.assertEqual(response.status_code, 400)
        self.assertIn("LLMRouter.app", response.json()["error"])

    def test_self_update_supported_matches_platform(self):
        self.assertEqual(paths.SELF_UPDATE_SUPPORTED, paths.IS_WINDOWS)
        self.assertEqual(paths.EXE_SUFFIX, ".exe" if paths.IS_WINDOWS else "")


def _run_update_script(base: pathlib.Path, **kwargs):
    script_path = base / "update-test.ps1"
    script_path.write_text(router._build_update_script(base, **kwargs), encoding="utf-8-sig")
    env = dict(os.environ)
    env["LLMROUTER_FORCE_UPDATE_FAIL"] = "1" if kwargs.get("force_fail") else "0"
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
        ],
        cwd=str(base),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


@unittest.skipUnless(os.name == "nt", "PowerShell update script is Windows-only")
class UpdateScriptExecutionTests(unittest.TestCase):
    def test_script_replaces_exe_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base / EXE_NAME).write_bytes(b"old-binary")
            (base / NEW_EXE_NAME).write_bytes(b"new-binary")
            (base / "crash.log").write_text("old crash", encoding="utf-8")
            result = _run_update_script(base, wait_for_process=False, launch_new=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((base / EXE_NAME).read_bytes(), b"new-binary")
            self.assertFalse((base / NEW_EXE_NAME).exists())
            self.assertFalse((base / PREV_EXE_NAME).exists())
            self.assertFalse((base / "crash.log").exists())
            self.assertFalse((base / "update-error.log").exists())

    def test_script_restores_old_exe_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base / EXE_NAME).write_bytes(b"old-binary")
            (base / NEW_EXE_NAME).write_bytes(b"new-binary")
            result = _run_update_script(
                base,
                wait_for_process=False,
                launch_new=False,
                force_fail=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((base / EXE_NAME).read_bytes(), b"old-binary")
            self.assertTrue((base / "update-error.log").exists())
            self.assertFalse((base / PREV_EXE_NAME).exists())


if __name__ == "__main__":
    unittest.main()
