# -*- coding: utf-8 -*-
import hashlib
import os
import pathlib
import subprocess
import tempfile
import unittest

import router


class UpdateStatusTests(unittest.TestCase):
    def test_missing_new_exe(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base / "LLMRouter.exe").write_bytes(b"old")
            status = router._update_file_status(base)
            self.assertTrue(status["current_exe"]["exists"])
            self.assertFalse(status["new_exe"]["exists"])

    def test_new_exe_is_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base / "LLMRouter.exe").write_bytes(b"old")
            (base / "LLMRouter.new.exe").write_bytes(b"new")
            status = router._update_file_status(base)
            self.assertTrue(status["new_exe"]["exists"])
            self.assertEqual(
                status["new_exe"]["sha256"],
                hashlib.sha256(b"new").hexdigest(),
            )

    def test_build_script_embeds_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            script = router._build_update_script(base)
            self.assertIn(str(base.resolve()), script)
            self.assertIn("LLMRouter.new.exe", script)
            self.assertIn("PYINSTALLER_RESET_ENVIRONMENT = '1'", script)
            self.assertIn("Name -like '_PYI_*'", script)
            self.assertIn("Start-Router", script)


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
            (base / "LLMRouter.exe").write_bytes(b"old-binary")
            (base / "LLMRouter.new.exe").write_bytes(b"new-binary")
            (base / "crash.log").write_text("old crash", encoding="utf-8")
            result = _run_update_script(base, wait_for_process=False, launch_new=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((base / "LLMRouter.exe").read_bytes(), b"new-binary")
            self.assertFalse((base / "LLMRouter.new.exe").exists())
            self.assertFalse((base / "LLMRouter.previous.exe").exists())
            self.assertFalse((base / "crash.log").exists())
            self.assertFalse((base / "update-error.log").exists())

    def test_script_restores_old_exe_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base / "LLMRouter.exe").write_bytes(b"old-binary")
            (base / "LLMRouter.new.exe").write_bytes(b"new-binary")
            result = _run_update_script(
                base,
                wait_for_process=False,
                launch_new=False,
                force_fail=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((base / "LLMRouter.exe").read_bytes(), b"old-binary")
            self.assertTrue((base / "update-error.log").exists())
            self.assertFalse((base / "LLMRouter.previous.exe").exists())


if __name__ == "__main__":
    unittest.main()
