# -*- coding: utf-8 -*-
import datetime
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import stats


class StatsFilteringTests(unittest.TestCase):
    def setUp(self):
        self.original_records = stats._records
        stats._records = [
            {
                "ts": 100.0,
                "model": "older-model",
                "supplier": "legacy",
                "status": "ok",
                "duration_ms": 1000,
            },
            {
                "ts": 990.0,
                "model": "current-model",
                "supplier": "current",
                "status": "ok",
                "duration_ms": 1250,
                "reasoning_effort": "high",
            },
            {
                "ts": 995.0,
                "model": "other-model",
                "supplier": "current",
                "status": "ok",
                "duration_ms": 2750,
                "reasoning_effort": "medium",
            },
        ]

    def tearDown(self):
        stats._records = self.original_records

    @patch("stats.time_now", return_value=1000.0)
    def test_summary_lists_all_historical_models_and_reports_seconds(self, _mock_now):
        result = stats.summary({}, model="current-model", hours=1)

        self.assertEqual(
            result["all_models"],
            ["current-model", "older-model", "other-model"],
        )
        self.assertEqual(result["total"]["requests"], 1)
        self.assertEqual(result["avg_duration_seconds"], 1.25)

    @patch("stats.time_now", return_value=1000.0)
    def test_recent_filtered_applies_model_and_time_filters(self, _mock_now):
        total, records = stats.recent_filtered(
            limit=50,
            offset=0,
            hours=1,
            model="current-model",
        )

        self.assertEqual(total, 1)
        self.assertEqual(records[0]["model"], "current-model")
        self.assertEqual(records[0]["reasoning_effort"], "high")

    @patch("stats.time_now", return_value=1000.0)
    def test_recent_filtered_splits_ok_and_error_requests(self, _mock_now):
        stats._records = [
            {"ts": 999.0, "model": "m", "supplier": "s", "status": "ok"},
            {"ts": 998.0, "model": "m", "supplier": "s", "status": "error"},
            {"ts": 997.0, "model": "m", "supplier": "s", "status": "failed"},
        ]

        total_ok, ok_rows = stats.recent_filtered(hours=1, status="ok")
        total_error, error_rows = stats.recent_filtered(hours=1, status="error")
        total_all, _ = stats.recent_filtered(hours=1)

        self.assertEqual((total_ok, total_error, total_all), (1, 2, 3))
        self.assertEqual([r["status"] for r in ok_rows], ["ok"])
        self.assertEqual([r["status"] for r in error_rows], ["error", "failed"])

    def test_rolling_hour_chart_sum_matches_card_across_clock_boundary(self):
        now = datetime.datetime(2026, 8, 5, 12, 30, 0).timestamp()
        stats._records = [
            {"ts": now - 75 * 60, "model": "m", "supplier": "s", "status": "ok"},
            {"ts": now - 45 * 60, "model": "m", "supplier": "s", "status": "ok"},
            {"ts": now - 15 * 60, "model": "m", "supplier": "s", "status": "ok"},
        ]

        with patch("stats.time_now", return_value=now):
            result = stats.summary({}, hours=1)

        self.assertEqual(result["total"]["requests"], 2)
        self.assertEqual(sum(bucket["requests"] for bucket in result["hourly"]), 2)
        self.assertEqual([bucket["hour"] for bucket in result["hourly"]], ["11:00", "12:00"])


class StatsModelIdTests(unittest.TestCase):
    def setUp(self):
        self.original_records = stats._records
        self.addCleanup(setattr, stats, "_records", self.original_records)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "stats.jsonl"

    def test_load_qualifies_historical_bare_model_names(self):
        self.path.write_text(
            "\n".join([
                json.dumps({"ts": 1, "model": "deepseek-v4.1-flash", "supplier": "pixel", "status": "ok"}),
                json.dumps({"ts": 2, "model": "xkiro/z-ai/glm-5.3", "supplier": "xkiro", "status": "ok"}),
                json.dumps({"ts": 3, "model": "?", "status": "ok"}),
            ]) + "\n",
            encoding="utf-8",
        )

        with patch.object(stats, "STATS_FILE", self.path):
            stats.load()

        self.assertEqual(
            [row["model"] for row in stats._records],
            ["pixel/deepseek-v4.1-flash", "xkiro/z-ai/glm-5.3", "?"],
        )

    def test_summary_has_no_reasoning_bucket(self):
        stats._records = [
            {"ts": 999.0, "model": "pixel/m", "supplier": "pixel", "status": "ok", "reasoning_tokens": 100},
        ]

        with patch("stats.time_now", return_value=1000.0):
            result = stats.summary({}, hours=1)

        self.assertNotIn("reasoning", result["total"])
        self.assertNotIn("reasoning", result["by_model"][0])
        self.assertNotIn("reasoning", result["by_supplier"][0])


class StatsSpeedTests(unittest.TestCase):
    """token 速度 = 总输出 token ÷ 生成耗时（总耗时减首字），不是把每次请求的速度再平均。"""

    def setUp(self):
        self.original_records = stats._records
        self.addCleanup(setattr, stats, "_records", self.original_records)
        self.now = time.time()
        stats._records = [
            # 1 秒出 100 token = 100 t/s
            {"ts": self.now - 10, "model": "pixel/a", "supplier": "pixel", "status": "ok", "output_tokens": 100, "duration_ms": 1000},
            # 9 秒出 100 token = 11.1 t/s
            {"ts": self.now - 8, "model": "pixel/a", "supplier": "pixel", "status": "ok", "output_tokens": 100, "duration_ms": 9000},
            # 2 秒出 400 token = 200 t/s
            {"ts": self.now - 6, "model": "xkiro/b", "supplier": "xkiro", "status": "ok", "output_tokens": 400, "duration_ms": 2000},
            # 失败请求没有耗时，不能拖坏分母
            {"ts": self.now - 4, "model": "pixel/a", "supplier": "pixel", "status": "error", "output_tokens": 0, "duration_ms": None},
        ]

    @patch("stats.time_now")
    def test_total_speed_uses_tokens_over_time(self, mock_now):
        mock_now.return_value = self.now

        result = stats.summary({}, hours=1)

        # (100+100+400) token / (1+9+2) 秒 = 50 t/s；直接平均每次速度会得到约 104 t/s。
        self.assertEqual(result["total"]["speed"], 50.0)
        self.assertEqual(result["today"]["speed"], 50.0)

    @patch("stats.time_now")
    def test_model_and_supplier_speed_use_the_same_aggregation(self, mock_now):
        mock_now.return_value = self.now

        result = stats.summary({}, hours=1)

        by_model = {row["name"]: row for row in result["by_model"]}
        self.assertEqual(by_model["pixel/a"]["speed"], 20.0)  # 200 token / 10 秒
        self.assertEqual(by_model["xkiro/b"]["speed"], 200.0)  # 400 token / 2 秒
        by_supplier = {row["name"]: row for row in result["by_supplier"]}
        self.assertEqual(by_supplier["pixel"]["speed"], 20.0)
        self.assertEqual(by_supplier["xkiro"]["speed"], 200.0)

    @patch("stats.time_now")
    def test_speed_uses_generation_time_after_first_token(self, mock_now):
        mock_now.return_value = self.now
        stats._records = [
            {"ts": self.now - 3, "model": "pixel/a", "supplier": "pixel", "status": "ok",
             "output_tokens": 100, "duration_ms": 5000, "first_token_ms": 1000},
            {"ts": self.now - 2, "model": "pixel/a", "supplier": "pixel", "status": "ok",
             "output_tokens": 300, "duration_ms": 3000, "first_token_ms": 0},
        ]

        result = stats.summary({}, hours=1)

        # (100 + 300) token / ((5-1) + 3) 秒 = 400 / 7 ≈ 57.1 t/s
        self.assertEqual(result["total"]["speed"], 57.1)

    @patch("stats.time_now")
    def test_average_first_token_is_reported_per_bucket(self, mock_now):
        mock_now.return_value = self.now
        stats._records = [
            {"ts": self.now - 3, "model": "pixel/a", "supplier": "pixel", "status": "ok",
             "output_tokens": 10, "duration_ms": 5000, "first_token_ms": 1000},
            {"ts": self.now - 2, "model": "pixel/a", "supplier": "pixel", "status": "ok",
             "output_tokens": 10, "duration_ms": 7000, "first_token_ms": 3000},
        ]

        result = stats.summary({}, hours=1)

        self.assertEqual(result["total"]["avg_first_token_ms"], 2000.0)
        self.assertEqual(result["by_model"][0]["avg_first_token_ms"], 2000.0)
        self.assertEqual(result["by_supplier"][0]["avg_first_token_ms"], 2000.0)

    @patch("stats.time_now")
    def test_speed_is_zero_when_no_request_has_a_duration(self, mock_now):
        mock_now.return_value = self.now
        stats._records = [
            {"ts": self.now - 3, "model": "pixel/a", "supplier": "pixel", "status": "ok", "output_tokens": 10, "duration_ms": 0},
        ]

        result = stats.summary({}, hours=1)

        self.assertEqual(result["total"]["speed"], 0)
        self.assertEqual(result["total"]["avg_first_token_ms"], 0)


if __name__ == "__main__":
    unittest.main()
