# -*- coding: utf-8 -*-
import datetime
import unittest
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


if __name__ == "__main__":
    unittest.main()
