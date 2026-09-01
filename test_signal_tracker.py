import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import signal_tracker as tracker


class SignalTrackerMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temp_dir.name) / "signal_tracker_state.json"
        self.state_patch = patch.object(tracker, "STATE_FILE", self.state_path)
        self.state_patch.start()

    def tearDown(self):
        self.state_patch.stop()
        self.temp_dir.cleanup()

    def write_state(self, state):
        self.state_path.write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    def read_state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def test_all_legacy_records_are_preserved_and_migrated(self):
        legacy = []
        for index in range(23):
            legacy.append(
                {
                    "symbol": f"LEGACY{index}.NS",
                    "signal": "Accumulate",
                    "signal_date": "2026-08-09",
                    "entry_price": 100 + index,
                    "milestones_hit": [5],
                    "drawdown_alerted": False,
                    "peak_pct": 6.5,
                    "trough_pct": -2.0,
                }
            )
        legacy.append(
            {
                "symbol": "SWIGGY.NS",
                "signal": "Accumulate",
                "signal_date": "2026-08-30",
                "entry_price": 280.5,
                "peak_pct": 0.5,
                "trough_pct": -0.61,
                "checkpoint_reported": False,
            }
        )
        self.write_state({"tracking": legacy, "closed": []})
        prices = {item["symbol"]: item["entry_price"] * 1.02 for item in legacy}

        text = tracker.track_and_report(
            alerts={},
            prices=prices,
            now=datetime(2026, 9, 1, tzinfo=tracker.IST),
        )

        saved = self.read_state()
        self.assertEqual(saved["schema_version"], tracker.STATE_SCHEMA_VERSION)
        self.assertEqual(len(saved["tracking"]), 24)
        self.assertEqual(len(saved["closed"]), 0)
        self.assertTrue(
            all("checkpoint_reported" in item for item in saved["tracking"])
        )
        self.assertEqual(saved["tracking"][0]["milestones_hit"], [5])
        self.assertIn("15-DAY CHECKPOINT", text)

    def test_missed_weekend_checkpoint_is_reported_once(self):
        state = {
            "tracking": [
                {
                    "symbol": "TEST.NS",
                    "signal": "Buy Now",
                    "signal_date": "2026-08-14",
                    "entry_price": 100,
                    "peak_pct": 0,
                    "trough_pct": 0,
                }
            ],
            "closed": [],
        }
        self.write_state(state)
        now = datetime(2026, 8, 31, tzinfo=tracker.IST)  # age 17

        first = tracker.track_and_report({}, {"TEST.NS": 110}, now=now)
        second = tracker.track_and_report({}, {"TEST.NS": 111}, now=now)

        self.assertIn("15-DAY CHECKPOINT", first)
        self.assertNotIn("15-DAY CHECKPOINT", second)
        self.assertTrue(self.read_state()["tracking"][0]["checkpoint_reported"])

    def test_day_30_closes_without_emitting_late_checkpoint(self):
        self.write_state(
            {
                "tracking": [
                    {
                        "symbol": "TEST.NS",
                        "signal": "Accumulate",
                        "signal_date": "2026-08-01",
                        "entry_price": 100,
                        "peak_pct": 5,
                        "trough_pct": -3,
                    }
                ],
                "closed": [],
            }
        )

        text = tracker.track_and_report(
            {},
            {"TEST.NS": 112},
            now=datetime(2026, 8, 31, tzinfo=tracker.IST),
        )
        saved = self.read_state()

        self.assertEqual(saved["tracking"], [])
        self.assertEqual(len(saved["closed"]), 1)
        self.assertEqual(saved["closed"][0]["final_pct"], 12.0)
        self.assertIn("30-DAY FINAL", text)
        self.assertNotIn("15-DAY CHECKPOINT", text)

    def test_new_signals_filter_duplicates_and_preserve_cooldown(self):
        self.write_state(
            {
                "schema_version": tracker.STATE_SCHEMA_VERSION,
                "tracking": [
                    {
                        "symbol": "ACTIVE.NS",
                        "signal": "Buy Now",
                        "signal_date": "2026-08-30",
                        "entry_price": 100,
                        "peak_pct": 0,
                        "trough_pct": 0,
                        "checkpoint_reported": False,
                    }
                ],
                "closed": [
                    {
                        "symbol": "COOLDOWN.NS",
                        "signal": "Accumulate",
                        "signal_date": "2026-07-20",
                        "entry_price": 100,
                        "peak_pct": 5,
                        "trough_pct": -2,
                        "checkpoint_reported": True,
                        "close_date": "2026-08-28",
                        "close_price": 105,
                        "final_pct": 5,
                        "days_held": 30,
                    }
                ],
            }
        )
        alerts = {
            "ACTIVE.NS": {"signal": "Buy Now", "price": 101},
            "COOLDOWN.NS": {"signal": "Accumulate", "price": 106},
            "SUPPORT.NS": {"signal": "Support Zone", "price": 90},
            "NEW.NS": {"signal": "Aggressive Accumulation", "price": 200},
        }
        tracker.track_and_report(
            alerts,
            {"ACTIVE.NS": 101},
            now=datetime(2026, 9, 1, tzinfo=tracker.IST),
        )

        symbols = [item["symbol"] for item in self.read_state()["tracking"]]
        self.assertEqual(symbols.count("ACTIVE.NS"), 1)
        self.assertIn("NEW.NS", symbols)
        self.assertNotIn("COOLDOWN.NS", symbols)
        self.assertNotIn("SUPPORT.NS", symbols)

    def test_corrupt_state_is_not_overwritten(self):
        corrupt = "{not valid json"
        self.state_path.write_text(corrupt, encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "original file preserved"):
            tracker.track_and_report(
                {}, {}, now=datetime(2026, 9, 1, tzinfo=tracker.IST)
            )

        self.assertEqual(self.state_path.read_text(encoding="utf-8"), corrupt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
