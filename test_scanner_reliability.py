import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd


# The production repository installs these dependencies. Lightweight stubs let
# the reliability helpers be unit-tested in an offline development container.
if "requests" not in sys.modules:
    try:
        importlib.import_module("requests")
    except ModuleNotFoundError:
        requests_stub = types.ModuleType("requests")
        requests_stub.post = None
        sys.modules["requests"] = requests_stub

if "yfinance" not in sys.modules:
    try:
        importlib.import_module("yfinance")
    except ModuleNotFoundError:
        yfinance_stub = types.ModuleType("yfinance")
        yfinance_stub.download = None
        sys.modules["yfinance"] = yfinance_stub

import scanner


class ScannerReliabilityTests(unittest.TestCase):
    def test_large_multiline_tracker_text_is_split_below_limit(self):
        tracker_text = "\n".join(
            f"📈 <b>STOCK{i}</b> — checkpoint details" for i in range(500)
        )
        chunks = scanner.build_telegram_chunks(
            ["HEADER", tracker_text], max_units=500
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(scanner._telegram_units(chunk) <= 500 for chunk in chunks)
        )
        self.assertIn("STOCK0", chunks[0])
        self.assertIn("STOCK499", chunks[-1])

    def test_tracker_exception_becomes_warning_instead_of_escaping(self):
        with patch.object(
            scanner, "_track_and_report", side_effect=KeyError("legacy_field")
        ):
            result = scanner.safe_tracking_report({}, {})

        self.assertIn("Journey tracker unavailable", result)
        self.assertIn("Main scan completed", result)

    def test_full_scanner_still_sends_when_tracker_crashes(self):
        fake_data = {
            "TEST.NS": {
                "price": 123.45,
                "gap_today": -9.0,
            }
        }

        with (
            patch.object(scanner, "fetch_all_data", return_value=fake_data),
            patch.object(scanner, "classify", return_value=("accum", "TEST LINE")),
            patch.object(
                scanner, "_track_and_report", side_effect=KeyError("legacy_field")
            ),
            patch.object(scanner, "send_telegram") as send,
        ):
            scanner.run_scanner()

        self.assertGreaterEqual(send.call_count, 1)
        delivered = "\n".join(call.args[0] for call in send.call_args_list)
        self.assertIn("TEST LINE", delivered)
        self.assertIn("Journey tracker unavailable", delivered)

    def test_telegram_retries_and_then_succeeds(self):
        good_response = Mock()
        good_response.raise_for_status.return_value = None
        good_response.json.return_value = {"ok": True}

        with (
            patch.object(scanner, "TELEGRAM_BOT_TOKEN", "test-token"),
            patch.object(scanner, "TELEGRAM_CHAT_ID", "test-chat"),
            patch.object(
                scanner.requests,
                "post",
                side_effect=[RuntimeError("temporary"), good_response],
                create=True,
            ) as post,
            patch.object(scanner.time, "sleep"),
        ):
            scanner.send_telegram("test")

        self.assertEqual(post.call_count, 2)

    def test_ticker_frame_supports_both_multiindex_orders(self):
        ticker_first = pd.DataFrame(
            [[100.0, 10.0]],
            columns=pd.MultiIndex.from_tuples(
                [("ABC.NS", "Close"), ("ABC.NS", "Volume")]
            ),
        )
        price_first = pd.DataFrame(
            [[100.0, 10.0]],
            columns=pd.MultiIndex.from_tuples(
                [("Close", "ABC.NS"), ("Volume", "ABC.NS")]
            ),
        )

        self.assertEqual(scanner._ticker_frame(ticker_first, "ABC.NS")["Close"].iloc[0], 100)
        self.assertEqual(scanner._ticker_frame(price_first, "ABC.NS")["Close"].iloc[0], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
