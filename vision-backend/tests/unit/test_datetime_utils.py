"""
Unit tests for app.utils.datetime_utils
"""
from datetime import datetime, timezone

import pytest
from app.utils.datetime_utils import ensure_utc, parse_iso, to_iso, utc_now


class TestEnsureUtc:
    """Tests for ensure_utc - no config needed"""

    def test_none_returns_none(self):
        assert ensure_utc(None) is None

    def test_naive_assumed_utc(self):
        dt = datetime(2025, 1, 15, 12, 0, 0)
        result = ensure_utc(dt)
        assert result.tzinfo == timezone.utc
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_aware_converted_to_utc(self):
        from datetime import timedelta
        # UTC+5:30
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        result = ensure_utc(dt)
        assert result.tzinfo == timezone.utc
        assert result.hour == 6  # 12 - 5.5 = 6:30
        assert result.minute == 30


class TestParseIso:
    """Tests for parse_iso"""

    def test_none_empty_returns_none(self, mock_settings):
        assert parse_iso(None) is None
        assert parse_iso("") is None

    def test_parse_utc_z_suffix(self, mock_settings):
        dt = parse_iso("2025-01-15T12:00:00Z")
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12

    def test_parse_with_offset(self, mock_settings):
        dt = parse_iso("2025-01-15T12:00:00+05:30")
        assert dt is not None
        assert dt.year == 2025

    def test_invalid_returns_none(self, mock_settings):
        assert parse_iso("not-a-date") is None
        assert parse_iso("2025-13-45T99:99:99") is None


class TestToIso:
    """Tests for to_iso"""

    def test_none_returns_none(self, mock_settings):
        assert to_iso(None) is None

    def test_utc_datetime_format(self, mock_settings):
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = to_iso(dt)
        assert result is not None
        assert "2025-01-15" in result
        assert "12:00:00" in result
