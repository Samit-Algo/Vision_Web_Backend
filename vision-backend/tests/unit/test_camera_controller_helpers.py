"""
Unit tests for camera_controller helper functions (e.g. _reconstruct_frame).
"""
import numpy as np
import pytest

from app.api.v1.camera_controller import _reconstruct_frame


class TestReconstructFrame:
    """Tests for _reconstruct_frame"""

    def test_valid_entry_returns_frame(self):
        arr = np.zeros((10, 20, 3), dtype=np.uint8)
        arr[5, 10, 1] = 255
        entry = {
            "bytes": arr.tobytes(),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        result = _reconstruct_frame(entry)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 20, 3)
        assert result[5, 10, 1] == 255

    def test_empty_entry_returns_none(self):
        assert _reconstruct_frame(None) is None
        assert _reconstruct_frame({}) is None

    def test_missing_keys_returns_none(self):
        assert _reconstruct_frame({"bytes": b"x"}) is None
        assert _reconstruct_frame({"shape": [1, 2, 3]}) is None
        assert _reconstruct_frame({"dtype": "uint8"}) is None

    def test_invalid_shape_returns_none(self):
        entry = {
            "bytes": bytes(100),
            "shape": [5, 5, 3],  # 75 bytes expected, 100 provided
            "dtype": "uint8",
        }
        result = _reconstruct_frame(entry)
        assert result is None
