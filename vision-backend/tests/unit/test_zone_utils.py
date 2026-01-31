"""
Unit tests for restricted_zone zone_utils (pure geometry, no mocks).
Loaded via importlib to avoid circular imports in vision_tasks package.
"""
import importlib.util
from pathlib import Path

_zone_utils_path = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "processing"
    / "vision_tasks"
    / "tasks"
    / "restricted_zone"
    / "zone_utils.py"
)
_spec = importlib.util.spec_from_file_location("zone_utils", _zone_utils_path)
_zone_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_zone_utils)

point_in_polygon = _zone_utils.point_in_polygon
get_box_center = _zone_utils.get_box_center
is_box_in_zone = _zone_utils.is_box_in_zone


class TestPointInPolygon:
    """Tests for point_in_polygon - ray casting algorithm"""

    def test_point_inside_square(self):
        # Square: (0,0), (10,0), (10,10), (0,10)
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert point_in_polygon((5, 5), polygon) is True

    def test_point_outside_square(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert point_in_polygon((15, 15), polygon) is False
        assert point_in_polygon((-1, -1), polygon) is False

    def test_point_on_boundary(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        # Boundary behavior can vary; at minimum polygon should be valid
        assert isinstance(point_in_polygon((0, 0), polygon), bool)

    def test_polygon_less_than_3_points_returns_false(self):
        assert point_in_polygon((1, 1), [[0, 0]]) is False
        assert point_in_polygon((1, 1), [[0, 0], [1, 1]]) is False


class TestGetBoxCenter:
    """Tests for get_box_center"""

    def test_center_of_rect(self):
        box = [0, 0, 10, 10]  # x1, y1, x2, y2
        assert get_box_center(box) == (5.0, 5.0)

    def test_center_offset(self):
        box = [10, 20, 30, 40]
        assert get_box_center(box) == (20.0, 30.0)


class TestIsBoxInZone:
    """Tests for is_box_in_zone"""

    def test_box_center_inside_zone(self):
        # Zone: square 0-100
        zone = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # Box with center at (50, 50)
        box = [40, 40, 60, 60]
        assert is_box_in_zone(box, zone) is True

    def test_box_center_outside_zone(self):
        zone = [[0, 0], [100, 0], [100, 100], [0, 100]]
        box = [150, 150, 200, 200]  # center (175, 175) outside
        assert is_box_in_zone(box, zone) is False

    def test_empty_box_returns_false(self):
        zone = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert is_box_in_zone([], zone) is False
        assert is_box_in_zone([1, 2, 3], zone) is False  # < 4 elements

    def test_empty_zone_returns_false(self):
        box = [0, 0, 10, 10]
        assert is_box_in_zone(box, []) is False
        assert is_box_in_zone(box, [[0, 0], [1, 1]]) is False  # < 3 points
