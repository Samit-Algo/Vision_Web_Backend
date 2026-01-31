"""
Unit tests for camera use cases (ListCameras, GetCamera).
"""
from unittest.mock import AsyncMock

import pytest
from app.application.use_cases.camera.list_cameras import ListCamerasUseCase
from app.application.use_cases.camera.get_camera import GetCameraUseCase
from app.domain.models.camera import Camera


def _make_camera(cam_id: str, owner: str, name: str = "Cam 1") -> Camera:
    return Camera(
        id=cam_id,
        owner_user_id=owner,
        name=name,
        stream_url="rtsp://localhost/stream",
    )


class TestListCamerasUseCase:
    """Tests for ListCamerasUseCase"""

    @pytest.mark.asyncio
    async def test_list_empty(self):
        repo = AsyncMock()
        repo.find_by_owner.return_value = []
        use_case = ListCamerasUseCase(repo)
        result = await use_case.execute("user-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_returns_cameras(self):
        repo = AsyncMock()
        repo.find_by_owner.return_value = [
            _make_camera("cam-1", "user-1", "Front Door"),
            _make_camera("cam-2", "user-1", "Backyard"),
        ]
        use_case = ListCamerasUseCase(repo)
        result = await use_case.execute("user-1")
        assert len(result) == 2
        assert result[0].id == "cam-1"
        assert result[0].name == "Front Door"
        assert result[1].id == "cam-2"


class TestGetCameraUseCase:
    """Tests for GetCameraUseCase"""

    @pytest.mark.asyncio
    async def test_get_found(self):
        repo = AsyncMock()
        repo.find_by_id.return_value = _make_camera("cam-1", "user-1", "My Camera")
        use_case = GetCameraUseCase(repo)
        result = await use_case.execute("cam-1", "user-1")
        assert result.id == "cam-1"
        assert result.name == "My Camera"

    @pytest.mark.asyncio
    async def test_get_not_found_raises(self):
        repo = AsyncMock()
        repo.find_by_id.return_value = None
        use_case = GetCameraUseCase(repo)
        with pytest.raises(ValueError, match="Camera not found"):
            await use_case.execute("nonexistent", "user-1")

    @pytest.mark.asyncio
    async def test_get_wrong_owner_raises(self):
        repo = AsyncMock()
        repo.find_by_id.return_value = _make_camera("cam-1", "other-user", "Cam")
        use_case = GetCameraUseCase(repo)
        with pytest.raises(ValueError, match="Camera not found"):
            await use_case.execute("cam-1", "user-1")
