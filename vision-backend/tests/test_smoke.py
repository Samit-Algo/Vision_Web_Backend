"""
Smoke test - verifies test infrastructure is working.
Run: pytest tests/test_smoke.py -v
"""


def test_import_app():
    """Verify app package can be imported."""
    from app.core.config import get_settings

    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, "mongo_uri")


def test_pytest_runs():
    """Basic sanity check that pytest executes tests."""
    assert True
