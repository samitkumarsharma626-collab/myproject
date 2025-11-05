from collections.abc import Iterator

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Iterator[None]:
    """Ensure configuration cache is cleared between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
