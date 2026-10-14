"""Project level pytest configuration."""

from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config):  # pytest >= 7/8 signature
    """Exclude vendored upstream test-suites from collection.

    NOTE: the previous implementation used the pre-7.0 ``(path, config)``
    signature, which pytest 8 silently ignores — the exclusions never fired.
    """
    if "resources/" in str(collection_path):
        return True
    return None
