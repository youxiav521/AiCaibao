"""Three-statement extraction service package.

Submodules are intentionally imported lazily by callers. Keeping this package
initializer lightweight avoids importing storage/MinIO dependencies during
strategy-only tests.
"""

__all__: list[str] = []
