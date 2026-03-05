from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

"""
Only add imports with the core dependencies here at the top level.
For the optional extras, these need to be imported from their respective sub packages.
"""

__all__ = [
    "__version__",
]

try:
    __version__ = version("genai-tools")
except PackageNotFoundError:  # pragma: no cover - during local dev without install
    __version__ = "0.0.0"
