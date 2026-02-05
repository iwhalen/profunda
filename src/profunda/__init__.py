"""Main module of profunda."""

import importlib.util
import warnings

from profunda.compare_reports import compare
from profunda.controller import pandas_decorator
from profunda.profile_report import ProfileReport

__all__ = [
    "pandas_decorator",
    "ProfileReport",
    "__version__",
    "compare",
]
