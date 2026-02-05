"""Main module of profunda."""

from importlib.metadata import version

from profunda.compare_reports import compare
from profunda.controller import pandas_decorator
from profunda.profile_report import ProfileReport

__version__ = version("profunda")

__all__ = [
    "pandas_decorator",
    "ProfileReport",
    "__version__",
    "compare",
]
