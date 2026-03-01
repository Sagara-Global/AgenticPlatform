"""Built-in tools — ready-to-use tools for Suluv agents."""

from suluv.core.tools.builtins.calculator import calculator
from suluv.core.tools.builtins.json_extractor import json_extractor
from suluv.core.tools.builtins.web_search import web_search
from suluv.core.tools.builtins.http_fetch import http_fetch
from suluv.core.tools.builtins.file_reader import file_reader
from suluv.core.tools.builtins.file_writer import file_writer
from suluv.core.tools.builtins.shell_exec import shell_exec
from suluv.core.tools.builtins.datetime_tool import datetime_now, date_diff

__all__ = [
    "calculator",
    "json_extractor",
    "web_search",
    "http_fetch",
    "file_reader",
    "file_writer",
    "shell_exec",
    "datetime_now",
    "date_diff",
]
