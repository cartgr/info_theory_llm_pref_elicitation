"""Verbose logging for experiment tracing."""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class VerboseLogger:
    """Logger that writes detailed traces to both console and file."""

    def __init__(self, log_file: Optional[Path] = None, console: bool = True):
        self.log_file = log_file
        self.console = console
        self._file_handle = None
        self._indent = 0

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(log_file, 'w')

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def _write(self, text: str):
        indent = "  " * self._indent
        lines = text.split('\n')
        for line in lines:
            formatted = f"{indent}{line}\n"
            if self.console:
                sys.stdout.write(formatted)
                sys.stdout.flush()
            if self._file_handle:
                self._file_handle.write(formatted)
                self._file_handle.flush()

    @contextmanager
    def indent(self):
        self._indent += 1
        try:
            yield
        finally:
            self._indent -= 1

    def header(self, title: str, char: str = "=", width: int = 80):
        self._write("")
        self._write(char * width)
        self._write(title.center(width))
        self._write(char * width)

    def subheader(self, title: str, char: str = "-", width: int = 60):
        self._write("")
        self._write(char * width)
        self._write(title)
        self._write(char * width)

    def section(self, title: str):
        self._write(f"\n>>> {title}")

    def log(self, message: str):
        self._write(message)

    def blank(self):
        self._write("")

    def key_value(self, key: str, value: Any, width: int = 25):
        self._write(f"{key:<{width}}: {value}")

    def list_items(self, items: List[str], prefix: str = "-"):
        for item in items:
            self._write(f"{prefix} {item}")

    def numbered_list(self, items: List[str]):
        for i, item in enumerate(items, 1):
            self._write(f"{i}. {item}")

    def prompt_block(self, label: str, content: str):
        self._write(f"\n[{label}]")
        self._write("```")
        for line in content.split('\n'):
            self._write(line)
        self._write("```")

    def table_row(self, cols: List[str], widths: List[int]):
        row = ""
        for col, w in zip(cols, widths):
            row += f"{str(col):<{w}}"
        self._write(row)

    def divider(self, char: str = "-", width: int = 60):
        self._write(char * width)


# Global logger instance
_logger: Optional[VerboseLogger] = None


def init_logger(log_file: Optional[Path] = None, console: bool = True) -> VerboseLogger:
    """Initialize the global logger."""
    global _logger
    if _logger:
        _logger.close()
    _logger = VerboseLogger(log_file, console)
    return _logger


def get_logger() -> VerboseLogger:
    """Get the global logger (creates one if needed)."""
    global _logger
    if _logger is None:
        _logger = VerboseLogger()
    return _logger


def close_logger():
    """Close the global logger."""
    global _logger
    if _logger:
        _logger.close()
        _logger = None
