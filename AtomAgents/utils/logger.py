"""
Logging configuration for AtomAgents.
"""

import logging
import sys
from typing import Optional

__all__ = ["configure_agent_logger"]

# --------------------------------------------------------------------------- #
# 1.  Minimal in-house ANSI colour formatter (only used if colorlog is absent)
# --------------------------------------------------------------------------- #
RESET = "\033[0m"
_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[41m\033[97m",  # white on red background
}


class _AnsiColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = _COLORS.get(record.levelno, "")
        return f"{color}{msg}{RESET}"


# --------------------------------------------------------------------------- #
# 2.  Public API
# --------------------------------------------------------------------------- #
def configure_agent_logger(
    verbose: bool = True,
    use_color: Optional[bool] = None,  # None = auto-detect
) -> None:
    """
    Configure the root logger for AtomAgents.

    Parameters
    ----------
    verbose : bool, default True
        DEBUG level if True, INFO otherwise.
    use_color : bool or None, default None
        • True  – force colour
        • False – force plain
        • None  – colour only if stderr is an interactive terminal
    """
    level = logging.DEBUG if verbose else logging.INFO
    # Configure the AtomAgents namespace, not the root logger: clearing root
    # handlers destroyed uvicorn's (and other libraries') logging, and the
    # handler-level INFO cap silently discarded the per-agent DEBUG records
    # the verbose flag was supposed to enable.
    root_logger = logging.getLogger("AtomAgents")
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.propagate = False

    # Decide whether to emit colour
    if use_color is None:
        use_color = sys.stderr.isatty()

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)  # logger level is the single gate

    if use_color:
        try:
            # Prefer the richer third-party formatter if available
            from colorlog import ColoredFormatter

            handler.setFormatter(
                ColoredFormatter(
                    fmt=(
                        "%(log_color)s%(asctime)s "
                        "[%(levelname)s] %(name)s:%(reset)s "
                        "%(message_log_color)s%(message)s"
                    ),
                    datefmt="%Y-%m-%d %H:%M:%S",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                    secondary_log_colors={
                        "message": {
                            "DEBUG": "cyan",
                            "INFO": "white",
                            "WARNING": "yellow",
                            "ERROR": "red",
                            "CRITICAL": "red",
                        }
                    },
                    style="%",
                )
            )
        except ModuleNotFoundError:
            # Fall back to the built-in ANSI formatter
            handler.setFormatter(
                _AnsiColorFormatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
    else:
        # Plain, portable formatter (no ANSI codes)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)
