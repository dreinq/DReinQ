import os
import sys
import json
import warnings
import logging
import logging.config
from logging import LogRecord
import datetime
import multiprocessing
import time
import inspect
import functools

import tqdm

from MCQ.base import DecoratorContextManager

from .io import RotateItems


class WaitingBar(DecoratorContextManager):
    def __init__(self, msg: str, ncols: int = 10):
        assert ncols > 8, f"ncols must greater than 8, got {ncols}"
        self._msg = msg
        self._ticker = None
        self._stillRunning = None
        self._ncols = ncols
        self.animation = list()
        # "       =       "
        template = (" " * (ncols + 1) + "=" * (ncols - 8) + " " * (ncols + 1))
        for i in range(2 * (ncols - 2)):
            start = 2 * (ncols - 2) - i
            end = 3 * (ncols - 2) - i
            self.animation.append("[" + template[start:end] + "]" + r" %s")

    def __enter__(self):
        self._stillRunning = multiprocessing.Value("b", True)
        self._ticker = multiprocessing.Process(name="waitingBarTicker", target=self._print, args=[self._stillRunning])
        self._ticker.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stillRunning.value = False
        self._ticker.join()
        print(" " * (len(self._msg) + self._ncols + 1), end="\r", file=sys.stderr)

    def _print(self, stillRunning: multiprocessing.Value):
        i = 0
        while bool(stillRunning.value):
            print(self.animation[i % len(self.animation)] % self._msg, end='\r', file=sys.stderr)
            time.sleep(.06)
            i += 1


class LoggingDisabler:
    def __init__(self, logger: logging.Logger, disable: bool):
        self._logger = logger
        self._disable = disable
        self._previous_status = False

    def __enter__(self):
        if self._disable:
            self._previous_status = self._logger.disabled
            self._logger.disabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._disable:
            self._logger.disabled = self._previous_status


class DeprecationFilter:
    def filter(self, record: LogRecord):
        if "depreca" in record.msg:
            return 0
        return 1

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def ConfigLogging(logDir: str, rootName: str, level: str, useTqdm: bool = False, logName: str = None, rotateLogs: int = 10, ignoreWarnings: list = None) -> logging.Logger:
    os.makedirs(logDir, exist_ok=True)
    if rotateLogs > 0:
        RotateItems(logDir, rotateLogs)
    if logName is None:
        fPrefix = os.path.join(logDir, "{0}".format(datetime.datetime.now().strftime(r"%y%m%d-%H%M%S")))
    else:
        fPrefix = os.path.join(logDir, logName)
    logging_config = {
        "version": 1,
        "formatters": {
            "full": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": "%(asctime)s - %(message)s",
                "datefmt": "%m/%d %H:%M:%S"
            }
        },
        "filters": {
            "deprecation": {
                "()": DeprecationFilter
            }
        },
        "handlers": {
            "console": {
                "class": TqdmLoggingHandler if useTqdm else "logging.StreamHandler",
                "level": level,
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "info_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "full",
                "filename": f"{fPrefix}.log",
                "mode": "w"
            },
            "err_file": {
                "class": "logging.FileHandler",
                "level": "ERROR",
                "formatter": "full",
                "filename": f"{fPrefix}.err",
                "mode": "w"
            }
        },
        "loggers": {
            rootName: {
                "propagate": False,
                "level": level,
                "handlers": [
                    "console",
                    "info_file",
                    "err_file"
                ],
                "filters": [
                    "deprecation"
                ]
            }
        }
    }
    logging.config.dictConfig(logging_config)

    def handleException(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = logging.getLogger(rootName)
        logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handleException

    def handleWarning(message, category, filename, lineno, file=None, line=None):
        logger = logging.getLogger(rootName)
        if ignoreWarnings is not None and category in ignoreWarnings:
            return
        logger.warning(warnings.formatwarning(message, category, filename, lineno, line))
    warnings.showwarning = handleWarning
    return logging.getLogger(rootName)


def PPrint(d: dict) -> str:
    return str(json.dumps(d, default=lambda x: x.__dict__, indent=4))
