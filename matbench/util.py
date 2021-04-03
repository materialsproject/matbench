"""
Utils and quality-of-life enhancements.
"""

import os
import sys
import json
import logging
import datetime


class RecursiveDotDict(dict):
    """
    Adapted from user Curt Hagenlocher from
    https://stackoverflow.com/questions/3031219/recursively
    -access-dict-via-attributes-as-well-as-index-access
    """

    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("expected dict")

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, RecursiveDotDict):
            value = RecursiveDotDict(value)
        super(RecursiveDotDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, RecursiveDotDict.MARKER)
        if found is RecursiveDotDict.MARKER:
            found = RecursiveDotDict()
            super(RecursiveDotDict, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


class MSONable2File:
    """
    Adds some very basic to_file and from_file methods common to
    all msonables in this package.
    """

    def to_file(self, filename):
        with open(filename, "w") as f:
            d = self.as_dict()
            json.dump(d, f)

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


def initialize_logger(logger_name, log_dir=None, level=None) -> logging.Logger:
    """Initialize the default logger with stdout and file handlers.
    Args:
        logger_name (str): The package name.
        log_dir (str, None): Path to the folder where the log file will be
            written. If None, no file will be written.
        level (int): The log level. For example logging.DEBUG.
    Returns:
        (Logger): A logging instance with customized formatter and handlers.
    """
    level = level or logging.INFO

    logger = logging.getLogger(logger_name)
    logger.handlers = []  # reset logging handlers if they already exist

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if log_dir:
        logpath = os.path.join(log_dir, logger_name)
        if os.path.exists(logpath + ".log"):
            logpath += "_" + datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S")
        logpath += ".log"

        handler = logging.FileHandler(logpath, mode="w")
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    screen_handler.setLevel(level)
    logger.addHandler(screen_handler)

    logger.setLevel(level)
    return logger
