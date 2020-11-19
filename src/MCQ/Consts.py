
import logging
import os

import MCQ

srcRoot = os.path.join(os.path.dirname(os.path.abspath(MCQ.__file__)), os.pardir)

class Consts:
    CheckpointName = "saved.ckpt"
    DumpConfigName = "config.json"
    NewestDir = "latest"
    LoggerName = "main"
    RootDir = srcRoot
    LogDir = os.path.join(srcRoot, "../log")
    TempDir = "/tmp/MCQ/"
    DataDir = os.path.join(srcRoot, "../data")
    SaveDir = os.path.join(srcRoot, "../saved")
    Logger = logging.getLogger(LoggerName)
    Eps = 1e-7
