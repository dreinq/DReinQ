
import logging
from typing import Dict, Type as T, Callable, Union

import tensorflow as tf


class Consts(object):
    GlobalBatchSize = -1
    CurrentMode = "Train"
    SummarySaveStep = 100
    TransferMatrix = None
    TransferMatrixT = None
    MultiGPU = False
    TowerName = "tower"
    LoggerName = "tensorflow"
    LogDir = "../log"
    TempDir = "/tmp/qframework/"
    DataDir = "../data"
    SaveDir = "../saved"
    ConfigDir = "../config"
    Metadata = "metadata.json"
    Logger = logging.getLogger(LoggerName)
    Eps = 1e-7

    class Type(object):
        def __contains__(self, mode: str):
            return mode in [self.Classification, self.Detection]
        Classification = "classification"
        Detection = "detection"

        def __repr__(self):
            return repr([self.Classification, self.Detection])

        def __str__(self):
            return str([self.Classification, self.Detection])

    class WarmStart(object):
        Compatible = "compatible"
        All = "all"
        RemoveScope = "remove-scope"

    class Params(object):
        Backbone = {}
        Extractor = {}
    Params.Extractor.update({
        Type.Detection: {
            "min_score_threshold": 0.5,
            "max_box_to_hold": 32,
            "sort_by_score": "descending"
        },
        Type.Classification: {

        }
    })

    class Mode(object):
        def __contains__(self, mode: str):
            return mode in [self.Train, self.Encode, self.Test, self.Eval]
        Train = "train"
        Encode = "base"
        Test = "infer"
        Eval = "eval"

        def __repr__(self):
            return repr([self.Train, self.Encode, self.Test, self.Eval])

        def __str__(self):
            return str([self.Train, self.Encode, self.Test, self.Eval])

    class File(object):
        StaticCenters = "centers.npy"
