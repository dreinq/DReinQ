import os
import shutil
import datetime
from logging import Logger

import torch
from torch.utils.tensorboard import SummaryWriter

from MCQ.utils import RotateItems
from MCQ import Consts

class Saver(SummaryWriter):
    def __init__(self, config, saveDir: str, autoManage: bool = True, rotateItems: int = 25, reserve: bool = False):
        if Consts.NewestDir in saveDir:
            autoManage = False

        if autoManage:
            if os.path.exists(os.path.join(saveDir, Consts.NewestDir)) and not reserve:
                shutil.move(os.path.join(saveDir, Consts.NewestDir), os.path.join(saveDir, datetime.datetime.now().strftime(r"%y%m%d-%H%M%S")))
            os.makedirs(os.path.join(saveDir, Consts.NewestDir), exist_ok=True)
            RotateItems(saveDir, rotateItems)
            self._saveDir = os.path.join(saveDir, Consts.NewestDir)
        else:
            self._saveDir = saveDir
        super().__init__(self._saveDir)
        self._savePath = os.path.join(self._saveDir, Consts.CheckpointName)
        if not reserve:
            self._dumpFile(config)

    def _dumpFile(self, config):
        rootDir = Consts.RootDir
        shutil.copytree(rootDir, os.path.join(self._saveDir, "dump"), symlinks=True, ignore=lambda src, path: [x for x in path if x == "__pycache__"], ignore_dangling_symlinks=True)
        with open(os.path.join(self._saveDir, "config.json"), "w") as fp:
            fp.write(str(config))

    @property
    def SaveDir(self):
        return self._saveDir

    @property
    def SavePath(self):
        return self._savePath

    def Save(self, logger: Logger = None, **objs):
        saveDict = dict()
        for key, value in objs.items():
            if isinstance(value, torch.nn.DataParallel):
                saveDict[key] = value.module.state_dict()
            elif hasattr(value, "state_dict"):
                saveDict[key] = value.state_dict()
            else:
                saveDict[key] = value
        torch.save(saveDict, self._savePath)
        (logger or Consts.Logger).debug("Successfully saved checkpoint with keys: %s", list(saveDict.keys()))

    @staticmethod
    def Load(filePath, logger: Logger = None, **objs):
        savedDict = torch.load(filePath)
        logger = logger or Consts.Logger
        logger.debug("Load state_dict with keys:\r\n%s", savedDict.keys())
        for key, value in objs.items():
            stateDict = savedDict[key]
            if isinstance(value, torch.nn.DataParallel):
                value.module.load_state_dict(stateDict)
            elif callable(getattr(value, "load_state_dict", None)):
                value.load_state_dict(stateDict)
            else:
                value.data = stateDict
        return objs
