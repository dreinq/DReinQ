import os
import logging
import time

import torch
import pynvml

from MCQ import Consts

def PreAllocateMem(memSize: int):
    devices = torch.cuda.device_count()
    for d in range(devices):
        x = torch.rand((256, 1024, memSize), device=f"cuda:{d}")
        del x
    return

def GPUInfo():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    gpus = list()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus.append({ "memory.used": info.used / 1048576, "memory.total": info.total / 1048576 })
    return gpus

def QueryGPU(wantsMore: bool = True, givenList: list = None, needGPUs: int = -1, needVRamEachGPU: int = -1, WriteOSEnv: bool = True, logger: logging.Logger = None) -> list:

    logger = logger or Consts.Logger

    # keep the devices order same as in nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    gpus = GPUInfo()
    if needGPUs < 0:
        needGPUs = len(gpus)

    # logger.debug("\n" + str(gpus))
    if isinstance(givenList, list):
        it = givenList
    else:
        it = range(len(gpus))

    gpus = [(i, gpus[i]) for i in it]
    if wantsMore:
        gpus = sorted(gpus, key=lambda item: item['memory.used'])

    gpuList = []
    for i, g in gpus:
        if needVRamEachGPU < 0:
            if g['memory.used'] < 64:
                # give space for basic vram
                gpuList.append((i, (g['memory.total'] - g['memory.used'] - 64)))
                logger.debug("adding gpu[%d] with %f free.", i, g['memory.total'] - g['memory.used'])
        elif g['memory.total'] - g['memory.used'] > needVRamEachGPU + 64:
            gpuList.append((i, (g['memory.total'] - g['memory.used'] - 64)))
            logger.debug("adding gpu[%d] with %f free.", i, g['memory.total'] - g['memory.used'])
        if len(gpuList) >= needGPUs and not wantsMore:
            break

    if len(gpuList) >= needGPUs:
        # keep order
        gpuList = sorted(gpuList, key=lambda item: item[0])
        logger.debug("Found %d %s satisfied", len(gpuList), "gpu" if len(gpuList) == 1 else "gpus")
        if WriteOSEnv:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, [item[0] for item in gpuList]))
            newGPUList = []
            j = 0
            for i, mem in gpuList:
                newGPUList.append((j, mem))
                j += 1
            gpuList = newGPUList
        else:
            try:
                os.environ.pop("CUDA_VISIBLE_DEVICES")
            except KeyError:
                pass
        return gpuList
    else:
        raise EnvironmentError("Current system status is not satisfied")


class Timer:
    def __init__(self):
        self._initial = self._tick = time.time()

    def Tick(self):
        tock = time.time()
        interval = tock - self._tick
        self._tick = tock
        return interval, self._tick - self._initial
