
import os
import shutil
import atexit
import time
import logging
import multiprocessing
from subprocess import Popen, PIPE
import json
import logging.config
from logging import LogRecord
import sys
import warnings
import datetime
from typing import Dict, Type

import tensorflow as tf
import gpustat

from MCQ import Consts
from MCQ.optimizers import GradientAccumulatedAdam

def HotPatch(function, elementSpec):
    return tf.function(elementSpec)(function)


def SplitToVirtualGPUs(avaliableGPUAndMems, vRamTotal):
    # split physical devices into parts to use more gpus
    # BUT, use virtual GPU can't perform NCCL all-reduce
    gpus = tf.config.experimental.list_physical_devices('GPU')
    minimumMem = min(avaliableGPUAndMems, key=lambda item: item[1])[1]
    allocateMem = int(minimumMem // (minimumMem / vRamTotal)) - 256
    print(allocateMem, "MB")
    if gpus:
        for (_, mem), gpu in zip(avaliableGPUAndMems, gpus):
            try:
                virtualGPUs = mem // vRamTotal
                tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=allocateMem) for _ in range(virtualGPUs)])
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    else:
        raise RuntimeError("Can't find gpu")


class DeprecationFilter(object):
    def filter(self, record: LogRecord):
        if "depreca" in record.msg:
            return 0
        return 1

def ConfigLogging(log_dir:str, root_name:str, level:str, rotate_logs:int=10, ignore_warnings:list=None) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    RotateItems(log_dir, rotate_logs)
    f_prefix = os.path.join(log_dir, "{0}".format(datetime.datetime.now().strftime("%m-%d %H:%M")))
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
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "info_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "full",
                "filename": f"{f_prefix}.log",
                "mode": "w"
            },
            "err_file": {
                "class": "logging.FileHandler",
                "level": "ERROR",
                "formatter": "full",
                "filename": f"{f_prefix}.err",
                "mode": "w"
            }
        },
        "loggers":{
            root_name: {
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

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = logging.getLogger(root_name)
        logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    def handle_warning(message, category, filename, lineno, file=None, line=None):
        logger = logging.getLogger(root_name)
        if ignore_warnings is not None and category in ignore_warnings:
            return
        logger.warning(warnings.formatwarning(message, category, filename, lineno, line))
    warnings.showwarning = handle_warning
    return logging.getLogger(root_name)


def PPrint(d:dict):
    return str(json.dumps(d, default=lambda x: x.__dict__, indent=4))

def QueryGPU(wantsMore: bool = True, givenList: list = None, needGPUs: int = -1, needVRamEachGPU: int = -1, WriteOSEnv: bool = True, logger: logging.Logger = None) -> list:

    logger = logger or Consts.Logger

    # keep the devices order same as in nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    gpus = gpustat.new_query().gpus
    if needGPUs < 0:
        needGPUs = len(gpus)

    # logger.debug("\n" + str(gpus))
    if isinstance(givenList, list):
        it = givenList
    else:
        it = range(len(gpus))

    gpus = [(i, gpus[i]) for i in it]
    if wantsMore:
        gpus = sorted(gpus, key=lambda item: item[1].entry['memory.used'])

    gpuList = []
    for i, g in gpus:
        if needVRamEachGPU < 0:
            if g.entry['memory.used'] < 64:
                # give space for basic vram
                gpuList.append((i, (g.entry['memory.total'] - g.entry['memory.used'] - 64)))
                logger.debug(f"adding gpu[{i}] with {g.entry['memory.total'] - g.entry['memory.used']} free.")
        elif g.entry['memory.total'] - g.entry['memory.used'] > needVRamEachGPU + 64:
                gpuList.append((i, (g.entry['memory.total'] - g.entry['memory.used'] - 64)))
                logger.debug(f"adding gpu[{i}] with {g.entry['memory.total'] - g.entry['memory.used']} free.")
        if len(gpuList) >= needGPUs and not wantsMore:
            break

    if len(gpuList) >= needGPUs:
        # keep order
        gpuList = sorted(gpuList, key=lambda item: item[0])
        logger.debug(f"Found {gpuList} satisfied")
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
            except:
                pass
        return gpuList
    else:
        raise EnvironmentError("Current system status is not satisfied")



def open_tensorboard(log_dir: str, port: int):
    process = Popen(["tensorboard", "--logdir", log_dir, "--port", str(port)], stdout=PIPE, stderr=PIPE)
    def cleanup():
        process.terminate()
        timeout_sec = 5
        for _ in range(timeout_sec):
            if process.poll() == None:
                time.sleep(1)
            else:
                Consts.Logger.info("Tensorboard quitted")
                return
        process.kill()
        Consts.Logger.info("Tensorboard killed")
    atexit.register(cleanup)
    Consts.Logger.info(f"Tensorboard opened at {port}")

def delete_dir(path):
    shutil.rmtree(path, ignore_errors=True)

class LoggingDisabler(object):
    def __init__(self, logger:logging.Logger, disable:bool):
        self._logger = logger
        self._disable = disable
    def __enter__(self):
        if self._disable:
            self._previous_status = self._logger.disabled
            self._logger.disabled = True
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._disable:
            self._logger.disabled = self._previous_status

def delete_files_older_than(folder:str, seconds:int):
    now = time.time()
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            if os.stat(os.path.join(folder, f)).st_mtime < now - seconds:
                os.remove(os.path.join(folder, f))

def RotateItems(folder:str, count:int):
    file_list = []
    for f in os.listdir(folder):
        file_list.append(f)
    file_list = sorted(file_list, key=lambda f:os.stat(os.path.join(folder, f)).st_mtime)
    if len(file_list) <= count:
        return
    for f in file_list[count:]:
        if os.path.isfile(f):
            os.remove(os.path.join(folder, f))
        else:
            shutil.rmtree(os.path.join(folder, f))


def override_flags_and_set_envars_for_gpu_thread_pool(num_gpus, tf_gpu_thread_mode="gpu_private", logger:logging.Logger=None):
    """Override flags and set env_vars for performance.
    These settings exist to test the difference between using stock settings
    and manual tuning. It also shows some of the ENV_VARS that can be tweaked to
    squeeze a few extra examples per second.  These settings are defaulted to the
    current platform of interest, which changes over time.
    On systems with small numbers of cpu cores, e.g. under 8 logical cores,
    setting up a gpu thread pool with `tf_gpu_thread_mode=gpu_private` may perform
    poorly.
    """
    logger = logger or Consts.Logger
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"Logical CPU cores: {cpu_count}")

    os.environ["NUMEXPR_MAX_THREADS"] = str(cpu_count)
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'


    # Sets up thread pool for each GPU for op scheduling.
    per_gpu_thread_count = 2
    total_gpu_thread_count = per_gpu_thread_count * num_gpus
    os.environ['TF_GPU_THREAD_MODE'] = tf_gpu_thread_mode
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    logger.info(f"TF_GPU_THREAD_COUNT: {os.environ['TF_GPU_THREAD_COUNT']}")
    logger.info(f"TF_GPU_THREAD_MODE: {os.environ['TF_GPU_THREAD_MODE']}")

    # Reduces general thread pool by number of threads used for GPU pool.
    main_thread_count = cpu_count - total_gpu_thread_count
    inter_op_parallelism_threads = main_thread_count

    # Sets thread count for tf.data. Logical cores minus threads assign to the
    # private GPU pool along with 2 thread per GPU for event monitoring and
    # sending / receiving tensors.
    num_monitoring_threads = 2 * num_gpus
    datasets_num_private_threads = (cpu_count - total_gpu_thread_count - num_monitoring_threads)

    return inter_op_parallelism_threads, datasets_num_private_threads
