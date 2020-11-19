
import os
import shutil
import atexit
import time
from subprocess import Popen, PIPE

from MCQ import Consts


def OpenTensorBoard(log_dir: str, port: int):
    process = Popen(["tensorboard", "--logdir", log_dir, "--port", str(port)], stdout=PIPE, stderr=PIPE)

    def cleanup():
        process.terminate()
        timeout_sec = 5
        for _ in range(timeout_sec):
            if process.poll() is None:
                time.sleep(1)
            else:
                Consts.Logger.info("Tensorboard quitted")
                return
        process.kill()
        Consts.Logger.info("Tensorboard killed")
    atexit.register(cleanup)
    Consts.Logger.info("Tensorboard opened at %d", port)

def DeleteDir(path):
    shutil.rmtree(path, ignore_errors=True)

def DeleteFilesOlderThan(folder: str, seconds: int):
    now = time.time()
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            if os.stat(os.path.join(folder, f)).st_mtime < now - seconds:
                os.remove(os.path.join(folder, f))

def RotateItems(folder: str, count: int):
    file_list = []
    for f in os.listdir(folder):
        file_list.append(f)
    file_list = sorted(file_list, key=lambda f: os.stat(os.path.join(folder, f)).st_mtime)
    if len(file_list) <= count:
        return
    for f in file_list[:-count]:
        f = os.path.join(folder, f) 
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)
