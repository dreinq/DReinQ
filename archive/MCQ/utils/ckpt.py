import os, shutil, datetime

import tensorflow as tf

from MCQ.utils import RotateItems

def SaveAndRestore(savePath: str, isContinue: bool, **savingThings):
    ckpt = tf.train.Checkpoint(**savingThings)
    # ckpt = tf.train.Checkpoint(step=tf.Variable(0), tOptim=reinforce._modelOptimizer, cCOptim=reinforce._codebookOptimizer, model=reinforce._model, codebook=reinforce._codebook)
    manager = tf.train.CheckpointManager(ckpt, savePath, max_to_keep=3)
    if isContinue and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        if os.path.exists(savePath):
            newPath = os.path.join(os.path.join(savePath, os.path.pardir), datetime.datetime.now().strftime("%m-%d %H:%M"))
            shutil.move(savePath, newPath)
            print(savePath.split('/')[-1:])
            print("/".join(savePath.split('/')[-1:]))
            RotateItems("/".join(savePath.split('/')[-1:]), 3)
            print(f"Move old ckpt to {newPath} and rotate to keep 3 old ckpts.")
        print("Initializing from scratch.")

    return ckpt, manager
