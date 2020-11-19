import abc
import tensorflow as tf
import tensorflow.python.distribute.values as V

class HotPatch(abc.ABC):
    def __init__(self):
        self._patched = dict()

    def HotPatch(self, func, elementSpec):
        if isinstance(elementSpec, (tf.TensorSpec, V.PerReplicaSpec)):
            elementSpec = (elementSpec, )
        if func.__name__ in self._patched:
            return
        setattr(self, func.__name__, tf.function(input_signature=elementSpec)(func))
        self._patched[func.__name__] = True