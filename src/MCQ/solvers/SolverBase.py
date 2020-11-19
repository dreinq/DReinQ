from abc import ABC
from logging import Logger

from MCQ.utils.logging import PPrint
from MCQ import Consts

class SolverBase(ABC):
    BACKENDS = [ "scipy", "torch", "cupy" ]
    def __init__(self, backend: str = None, logger: Logger = None):
        assert backend in self.BACKENDS, f"Require backend is one of {self.BACKENDS}, got {backend}."
        self._backend = backend
        self._logger = logger or Consts.Logger

    def Summary(self):
        return {
            "Type": self.__class__.__name__,
            "Backend": self._backend
        }

    def __str__(self):
        return PPrint({
            "Type": self.__class__.__name__,
            "Backend": self._backend
        })
