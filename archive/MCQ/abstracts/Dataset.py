import abc


class Dataset(abc.ABC):
    def __init__(self, path: str):
        self._path = path

    @abc.abstractmethod
    def __call__(self, mode):
        ...