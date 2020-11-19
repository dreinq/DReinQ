from typing import Any

class Restorable:
    def __init__(self):
        self.valuesToSave = set()

    def state_dict(self):
        return { key: self.__dict__[key] for key in self.valuesToSave }

    def load_state_dict(self, stateDict):
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                continue
            if callable(getattr(value, "load_state_dict", None)):
                value.load_state_dict(stateDict[key])
            else:
                self.__dict__[key] = stateDict[key]

    def __setattr__(self, name: str, value: Any):
        self.__dict__[name] = value
        if name.startswith("_"):
            self.valuesToSave.add(name)
