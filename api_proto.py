import sanitize
import pathlib
import torch
import numpy as np
from abc import ABC, abstractmethod

class apiProto(ABC):
    def __init__(self, model_name, freeze_seed=None, device=None):
        self.freeze_seed = freeze_seed
        self.manual_seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device is None else device
        self.model_name = model_name

    def manual_seed(self):
        if not (self.freeze_seed is None):
            np.random.seed(self.freeze_seed)
            if type(self.freeze_seed) == int:
                torch.manual_seed(self.freeze_seed)
                torch.cuda.manual_seed(self.freeze_seed)
            elif type(self.freeze_seed) == bytes:
                arr = np.frombuffer(self.freeze_seed, dtype=np.uint8)
                Rtensor = torch.from_numpy(arr)
                torch.set_rng_state(Rtensor)
                torch.cuda.set_rng_state(Rtensor)
            else:
                raise ValueError
            

    @abstractmethod
    def exec(self, *args, **kwargs):
        pass

    @staticmethod
    def path_check(thing, type='img'):
        if isinstance(thing, str) or isinstance(thing, pathlib.Path):
            if type=='img':
                arr = sanitize.PILopen(thing)
            elif type=='npy':
                arr = np.load(thing)
        elif isinstance(thing, np.ndarray):
            arr = thing
        return arr