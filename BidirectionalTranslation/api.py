import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).parent / '..'
sys.path.append(str(PROJECT_ROOT))

from typing import Union, List

import cv2
import numpy as np
import torch
from .models.cyclegan_stft_model import CycleGANSTFTModel

import sanitize


def path_check(thing, type='img'):
    if isinstance(thing, str) or isinstance(thing, pathlib.Path):
        if type=='img':
            arr = sanitize.PILopen(thing)
        elif type=='npy':
            arr = np.load(thing)
    elif isinstance(thing, np.ndarray):
        arr = thing
    return arr

class BidirectionalTranslation_cvt:
    def __init__(self, model_name='BidirectionalTranslation', freeze_seed=None, device=None):
        self.freeze_seed = freeze_seed
        if not (self.freeze_seed is None):
            torch.manual_seed(self.freeze_seed)
            torch.cuda.manual_seed(self.freeze_seed)
            np.random.seed(self.freeze_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device is None else device
        main_dir = pathlib.Path(__file__).parent
        self.model = CycleGANSTFTModel(
            output_nc=3,
            nz=64,
            nef=48,
            ngf=48,
            init_gain=0.02,
            checkpoints_dir=str(main_dir/'checkpoints'),
            name=model_name,
            device=self.device
        )
        self.model.eval()

    def exec(self, mode: str, *input):
        if mode=='AtoB':
            img, line = input
            line = cv2.erode(line, np.ones((3,3), np.uint8))
            img_torch = sanitize.rgb2tensor(img)
            line_torch = sanitize.gray2tensor(line)
            img_torch = img_torch.to(self.device)
            line_torch = line_torch.to(self.device)
            if not (self.freeze_seed is None):
                torch.manual_seed(self.freeze_seed)
                torch.cuda.manual_seed(self.freeze_seed)
                np.random.seed(self.freeze_seed)
            scr_torch = self.model(mode, img_torch, line_torch)
            scr_torch = scr_torch.detach().cpu()
            scr = np.squeeze(scr_torch.numpy())
            return scr
        elif mode=='BtoA':
            scr, line, styref = input
            scr_torch = torch.from_numpy(scr)
            scr_torch = scr_torch.unsqueeze(0) if scr_torch.ndim==3 else scr_torch
            scr_torch = scr_torch.to(self.device)
            line_torch = sanitize.gray2tensor(line).to(self.device)
            styref = self.model.get_z_random(scr_torch.shape[0], 64, truncation=True, tvalue=1.25) \
                     if styref is None else \
                     torch.from_numpy(styref.copy()).float() 
            if not (self.freeze_seed is None):
                torch.manual_seed(self.freeze_seed)
                torch.cuda.manual_seed(self.freeze_seed)
                np.random.seed(self.freeze_seed)
            color_torch = self.model(mode, scr_torch, line_torch, styref)
            color_torch = sanitize.unsquash(color_torch)
            color_torch = color_torch.detach().cpu()
            color = np.squeeze(color_torch.numpy().transpose((0, 2, 3, 1)))
            color = sanitize.any2pic(color, method='clip')
            return color
        else:
            return

    def color2map(self,
        img: Union[str, pathlib.Path, np.ndarray],
        line: Union[str, pathlib.Path, np.ndarray]
    ) -> np.ndarray :
        img = path_check(img)
        line = path_check(line)
        img = sanitize.PILconvert(img, 'RGB')
        line = sanitize.PILconvert(line, 'L')
        with torch.no_grad():
            ret = self.exec('AtoB', img, line)
        return ret

    def map2color(self,
        scr: Union[str, pathlib.Path, np.ndarray],
        line: Union[str, pathlib.Path, np.ndarray]
    ) -> np.ndarray :
        scr = path_check(scr, type='npy')
        line = path_check(line)
        with torch.no_grad():
            ret = self.exec('BtoA', scr, line, None)
        return ret

    def img2map_batch(self,
        img: Union[List[np.ndarray], np.ndarray],
        line: Union[List[np.ndarray], np.ndarray]
    ) -> List[np.ndarray] :
        if not (isinstance(img, np.ndarray) and isinstance(line, np.ndarray)):
            imgshapes = [i.shape[:2] for i in img]
            lineshapes = [l.shape[:2] for l in line]
            assert (imgshapes[0])*len(img) == imgshapes == lineshapes
        img = [sanitize.PILconvert(i, 'RGB') for i in img]
        line = [sanitize.PILconvert(l, 'L') for l in line]
        img = np.array(img)
        line = np.array(line)
        with torch.no_grad():
            ret = self.exec('AtoB', img, line)
        ret = [r for r in ret]
        return ret

    def map2img_batch(self,
        scr: Union[List[np.ndarray], np.ndarray],
        line: Union[List[np.ndarray], np.ndarray]
    ) -> List[np.ndarray] :
        if not isinstance(scr, np.ndarray):
            scrshapes = [s.shape[-2:] for s in scr]
            lineshapes = [l.shape[:2] for l in line]
            assert (scrshapes[0])*len(scr) == scrshapes == lineshapes
        line = [sanitize.PILconvert(l, 'L') for l in line]
        line = np.array(line)
        with torch.no_grad():
            ret = self.exec('BtoA', scr, line, None)
        ret = [r for r in ret]
        return ret