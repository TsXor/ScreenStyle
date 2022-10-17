from typing import Union, List

import torch
from .models.screenvae import ScreenVAE

import numpy as np
from sklearn.decomposition import PCA

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).parent / '..'
sys.path.append(str(PROJECT_ROOT))
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

class ScreenVAE_rec:
    def __init__(self, model_name='ScreenVAE', freeze_seed=None, device=None):
        self.freeze_seed = freeze_seed
        if not (self.freeze_seed is None):
            torch.manual_seed(self.freeze_seed)
            torch.cuda.manual_seed(self.freeze_seed)
            np.random.seed(self.freeze_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device is None else device
        main_dir = pathlib.Path(__file__).parent
        self.model_trainable = ScreenVAE(
            inc=1,
            outc=4,
            blocks=3,
            save_dir=str(main_dir/'checkpoints'/model_name),
            device=self.device
        )
        self.model = self.model_trainable.eval()

    def exec(self, mode: str, *input):
        if mode=='encode':
            img, line = input
            img = img[np.newaxis,...] if img.ndim==2 else img
            line = line[np.newaxis,...] if line.ndim==2 else line
            img = img[:,np.newaxis,:,:]
            line = line[:,np.newaxis,:,:]
            img_torch = torch.from_numpy(img.copy()).float()
            line_torch = torch.from_numpy(line.copy()).float()
            img_torch = sanitize.squash(img_torch)
            line_torch = sanitize.squash(line_torch)
            img_torch = img_torch.to(self.device)
            line_torch = line_torch.to(self.device)
            if not (self.freeze_seed is None):
                torch.manual_seed(self.freeze_seed)
                torch.cuda.manual_seed(self.freeze_seed)
                np.random.seed(self.freeze_seed)
            ret_torch, _ = self.model(mode, img_torch, line_torch)
            ret_torch = ret_torch.detach()
            scr_torch = ret_torch*(line_torch+1)/2
            scr_torch = scr_torch.cpu()
            scr = np.squeeze(scr_torch.numpy())
            return scr
        elif mode=='decode':
            scr, line = input
            scr_torch = torch.from_numpy(scr)
            scr_torch = scr_torch.unsqueeze(0) if scr_torch.ndim==3 else scr_torch
            scr_torch = scr_torch.float()
            scr_torch = scr_torch.to(self.device)
            if line is None:
                line_torch = None
            else:
                line_torch = torch.from_numpy(line)
                line_torch = line_torch.float()
                line_torch = line_torch.to(self.device)
            if not (self.freeze_seed is None):
                torch.manual_seed(self.freeze_seed)
                torch.cuda.manual_seed(self.freeze_seed)
                np.random.seed(self.freeze_seed)
            recons_torch = self.model(mode, scr_torch, line_torch)
            recons_torch = sanitize.unsquash(recons_torch)
            recons_torch = recons_torch.cpu().detach()
            recons = np.squeeze(recons_torch.numpy())
            recons = sanitize.any2pic(recons, method='clip')
            return recons
        else:
            return

    def img2map(self,
        img: Union[str, pathlib.Path, np.ndarray],
        line: Union[str, pathlib.Path, np.ndarray]
    ) -> np.ndarray :
        img = path_check(img)
        line = path_check(line)
        img = sanitize.PILconvert(img, 'L')
        line = sanitize.PILconvert(line, 'L')
        with torch.no_grad():
            ret = self.exec('encode', img, line)
        return ret

    def map2img(self,
        scr: Union[str, pathlib.Path, np.ndarray]
    ) -> np.ndarray :
        scr = path_check(scr, type='npy')
        with torch.no_grad():
            ret = self.exec('decode', scr, None)
        return ret

    def img2map_batch(self,
        img: Union[List[np.ndarray], np.ndarray],
        line: Union[List[np.ndarray], np.ndarray]
    ) -> List[np.ndarray] :
        if not (isinstance(img, np.ndarray) and isinstance(line, np.ndarray)):
            imgshapes = [i.shape for i in img]
            lineshapes = [l.shape for l in line]
            assert (imgshapes[0])*len(img) == imgshapes == lineshapes
        img = [sanitize.PILconvert(i, 'L') for i in img]
        line = [sanitize.PILconvert(l, 'L') for l in line]
        img = np.array(img)
        line = np.array(line)
        with torch.no_grad():
            ret = self.exec('encode', img, line)
        ret = [r for r in ret]
        return ret

    def map2img_batch(self,
        scr: Union[List[np.ndarray], np.ndarray]
    ) -> List[np.ndarray] :
        if not isinstance(scr, np.ndarray):
            scrshapes = [s.shape for s in scr]
            assert (scrshapes[0])*len(scr) == scrshapes
        with torch.no_grad():
            ret = self.exec('decode', scr, None)
        ret = [r for r in ret]
        return ret

    @staticmethod
    def get_pca(scr):
        result = np.concatenate([im.reshape(1,-1) for im in scr], axis=0)
        pca = PCA(n_components=3)
        pca.fit(result)
        result = pca.components_.copy()
        result = result.transpose().reshape((scr.shape[1], scr.shape[2], 3))
        for i in range(3):
            tmppic = result[:,:,i]
            result[:,:,i] = (tmppic - tmppic.min()) / (tmppic.max() - tmppic.min())
            # cv2.normalize(tmppic,resultPic[:,:,i],0,255,dtype=cv2.NORM_MINMAX)
        return result
