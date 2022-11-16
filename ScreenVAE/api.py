import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).parent / '..'
sys.path.append(str(PROJECT_ROOT))

from typing import Union, List
import sanitize
from api_proto import apiProto

import numpy as np
import torch
from .models.screenvae import ScreenVAE
from .models.screenvae_alt import define_SVAE

from sklearn.decomposition import PCA


class ScreenVAE_rec(apiProto):
    def __init__(self, model_name='ScreenVAE', freeze_seed=None, device=None, BTalter=False):
        super().__init__(model_name, freeze_seed, device)
        self.main_dir = pathlib.Path(__file__).parent
        if BTalter:
            init_type = 'kaiming'
        else:
            init_type = 'normal'
        self.model = ScreenVAE(
            inc=1,
            outc=4,
            blocks=3,
            init_type=init_type,
            save_dir=str(self.main_dir/'checkpoints'/self.model_name),
            device=self.device
        )
        self.model = self.model.eval()

    def exec(self, mode: str, *input, rawscr=False):
        if mode=='encode':
            img, line = input
            img_torch = sanitize.gray2tensor(img)
            line_torch = sanitize.gray2tensor(line)
            img_torch = img_torch.to(self.device)
            line_torch = line_torch.to(self.device)
            self.manual_seed()
            ret_torch, _ = self.model(mode, img_torch, line_torch)
            ret_torch = ret_torch.detach()
            if rawscr: return ret_torch
            scr_torch = ret_torch*(line_torch+1)/2
            scr_torch = scr_torch.cpu()
            scr = np.squeeze(scr_torch.numpy())
            return scr
        elif mode=='decode':
            scr, line = input
            if rawscr:
                scr_torch = scr
            else:
                scr_torch = torch.from_numpy(scr)
                scr_torch = scr_torch.unsqueeze(0) if scr_torch.ndim==3 else scr_torch
                scr_torch = scr_torch.to(self.device)
            line_torch = None if line is None else sanitize.gray2tensor(line).to(self.device)
            self.manual_seed()
            recons_torch = self.model(mode, scr_torch, line_torch)
            recons_torch = sanitize.unsquash(recons_torch)
            recons_torch = recons_torch.detach().cpu()
            recons = np.squeeze(recons_torch.numpy().transpose((1, 0, 2, 3)))
            recons = sanitize.any2pic(recons, method='clip')
            return recons
        else:
            return

    def img2map(self,
        img: Union[str, pathlib.Path, np.ndarray],
        line: Union[str, pathlib.Path, np.ndarray],
        rawscr=False
    ) -> np.ndarray :
        img = self.path_check(img)
        line = self.path_check(line)
        img = sanitize.PILconvert(img, 'L')
        line = sanitize.PILconvert(line, 'L')
        with torch.no_grad():
            ret = self.exec('encode', img, line, rawscr=rawscr)
        return ret

    def map2img(self,
        scr: Union[str, pathlib.Path, np.ndarray],
        rawscr=False
    ) -> np.ndarray :
        if not rawscr: scr = self.path_check(scr, type='npy')
        with torch.no_grad():
            ret = self.exec('decode', scr, None, rawscr=rawscr)
        return ret

    @staticmethod
    def apply_line(img, line, thresh=128):
        return np.where(line<thresh, np.minimum(line, img), img)

    def img2map_batch(self,
        img: Union[List[np.ndarray], np.ndarray],
        line: Union[List[np.ndarray], np.ndarray]
    ) -> List[np.ndarray] :
        if not (isinstance(img, np.ndarray) and isinstance(line, np.ndarray)):
            imgshapes = [i.shape[:2] for i in img]
            lineshapes = [l.shape[:2] for l in line]
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
            scrshapes = [s.shape[-2:] for s in scr]
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
