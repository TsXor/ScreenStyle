import torch
import torch.nn.functional as F
from models.screenvae import ScreenVAE

import numpy as np
from sklearn.decomposition import PCA

import lib.sanitize as sanitize

class ScreenVAE_rec:
    def __init__(self, model_name='ScreenVAE', freeze_seed=None):
        self.freeze_seed = freeze_seed
        if not (self.freeze_seed is None):
            torch.manual_seed(self.freeze_seed)
            torch.cuda.manual_seed(self.freeze_seed)
            np.random.seed(self.freeze_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScreenVAE(inc=1, outc=4, blocks=3, save_dir='checkpoints/%s'%model_name, device=device)

    def eval(self, mode, *input):
        if mode=='encode':
            img, line = input
            img = sanitize.PILconvert(img, 'L')
            line = sanitize.PILconvert(line, 'L')
            sizey, sizex = img.shape
            img = sanitize.modpad(img, (16,16))
            line = sanitize.modpad(line, (16,16))
            img_torch = torch.from_numpy(img[np.newaxis,np.newaxis,...]).float()
            line_torch = torch.from_numpy(line[np.newaxis,np.newaxis,...]).float()
            img_torch = sanitize.squash(img_torch)
            line_torch = sanitize.squash(line_torch)
            img_torch = sanitize.todev(img_torch)
            line_torch = sanitize.todev(line_torch)
            if not (self.freeze_seed is None):
                torch.manual_seed(self.freeze_seed)
                torch.cuda.manual_seed(self.freeze_seed)
                np.random.seed(self.freeze_seed)
            ret_torch = self.model(mode, img_torch, line_torch)
            ret_torch = ret_torch.detach()
            scr_torch = ret_torch*(line_torch+1)/2
            scr_torch = scr_torch.cpu()
            scr = scr_torch.numpy()[0]
            scr = scr[:, 0:sizey, 0:sizex]
            return scr
        elif mode=='decode':
            scr = input[0]
            sizez, sizey, sizex = scr.shape
            scr = sanitize.modpad(scr, (0,128,128))
            scr_torch = torch.from_numpy(scr)
            scr_torch = scr_torch.unsqueeze(0).float()
            if not (self.freeze_seed is None):
                torch.manual_seed(self.freeze_seed)
                torch.cuda.manual_seed(self.freeze_seed)
                np.random.seed(self.freeze_seed)
            recons_torch = self.model(mode, sanitize.todev(scr_torch))
            recons_torch = sanitize.unsquash(recons_torch)
            recons_torch = recons_torch.cpu().detach()
            recons = recons_torch.numpy()[0,0]
            recons = recons[0:sizey, 0:sizex]
            recons = sanitize.any2pic(recons, method='clip')
            return recons
        else:
            return

    def img2map(self, img, line):
        return self.eval('encode', img, line)

    def map2img(self, scr):
        return self.eval('decode', scr)

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
