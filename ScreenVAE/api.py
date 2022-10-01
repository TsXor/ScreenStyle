import torch
import torch.nn.functional as F
from models.screenvae import ScreenVAE

import numpy as np
from skimage.segmentation import flood, flood_fill
from sklearn.decomposition import PCA

import lib.sanitize as sanitize

class ScreenVAE_rec:
    def __init__(self, model_name='ScreenVAE'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScreenVAE(inc=1, outc=4, blocks=3, save_dir='checkpoints/%s'%model_name, device=device)

    def get_screenmap(self, img, line):
        img = sanitize.PILconvert(img, 'L')
        line = sanitize.PILconvert(line, 'L')
        sizey, sizex = img.shape
        img = img/255; img = sanitize.modpad(img, (16,16))
        line = line/255; line = sanitize.modpad(line, (16,16))
        
        img = torch.from_numpy(img[np.newaxis,np.newaxis,...]).float()*2-1.0
        line = torch.from_numpy(line[np.newaxis,np.newaxis,...]).float()*2-1.0
        
        scr = self.model(sanitize.todev(img), sanitize.todev(line), rep=True).cpu().detach()
        scr = scr*(line+1)/2; scr_np = scr.numpy()[0]
        scr_np = scr_np[:, 0:sizey, 0:sizex]
        return scr_np

    def get_recons(self, scr):
        scr = torch.from_numpy(scr).unsqueeze(0).float()
        outs = self.model(sanitize.todev(scr), None, screen=True)*0.5+0.5
        outs = torch.clamp(outs, 0, 1).cpu()
        outs_copy = outs.detach()
        outs_copy_npy = outs.numpy()[0,0]
        return outs_copy_npy

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

    def apply_screenmap(self, img, scr, seedpt=(10,10)):
        filled = np.ones(scr.shape[1:])
        # scr[line[np.newaxis,:,:].repeat(4,axis=0)<0.75]=-1
        nscr = scr.copy()
        for i in range(4):
            filled_img = flood(nscr[i], seedpt, tolerance=0.15)
            filled[~filled_img] = 0
            rand = np.random.randn()*0.5
            nscr[i][filled==1] = rand
        out = self.get_recons(nscr)
        img = np.where(filled==1, out*255, img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
