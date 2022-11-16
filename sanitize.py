import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def PILopen(path, astype=None):
    opened = np.asarray(Image.open(str(path))) \
             if astype is None else \
             np.asarray(Image.open(str(path)).convert(astype))
    return opened

PILread = PILopen

def PILsave(img, path):
    Image.fromarray(img).save(str(path))

def PILconvert(img, mode):
    return np.asarray(Image.fromarray(img).convert(mode))

def togray(arr, batch=False):
    naxis = len(arr.shape)
    if batch: naxis -= 1
    if naxis == 2:
        return arr
    elif naxis == 3: # input image should be RGB
        return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114

def PILshow(img):
    Image.fromarray(img).show()

def modpad(img, mods, mode='linear_ramp', **kwargs):
    pads = tuple(mod-left if (left := s%mod) else 0 for s, mod in zip(img.shape, mods))
    padded = np.pad(img, [(0, pad) for pad in pads], mode, **kwargs)
    return padded

def modpad_img(img, mods, mode='linear_ramp', **kwargs):
    padded = modpad(img, (*mods, (0, 0)), mode, **kwargs) \
             if img.shape[2:]==(3,) else \
             modpad(img, mods, mode, **kwargs)
    return padded

def modpad_tensor(tensor, mods, mode, **kwargs):
    H, W = tensor.shape[-2:]
    modH, modW = mods
    leftH = H % modH
    padH = modH - leftH if leftH else 0
    leftW = W % modW
    padW = modW - leftW if leftW else 0
    pads = (0, padW, 0, padH)
    padded = F.pad(tensor, pads, mode, **kwargs)
    return padded

def squash(arr):
    # [0, 255] -> [-1, 1]
    return arr*2/255-1

def unsquash(arr):
    return (arr+1)*255/2

def any2pic(arr, method='squash'):
    if method=='squash':
        arr = arr-arr.min()
        arr = arr*255/arr.max()
    elif method=='clip':
        arr = np.clip(arr, 0, 255)
    arr = arr.astype(np.uint8)
    return arr

def gray2tensor(img):
    img = img[np.newaxis,...] if img.ndim==2 else img
    img = img[:,np.newaxis,:,:]
    img_torch = torch.from_numpy(img.copy()).float()
    img_torch = squash(img_torch)
    return img_torch

def rgb2tensor(img):
    img = img[np.newaxis,...] if img.ndim==3 else img
    img = img.transpose((0, 3, 1, 2))
    img_torch = torch.from_numpy(img.copy()).float()
    img_torch = squash(img_torch)
    return img_torch