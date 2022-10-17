import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pathlib

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

def PILshow(img):
    Image.fromarray(img).show()

def modpad(img, mods, mode='linear_ramp', **kwargs):
    pads = tuple(mod-s%mod if mod else 0 for s, mod in zip(img.shape, mods))
    padded = np.pad(img, tuple((0, pad) for pad in pads), mode, **kwargs)
    return padded

def img_modpad(img, mods, mode='linear_ramp', **kwargs):
    padded = modpad(img, (*mods, (0, 0)), mode, **kwargs) \
             if img.shape[2:]==(3,) else \
             modpad(img, mods, mode, **kwargs)
    return padded

def modpad_tensor(tensor, mods, dims=2):
    pads = tuple(mod-s%mod if mod else 0 for s, mod in zip(tensor.shape, mods))
    pads = [[p, 0] for p in pads]
    pads = sum(pads, []); pads.reverse()
    pads = pads[:2*dims]
    padded = F.pad(tensor, pads, mode='replicate')
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

tfunc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
tfunc_gray = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])

def img_normalize(img):
    notgray = img.shape[2:]==(3,)
    return tfunc(img) if notgray else tfunc_gray(img)