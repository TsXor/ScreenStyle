import numpy as np
import torch
from PIL import Image

def PILopen(path, astype='L'):
    return np.asarray(Image.open(str(path)).convert(astype))

PILread = PILopen

def PILsave(img, path):
    Image.fromarray(img).save(str(path))

def PILconvert(img, mode):
    return np.asarray(Image.fromarray(img).convert(mode))

def PILshow(img):
    Image.fromarray(img).show()

def modpad(img, mods):
    pads = tuple(mod-img%mod if mod else 0 for img, mod in zip(img.shape, mods))
    padded = np.pad(img, tuple((0, pad) for pad in pads), 'linear_ramp')
    return padded

def todev(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor.cpu()

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