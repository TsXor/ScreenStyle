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
    imgy, imgx = img.shape
    mody, modx = mods
    pady = mody-imgy%mody; padx = modx-imgx%modx
    padded = np.pad(img, ((0, pady), (0, padx)), 'linear_ramp')
    return padded

def todev(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor.cpu()