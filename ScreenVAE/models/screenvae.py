from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import functools
import os
# from .SuperPixelPool.suppixpool_layer import AveSupPixPool,SupPixUnpool
import random
from . import networks as networks

from ..lib import sanitize as sanitize

class ScreenVAE(nn.Module):
    def __init__(
        self,
        inc=1,
        outc=4,
        outplanes=64,
        downs=5,
        blocks=2,
        load_ext=True,
        save_dir='checkpoints/ScreenVAE',
        init_type="normal",
        init_gain=0.02,
        device=None
    ):
        super(ScreenVAE, self).__init__()
        self.inc = inc
        self.outc = outc
        self.save_dir = save_dir
        self.model_names=['enc','dec']
        self.enc=networks.define_C(
            inc+1,
            outc*2,
            24,
            netC='resnet_6blocks',
            norm='layer',
            nl='lrelu',
            use_dropout=True, 
            device=device,
            upsample='bilinear'
        )
        self.dec=networks.define_G(
            outc,
            inc,
            48,
            netG='unet_128_G', 
            norm='layer',
            nl='lrelu',
            use_dropout=True,
            device=device,
            where_add='input',
            upsample='bilinear',
            use_noise=True
        )
        self.load_networks('latest')
        for param in self.parameters():
            param.requires_grad = False

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(
                    load_path, map_location=lambda storage, loc: sanitize.todev(storage))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)
                del state_dict

    def load_gaborext(self, device=None):
        self.gaborext = GaborWavelet()
        self.gaborext.eval()
        self.gaborext.to(device)

    def npad(self, im, pad=128):
        h,w = im.shape[-2:]
        hp = h //pad*pad+pad
        wp = w //pad*pad+pad
        return F.pad(im, (0, wp-w, 0, hp-h), mode='constant',value=1)

    def forward(self, mode, *input):
        if mode=='encode':
            img, line = input
            line = torch.sign(line)
            img = torch.clamp(img + (1-line), -1, 1)
            inter = self.enc(torch.cat([img, line], 1))
            scr, logvar = torch.split(inter, (self.outc, self.outc), dim=1)#[:,:,32:-32,32:-32]
            # scr = scr*torch.sign(line+1)
            return scr
        elif mode=='decode':
            smap = input[0]
            recons = self.dec(smap)
            #recons = (recons+1)*(line+1)/2-1
            #recons = torch.clamp(recons,-1,1) # 理论上上一行得到的结果已在[-1, 1]范围内，这行只是保险
            # 默认line = torch.ones_like(smap)，因此上两行其实都是不必要的（铸币吧这！）
            return recons