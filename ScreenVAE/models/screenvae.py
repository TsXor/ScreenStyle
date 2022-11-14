import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).parent / '..' / '..'
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import os
# from .SuperPixelPool.suppixpool_layer import AveSupPixPool,SupPixUnpool

import sanitize, networks


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
            0,
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
        self.device = device
        self.load_networks('latest')

    def load_networks(self, epoch):
        for name in self.model_names:
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, name)
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)
            del state_dict

    def load_gaborext(self, device=None):
        self.gaborext = GaborWavelet()
        self.gaborext.eval()
        self.gaborext.to(device)

    def forward(self, mode, *input):
        if mode=='encode':
            img, line = input
            line = torch.sign(line)
            sizey, sizex = img.shape[-2:]
            img = sanitize.modpad_tensor(img, (16, 16), 'replicate')
            line = sanitize.modpad_tensor(line, (16, 16), 'replicate')
            img = torch.clamp(img + (1-line), -1, 1)
            inter = self.enc(torch.cat([img, line], 1))
            scr, logvar = torch.split(inter, (self.outc, self.outc), dim=1)#[:,:,32:-32,32:-32]
            # scr = scr*torch.sign(line+1)
            scr = scr[..., :sizey, :sizex]
            return scr, logvar
        elif mode=='decode':
            smap, line = input
            sizey, sizex = smap.shape[-2:]
            smap = sanitize.modpad_tensor(smap, (128, 128), 'replicate')
            recons = self.dec(smap)
            if not (line is None):
                line = sanitize.modpad_tensor(line, (128, 128), 'replicate')
                recons = (recons+1)*(line+1)/2-1
                recons = torch.clamp(recons,-1,1)
            recons = recons[..., :sizey, :sizex]
            return recons
        else:
            raise NotImplementedError