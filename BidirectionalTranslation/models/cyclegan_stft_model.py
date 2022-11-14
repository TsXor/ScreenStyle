import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).parent / '..' / '..'
sys.path.append(str(PROJECT_ROOT))

import torch
from .base_model import BaseModel

import networks, sanitize
from ScreenVAE import SVAE


def norm(im):
    return im * 2.0 - 1

def denorm(im):
    return (im + 1) / 2.0

def grayscale(input_image):
    rate = torch.Tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).to(input_image.device)
    return (input_image*rate).sum(1,keepdims=True)

padtensor = lambda tensor: sanitize.modpad_tensor(tensor, (2**8, 2**8), 'constant', value=1)


class CycleGANSTFTModel(BaseModel):

    def __init__(
        self,
        output_nc=3,
        nz=8,
        nef=64,
        ngf=64,
        init_gain=0.02,
        checkpoints_dir='.checkpoints',
        name='BidirectionalTranslation',
        device=None,
    ):
        self.name = name

        use_vae = True
        use_dropout = False
        self.interchnnls = 4
        use_noise = False

        self.netG_INTSCR2RGB = networks.define_G(
            self.interchnnls + 1,
            3,
            nz,
            ngf,
            netG='unet_256', 
            norm='layer',
            nl='lrelu',
            use_dropout=use_dropout,
            init_type='kaiming',
            init_gain=init_gain,
            device=device,
            where_add='all',
            upsample='bilinear',
            use_noise=use_noise
        )
        self.netG_RGB2INTSCR = networks.define_G(
            4,
            self.interchnnls,
            0,
            ngf,
            netG='unet_256',
            norm='layer',
            nl='lrelu',
            use_dropout=use_dropout,
            init_type='kaiming',
            init_gain=init_gain,
            device=device,
            where_add='input',
            upsample='bilinear',
            use_noise=use_noise
        )        
        self.netE = networks.define_E(
            output_nc,
            nz,
            nef,
            netE='conv_256',
            norm='none',
            nl='lrelu',
            init_type='xavier',
            init_gain=init_gain,
            device=device,
            vaeLike=use_vae
        )

        super().__init__(
            checkpoints_dir=checkpoints_dir
        )

        self.device = device

        self.model_names = ['G_INTSCR2RGB','G_RGB2INTSCR','E']
        self.nets = [self.netG_INTSCR2RGB, self.netG_RGB2INTSCR, self.netE]

        self.load_networks('latest')

    def forward(self, mode, img, line, styref=None):
        # modes: AtoB == color2tone
        #        BtoA == tone2color
        # image has to be RGB
        # styref: RGB image for style reference
        if mode=='AtoB':
            h, w = img.shape[-2:]
            img = padtensor(img); line = padtensor(line)
            img_and_line = torch.cat([line, img],1)
            scrmap = self.netG_RGB2INTSCR(img_and_line)
            return scrmap[:, :, :h, :w]
        elif mode=='BtoA':
            h, w = img.shape[-2:]
            img = padtensor(img); line = padtensor(line)
            styref = self.netE(torch.zeros(1))[0] if styref is None else styref
            scrmap_and_line = torch.cat([line, img], 1) # img is scrmap
            colorimg = self.netG_INTSCR2RGB(scrmap_and_line, styref)
            colorimg = torch.clamp(colorimg, -1, 1)
            colorimg = norm(denorm(colorimg) * denorm(line))
            return colorimg[:, :, :h, :w]
        else:
            raise NotImplementedError

    def get_z_random(self, batch_size, nz, random_type='gauss', truncation=False, tvalue=1):
        z = None
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz) * tvalue
            # do the truncation trick
            if truncation:
                for _k in range(15 * nz):
                    z = torch.where(torch.abs(z) > tvalue, torch.randn(batch_size, nz), z)
                z = torch.clamp(z, -tvalue, tvalue)

        return z.detach().to(self.device)

    def optimize_parameters(self):
        pass