import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import functools
import numpy as np

from .network_helpers import *


class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels, use_wscale, nl_layer):
        super(ApplyStyle, self).__init__()
        modules = [nn.Linear(latent_size, channels*2)]
        if nl_layer:
            modules += [nl_layer()]
        self.linear = nn.Sequential(*modules)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class LayerEpilogue(nn.Module):
    def __init__(self, channels, dlatent_size, use_wscale, use_noise,
                 use_pixel_norm, use_instance_norm, use_styles, nl_layer=None):
        super(LayerEpilogue, self).__init__()
        self.use_noise = use_noise
        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale, nl_layer=nl_layer)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):
        # if noise is not None:
        if self.use_noise:
            x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)

        return x


class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, noise=None, outermost=False, innermost=False, 
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='replicate'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=3, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        uprelu2 = nl_layer()
        uppad = nn.ReplicationPad2d(1)
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None
        upnorm2 = norm_layer(outer_nc) if norm_layer is not None else None
        self.noiseblock = ApplyNoise(outer_nc)
        self.noise = noise

        if outermost:
            upconv = upsampleLayer(inner_nc * 2, inner_nc, upsample=upsample, padding_type=padding_type)
            uppad = nn.ReplicationPad2d(3)
            upconv2 = nn.Conv2d(inner_nc, outer_nc, kernel_size=7, padding=0)
            # upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            # upconv2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p)
            down = downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [norm_layer(inner_nc)]
                # up += [norm_layer(outer_nc)]
            up +=[uprelu2, uppad, upconv2] #+ [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]
            model = down + up
        else:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x2 = self.model(x)
            if self.noise:
                x2 = self.noiseblock(x2, self.noise)
            return torch.cat([x2, x], 1)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock_A(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, noise=None, outermost=False, innermost=False, 
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='replicate'):
        super(UnetBlock_A, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        downconv += [spectral_norm(nn.Conv2d(input_nc, inner_nc,
                               kernel_size=3, stride=2, padding=p))]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        uprelu2 = nl_layer()
        uppad = nn.ReplicationPad2d(1)
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None
        upnorm2 = norm_layer(outer_nc) if norm_layer is not None else None
        self.noiseblock = ApplyNoise(outer_nc)
        self.noise = noise

        if outermost:
            upconv = upsampleLayer(inner_nc * 1, outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = spectral_norm(nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p))
            down = downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up +=[uprelu2, uppad, upconv2] #+ [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = spectral_norm(nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p))
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]
            model = down + up
        else:
            upconv = upsampleLayer(inner_nc * 1, outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = spectral_norm(nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p))
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x2 = self.model(x)
            if self.noise:
                x2 = self.noiseblock(x2, self.noise)
            if x2.shape[-1]==x.shape[-1]:
                return x2 + x
            else:
                x2 = F.interpolate(x2, x.shape[2:])
                return x2 + x


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0, 
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=None, nl_layer=None, use_dropout=False, 
                 upsample='basic', padding_type='replicate'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz

        # input_nc = input_nc + nz
        downconv += [spectral_norm(nn.Conv2d(input_nc, inner_nc,
                               kernel_size=3, stride=2, padding=p))]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        uprelu2 = nl_layer()
        uppad = nn.ReplicationPad2d(1)
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None
        upnorm2 = norm_layer(outer_nc) if norm_layer is not None else None

        use_styles=False
        uprelu = nl_layer()
        if self.nz >0:
            use_styles=True

        if outermost:
            self.adaIn = LayerEpilogue(inner_nc, self.nz, use_wscale=True, use_noise=False,
                                        use_pixel_norm=True, use_instance_norm=True, use_styles=use_styles, nl_layer=nl_layer)
            upconv = upsampleLayer(
                inner_nc , outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = spectral_norm(nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p))
            down = downconv
            up = [uprelu] + upconv 
            if upnorm is not None:
                up += [upnorm]
            up +=[uprelu2, uppad, upconv2] #+ [nn.Tanh()]
        elif innermost:
            self.adaIn = LayerEpilogue(inner_nc, self.nz, use_wscale=True, use_noise=True,
                                        use_pixel_norm=True, use_instance_norm=True, use_styles=use_styles, nl_layer=nl_layer)
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = spectral_norm(nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p))
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]
        else:
            self.adaIn = LayerEpilogue(inner_nc, self.nz, use_wscale=True, use_noise=False,
                                        use_pixel_norm=True, use_instance_norm=True, use_styles=use_styles, nl_layer=nl_layer)
            upconv = upsampleLayer(
                inner_nc , outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = spectral_norm(nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p))
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)


    def forward(self, x, z, noise):
        if self.outermost:
            x1 = self.down(x)
            x2 = self.submodule(x1, z[:,2:], noise[2:])
            return self.up(x2)

        elif self.innermost:
            x1 = self.down(x)
            x_and_z = self.adaIn(x1, noise[0], z[:,0])
            x2 = self.up(x_and_z)
            x2 = F.interpolate(x2, x.shape[2:])
            return x2 + x

        else:
            x1 = self.down(x)
            x2 = self.submodule(x1, z[:,2:], noise[2:])
            x_and_z = self.adaIn(x2, noise[0], z[:,0])
            return self.up(x_and_z) + x


class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=512,
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                 gain=2**(0.5),            # original gain in tensorflow.
                 nl_layer=None
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        func = [
            nn.Linear(self.mapping_fmaps, dlatent_size)
        ]
        if nl_layer:
            func += [nl_layer()]

        for j in range(0,4):
            func += [
                nn.Linear(dlatent_size, dlatent_size)
            ]
            if nl_layer:
                func += [nl_layer()]

        self.func = nn.Sequential(*func)
            #FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            #FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers


class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64, 
                 norm_layer=None, nl_layer=None, use_dropout=False, use_noise=False,
                 upsample='basic', device=0):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        max_nchn = 8
        noise = []
        for i in range(num_downs+1):
            if use_noise:
                noise.append(True)
            else:
                noise.append(False)

        # construct unet structure
        #print(num_downs)
        unet_block = UnetBlock_A(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, noise=noise[num_downs-1], 
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock_A(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block, noise[num_downs-i-3],
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_A(ngf * 4, ngf * 4, ngf * max_nchn, unet_block, noise[2],
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_A(ngf * 2, ngf * 2, ngf * 4, unet_block, noise[1],
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_A(ngf, ngf, ngf * 2, unet_block, noise[0],
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_A(input_nc + nz, output_nc, ngf, unet_block, None, 
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z


        return torch.tanh(self.model(x_with_z))
        # return self.model(x_with_z)


class G_Unet_add_input_G(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        nz,
        num_downs,
        ngf=64,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        use_noise=False,
        upsample='basic',
    ):
        super(G_Unet_add_input_G, self).__init__()
        self.nz = nz
        max_nchn = 8
        noise = []
        for i in range(num_downs+1):
            if use_noise:
                noise.append(True)
            else:
                noise.append(False)
        # construct unet structure
        #print(num_downs)
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, noise=False,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block, noise=False,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block, noise[2],
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block, noise[1],
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block, noise[0],
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block, None,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')

        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        # return F.tanh(self.model(x_with_z))
        return self.model(x_with_z)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64, 
                 norm_layer=None, nl_layer=None, use_dropout=False, use_noise=False, upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz
        self.mapping = G_mapping(self.nz, self.nz, 512, normalize_latents=False, lrmul=1)
        self.truncation_psi = 0
        self.truncation_cutoff = 0

        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        num_layers = int(np.log2(512)) * 2 - 2
        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to("cuda"))

        # construct unet structure
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, submodule=None, innermost=True, 
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, submodule=unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, submodule=unet_block, 
                                          norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 4, ngf * 4, ngf * 8, nz, submodule=unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 2, ngf * 2, ngf * 4, nz, submodule=unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf, ngf, ngf * 2, nz, submodule=unet_block, 
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, submodule=unet_block,
                                      outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):

        dlatents1, num_layers = self.mapping(z)
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)

        # Apply truncation trick.
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            """Linear interpolation.
               a + (b - a) * t (a = 0)
               reduce to
               b * t
            """
            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)

        return torch.tanh(self.model(x, dlatents1, self.noise_inputs))


def upsampleLayer(inplanes, outplanes, kw=1, upsample='basic', padding_type='replicate'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]#, padding_mode='replicate'
    elif upsample == 'bilinear' or upsample == 'nearest' or upsample == 'linear':
        upconv = [nn.Upsample(scale_factor=2, mode=upsample, align_corners=True),
                  #nn.ReplicationPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)]
        # p = kw//2
        # upconv = [nn.Upsample(scale_factor=2, mode=upsample, align_corners=True),
        #           nn.Conv2d(inplanes, outplanes, kernel_size=kw, stride=1, padding=p, padding_mode='replicate')]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, norm_layer=None, use_dropout=True, n_blocks=6, padding_type='replicate'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        model = [nn.ReplicationPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias)]
        if norm_layer is not None:
            model += [norm_layer(ngf)]
        model += [nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.ReplicationPad2d(1),nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias)]
            if norm_layer is not None:
                model += [norm_layer(ngf * mult * 2)]
            model += [nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            #                              kernel_size=3, stride=2,
            #                              padding=1, output_padding=1,
            #                              bias=use_bias)]
            # if norm_layer is not None:
            #     model += [norm_layer(ngf * mult / 2)]
            # model += [nn.ReLU(True)]
            model += upsampleLayer(ngf * mult, int(ngf * mult / 2), upsample='bilinear', padding_type=padding_type)
            if norm_layer is not None:
                model += [norm_layer(int(ngf * mult / 2))]
            model += [nn.ReLU(True)]
            model +=[nn.ReplicationPad2d(1),
                     nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, padding=0)]
            if norm_layer is not None:
                model += [norm_layer(ngf * mult / 2)]
            model += [nn.ReLU(True)]
        model += [nn.ReplicationPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]
        conv_block += [nn.ReLU(True)]
        # if use_dropout:
        #     conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AdaptiveAvgPool2d(4)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf * 16, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf * 16, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf * 16, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(BasicBlock, self).__init__()
        layers = []
        norm_layer=get_norm_layer(norm_type='layer') #functools.partial(LayerNorm)
        # norm_layer = None
        nl_layer=nn.ReLU()
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer]
        layers += [nn.ReplicationPad2d(1),
                   nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1,
                     padding=0, bias=True)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 3, 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw, padding_mode='replicate')), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=2, padding=padw, padding_mode='replicate'))]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AdaptiveAvgPool2d(4)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[spectral_norm(nn.Linear(ndf * nf_mult * 16, output_nc))])
        if vaeLike:
            self.fcVar = nn.Sequential(*[spectral_norm(nn.Linear(ndf * nf_mult * 16, output_nc))])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output