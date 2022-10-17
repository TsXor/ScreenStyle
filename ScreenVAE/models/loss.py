import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).parent / '..' / '..'
sys.path.append(str(PROJECT_ROOT))
import networks


class reconstruction_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Im, Rm):
        SE = F.mse_loss(Im, Rm, reduction = "none")
        SSE = SE.sum((1, 2, 3)) # sum dim C, H, W
        L2_norm = SSE**(1/2)
        return L2_norm.mean()


class superpixel_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def avgify(self, Is, Ispp):
        spp_avg = torch.zeros_like(Is)
        for i in range(torch.max(Ispp)):
            labelv = i+1
            labelbool = (Ispp==labelv)
            labelbool = labelbool.repeat_interleave(4, dim=1)
            part = torch.where(labelbool, Is, 0)
            avgs = torch.sum(part, dim=(-2, -1)) / torch.sum(labelbool, dim=(-2, -1))
            avgs = avgs.unsqueeze(-1).unsqueeze(-1)
            spp_avg = torch.where(labelbool, avgs, spp_avg)
        return spp_avg

    def forward(self, Is, Ispp, Il, line_thresh=255/4):
        wl = torch.where(Il>line_thresh, 1, 0)
        spp_average = self.avgify(Is, Ispp)
        norm = ((Is-spp_average)**2).sum(dim=1)
        return (wl*norm).mean()


class KL_regularization_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Is, logvar):
        # logvar is the second thing returned by model.forward(mode='encode')
        KLD_loss = torch.sum(Is**2 + logvar.exp()-logvar-1)/2
        return KLD_loss


nl_layer = networks.helpers.get_non_linearity(layer_type='lrelu')

class Discriminator(nn.Module):
    def __init__(self, ndf=64, device=None):
        super().__init__()
        n_layers, input_nc, kw, padw = 4, 1, 3, 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                    stride=2, padding=padw, padding_mode='replicate')), nl_layer()]
        nf_mults = ([1, 2, 4] + [8]*(n_layers-3))[:n_layers] # [1, 2, 4, 8, 8, 8, ...]
        nf_mults = zip(nf_mults[:-1],nf_mults[1:]) # [(1, 2), (2, 4), (4, 8), (8, 8), ...]
        sequence_more = [[spectral_norm(nn.Conv2d(ndf * mult1, ndf * mult2, kernel_size=kw,
                          stride=2, padding=padw, padding_mode='replicate')), nl_layer()] \
                         for mult1, mult2 in nf_mults]
        sequence_more = sum(sequence_more, [])
        sequence += sequence_more
        sequence += [nn.AdaptiveAvgPool2d(4)]
        self.conv = nn.Sequential(*sequence)
        self.conv = self.conv.to(device)
    def forward(self, input):
        conved = self.conv(input)
        return conved


class adversarial_loss(nn.Module):
    # https://gitee.com/taoxianpeng/wgan-gp/blob/master/training.py
    def __init__(self, device=None):
        super().__init__()
        self.D = Discriminator(device=device)
        self.gp_weight = 1
        self.device = device

    def Dsr(self, input):
        return self.D(input).mean((1, 2, 3))

    def _gradient_penalty(self, Im, Rm):
        batch_size = Im.shape[0]
        epsilon = 1e-12
        
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(Im)
        alpha = alpha.to(self.device)
        interpolated = alpha * Im.data + (1 - alpha) * Rm.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.device)
        D_interpolated = self.D(interpolated)
        
        gradients = torch_grad(
            outputs=D_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(D_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients_norm = torch.sqrt((gradients ** 2).sum((1, 2, 3)) + epsilon)
        
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def forward(self, Im, Rm):
        loss = (self.Dsr(Im) - self.Dsr(Rm)).mean()
        gradient_penalty = self._gradient_penalty(Im, Rm)
        return loss + gradient_penalty

class SVAE_loss(nn.Module):
    lambda_rec = 5
    lambda_spp = 20
    lambda_z = 1
    lambda_adv = 1

    def __init__(self, device=None):
        super().__init__()
        self.loss_rec = reconstruction_loss().to(device)
        self.loss_spp = superpixel_loss().to(device)
        self.loss_z = KL_regularization_loss().to(device)
        self.loss_adv = adversarial_loss(device=device).to(device)

    def forward(self, Im, Is, logvar, Rm, Il, Ispp, line_thresh=255/4):
        loss = self.lambda_rec * self.loss_rec(Im, Rm) \
             + self.lambda_spp * self.loss_spp(Is, Ispp, Il, line_thresh=line_thresh) \
             + self.lambda_z   * self.loss_z(Is, logvar) \
             + self.lambda_adv * self.loss_adv(Im, Rm)
        return loss