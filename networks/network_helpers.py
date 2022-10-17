import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.modules.normalization as t_n; LayerNorm = t_n.LayerNorm
import functools

import pathlib, sys
PROJECT_ROOT = pathlib.Path(__file__).parent / '..'
sys.path.append(str(PROJECT_ROOT))
import sanitize

###############################################################################
# Helper functions
###############################################################################

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                #init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, device=None, init=True):
    """Initialize a network: initialize the network weights
    Parameters:
        net (network)         -- the network to be initialized
        init_type (str)       -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)          -- scaling factor for normal, xavier and orthogonal.
        device (torch.device) -- a torch device
    Return an initialized network.
    """
    net.to(device)
    if init:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(LayerNormWarpper)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class LayerNormWarpper(nn.Module):
    def __init__(self, num_features):
        super(LayerNormWarpper, self).__init__()
        self.num_features = int(num_features)

    def forward(self, x):
        return (nn.LayerNorm([self.num_features, *x.size()[2:3+1]], elementwise_affine=False))(x)


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    elif layer_type == 'selu':
        nl_layer = functools.partial(nn.SELU, inplace=True)
    elif layer_type == 'prelu':
        nl_layer = functools.partial(nn.PReLU)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.randn(channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, x, noise):
        W,_ = torch.split(self.weight.view(1, -1, 1, 1), self.channels // 2, dim=1)
        B,_ = torch.split(self.bias.view(1, -1, 1, 1), self.channels // 2, dim=1)
        Z = torch.zeros_like(W)
        w = torch.cat([W,Z], dim=1).to(x.device)
        b = torch.cat([B,Z], dim=1).to(x.device)
        adds = w * torch.randn_like(x) + b
        return x + adds.type_as(x)