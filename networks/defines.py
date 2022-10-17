from .network_base import *
from .network_helpers import *

def define_G(
    input_nc,
    output_nc,
    nz,
    ngf,
    netG='unet_128',
    norm='layer',
    nl='lrelu',
    use_noise=False,
    level=0,
    use_dropout=False,
    init_type='xavier',
    init_gain=0.02,
    device=None,
    where_add='input',
    upsample='bilinear'
):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if nz == 0:
        where_add = 'input'
    if netG == 'unet_128' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_noise=use_noise,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_128_G' and where_add == 'input':
        net = G_Unet_add_input_G(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_noise=use_noise,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_noise=use_noise,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_256_G' and where_add == 'input':
        net = G_Unet_add_input_G(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_noise=use_noise,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_128' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_noise=use_noise,
                             use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_noise=use_noise,
                             use_dropout=use_dropout, upsample=upsample)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, device)


def define_C(
    input_nc,
    output_nc,
    ngf,
    netC='unet_128',
    norm='instance',
    nl='relu',
    use_dropout=False,
    init_type='normal',
    init_gain=0.02,
    device=None,
    upsample='basic'
):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if netC == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netC == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, device)


def define_E(
    input_nc,
    output_nc,
    ndf,
    netE,
    norm='batch',
    nl='lrelu',
    init_type='xavier',
    init_gain=0.02,
    device=None,
    vaeLike=False
):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if netE == 'resnet_128':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_128':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_256':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, device, False)