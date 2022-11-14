import os
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim.lr_scheduler as lr_scheduler


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(
        self,
        isTrain=False,
        continue_train=False,
        epoch='latest',
        checkpoints_dir='.checkpoints',
        verbose=False,
    ):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call  `BaseModel.__init__(self, opt)`
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.isTrain = isTrain
        self.continue_train = continue_train
        self.epoch = epoch
        self.verbose = verbose
        self.iter = 0
        self.last_iter = 0
        # save all the checkpoints to save_dir
        self.save_dir = os.path.join(checkpoints_dir, self.name)
        try:
            os.mkdir(self.save_dir)
        except:
            pass
        # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.sched_opts = []

        self.label_colours = np.random.randint(255, size=(100,3))

    def save_suppixel(self,l_inds):
        im_target_rgb = np.array([self.label_colours[ c % 100 ] for c in l_inds])
        b,h,w = l_inds.shape
        im_target_rgb = im_target_rgb.reshape(b,h,w,3).transpose(0,3,1,2)/127.5-1.0
        return torch.from_numpy(im_target_rgb)

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def is_train(self):
        """check if the current batch is good for training."""
        return True

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @staticmethod
    def get_scheduler(optimizer, lr_policy, opt):
        """Return a learning rate scheduler
        Parameters:
            optimizer          -- the optimizer of the network
            opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                                opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
        For 'linear', we keep the same learning rate for the first <opt.niter> epochs
        and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
        """
        if lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt['epoch_count'] - opt['niter']) / float(opt['niter_decay'] + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt['lr_decay_iters'], gamma=0.1)
        elif lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['niter'], eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt['lr_policy'])
        return scheduler

    def setup(self):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer, opt) 
                for optimizer, opt in zip(self.optimizers, self.sched_opts)]
        if not self.isTrain or self.continue_train:
            self.load_networks(self.epoch)
        self.print_networks(self.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                # print(save_path)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.state_dict(), save_path)
                    # net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

        save_filename = '%s_net_opt.pth' % (epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {'iter': str(self.iter // self.print_freq * self.print_freq)}
        for i, name in enumerate(self.optimizer_names):
            save_dict.update({name.lower(): self.optimizers[i].state_dict()})
        torch.save(save_dict, save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                # if isinstance(net, torch.nn.DataParallel):
                if isinstance(net, DDP):
                    net = net.module
                # print(net)
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # need to copy keys here because we mutate in loop
                #for key in list(state_dict.keys()):
                #    self.__patch_instance_norm_state_dict(
                #        state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)
                del state_dict

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' %
                      (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)