import torch.nn as nn

from base import BaseNetwork
from modules.regularization.network.spectral_normalization import SpectralNorm


class Generator(BaseNetwork):
    def __init__(self, ngf, z_dim, input_size: tuple = (3, 32, 32), n_extra_layers=0, **kwargs):
        super(Generator, self).__init__()

        self.ngf = ngf
        self.z_dim = z_dim
        self.image_size = input_size[1]
        self.n_channel = input_size[0]

        assert self.image_size % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = self.ngf // 2, 4
        while tisize != self.image_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial_{0}-{1}_convt'.format(self.z_dim, cngf),
                        nn.ConvTranspose2d(self.z_dim, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf), nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < self.image_size // 2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(
                                cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid_{0}_relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, self.n_channel),
                        nn.ConvTranspose2d(cngf, self.n_channel, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(self.n_channel), nn.Tanh())
        self.main = main

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, z):
        z = z.view(z.shape[0], self.z_dim, 1, 1)
        output = self.main(z)
        return output


class Critic(BaseNetwork):
    def __init__(self, ndf, z_dim, critic_dim, input_size: tuple = (3, 32, 32), n_extra_layers=0, **kwargs):
        super(Critic, self).__init__()

        self.ndf = ndf
        self.z_dim = z_dim
        self.image_size = input_size[1]
        self.n_channel = input_size[0]
        self.critic_dim = critic_dim

        assert self.image_size % 16 == 0, "self.image_size has to be a multiple of 16"

        main = nn.Sequential()
        # input is self.n_channel x self.image_size x self.image_size
        main.add_module('initial_conv_{0}-{1}'.format(self.n_channel, self.ndf),
                        SpectralNorm(nn.Conv2d(self.n_channel, self.ndf, 4, 2, 1, bias=False)))
        main.add_module('initial_relu_{0}'.format(self.ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = self.image_size / 2, self.ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cndf),
                            SpectralNorm(nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False)))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            SpectralNorm(nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final_{0}-{1}_conv'.format(cndf, self.critic_dim),
                        SpectralNorm(nn.Conv2d(cndf, self.critic_dim, 4, 1, 0, bias=False)))
        self.main = main

    def forward(self, input_data):
        output = self.main(input_data)
        output = output.view(input_data.size()[0], -1)
        return output
