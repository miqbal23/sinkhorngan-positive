# ResNet generator and discriminator
import numpy as np
from torch import nn

from base import BaseNetwork


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
        )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(
                in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(
            in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(BaseNetwork):
    def __init__(self, ngf, z_dim, input_size: tuple = (3, 32, 32), **kwargs):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        self.n_channels = input_size[0]

        self.dense = nn.Linear(self.z_dim, 4 * 4 * self.ngf)
        self.final = nn.Conv2d(
            self.ngf, self.n_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(self.ngf, self.ngf, stride=2),
            ResBlockGenerator(self.ngf, self.ngf, stride=2),
            ResBlockGenerator(self.ngf, self.ngf, stride=2),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.ngf, 4, 4))


class Critic(BaseNetwork):
    def __init__(self, ndf, critic_dim, input_size: tuple = (3, 32, 32), **kwargs):
        super(Critic, self).__init__()
        self.ndf = ndf
        self.n_channels = input_size[0]
        self.critic_dim = critic_dim

        self.model = nn.Sequential(
            FirstResBlockDiscriminator(
                self.n_channels, self.ndf, stride=2),
            ResBlockDiscriminator(self.ndf, self.ndf, stride=2),
            ResBlockDiscriminator(self.ndf, self.ndf),
            ResBlockDiscriminator(self.ndf, self.ndf),
            nn.ReLU(),
            nn.AvgPool2d(8),
        )
        self.fc = nn.Linear(self.ndf, self.critic_dim)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = self.fc

    def forward(self, input_data):
        return self.fc(self.model(input_data).view(-1, self.ndf))
