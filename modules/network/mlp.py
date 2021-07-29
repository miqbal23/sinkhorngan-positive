import numpy as np
import torch.nn as nn

from base import BaseNetwork


class Generator(BaseNetwork):
    def __init__(self, ngf, z_dim, input_size=(1, 28, 28), **kwargs):
        super(Generator, self).__init__()

        self.ngf = ngf
        self.z_dim = z_dim
        self.input_size = input_size

        self.main = nn.Sequential(
            # Z goes into a linear of size: self.ngf
            nn.Linear(self.z_dim, self.ngf),
            nn.ReLU(True),
            nn.Linear(self.ngf, self.ngf),
            nn.ReLU(True),
            nn.Linear(self.ngf, self.ngf),
            nn.ReLU(True),
            nn.Linear(self.ngf, np.prod(self.input_size)),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1))
        output = self.main(z)
        return output.view(output.size(0), *self.input_size)


class Critic(BaseNetwork):
    def __init__(self, ndf, z_dim, critic_dim, input_size=(1, 28, 28), **kwargs):
        super(Critic, self).__init__()

        self.ndf = ndf
        self.z_dim = z_dim
        self.input_size = input_size
        self.critic_dim = critic_dim

        self.main = nn.Sequential(
            # Z goes into a linear of size: self.ndf
            nn.Linear(int(np.prod(self.input_size)), self.ndf),
            nn.ReLU(True),
            nn.Linear(self.ndf, self.ndf),
            nn.ReLU(True),
            nn.Linear(self.ndf, self.ndf),
            nn.ReLU(True),
            nn.Linear(self.ndf, self.critic_dim),
        )

    def forward(self, input_data):
        input_data = input_data.view(
            input_data.shape[0], input_data.shape[1] * input_data.shape[2] * input_data.shape[3])
        output = self.main(input_data)
        output = output.view(input_data.shape[0], self.critic_dim)
        return output
