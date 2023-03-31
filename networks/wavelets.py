import torch
import numpy as np
import matplotlib.pyplot as plt

class WaveletLayer(torch.nn.Module):

    def __init__(self,
                 init_scale = 1.,
                 init_center = 0.,):
        super().__init__()

        self.scale = torch.nn.Parameter(torch.tensor(init_scale))
        self.center = torch.nn.Parameter(torch.tensor(init_center))

        self.f_i = 1 / torch.sqrt(torch.log(torch.tensor(2)))

    def forward(self, x):
        x = x - self.center
        return torch.cos(2 * torch.pi * self.f_i * x) * torch.exp(-x**2 / (2 * self.scale**2))

#TODO : think about how to execute this in a decoder
#TODO : should wavelet layer use an array of scales and centers?
#TODO : maybe the input to forward should be a scale and adds to the residual
#TODO : wavelets maybe should be seperate from the general x
if __name__ == "__main__":
    n_points = 1000
    #x = torch.linspace(-10, 10, n_points)
    x = torch.randn((8, n_points))

    waver = WaveletLayer()

    morlet = waver(x).detach().numpy()

    plt.plot(morlet)
