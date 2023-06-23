import torch
import einops
import numpy as np

#TODO : this class is unnecessary, keeping in case it's useful later
class Pairer:
    """Simple class using Cantor's pairing function or raster order to go from 2D to 1D and back.
    See https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.n = width * height

    def cantor_pair(self, x, y):
        assert x < self.width and y < self.height, "indices out of range"
        return int(((x + y) * (x + y + 1) / 2) + y)
    
    def cantor_unpair(self, z):
        assert z < self.n, "index is out of range"
        w = int(np.floor((np.sqrt(8 * z + 1) - 1) / 2))
        t = (w * w + w) / 2
        y = int(z - t)
        x = int(w - y)
        return x, y
    
    def raster_pair(self, x, y):
        assert x < self.width and y < self.height, "indices out of range"
        return int(x + y * self.width)
    
    def raster_unpair(self, z):
        assert z < self.n, "index is out of range"
        x = z % self.width
        y = z // self.width
        return x, y
    

class SOMGrid(torch.nn.Module):

    def __init__(self, 
                 height,
                 width,
                 neighbor_distance = 1,
                 kernel_type = "gaussian",
                 time_constant = 0.1):
        self.height = height
        self.width = width
        self.size = width * height

        self.t = 0
        self.time_constant = time_constant

        self.kernel_type = kernel_type
        self.neighbor_distance = neighbor_distance
        if neighbor_distance != 0:
            self.kernel_size = 2 * neighbor_distance + 1
            self.padding = neighbor_distance 
        else: # use the whole grid
            self.kernel_size = (2 * height + 1, 2 * width + 1)
            self.padding = (height, width)

        # layers to convert in and out of codebook
        self.codebook_to_grid = einops.layers.Rearrange("b dim (h w) -> b dim h w", h = height, w = width)
        self.grid_to_codebook = einops.layers.Rearrange("b dim h w -> b dim (h w)", h = height, w = width)

        if kernel_type == "gaussian":
            kernel_size_max = max(self.kernel_size)
            range_ = kernel_size_max // 2
            kernel_init = torch.exp(-torch.arange(-range_, range_ + 1)**2)

            # use outer product to get 2d kernel
            kernel_init = torch.outer(kernel_init, kernel_init)

        elif kernel_type == "hard":
            kernel_init = torch.ones(self.kernel_size)
        else:
            raise ValueError("kernel_type must be gaussian or hard")
        
        self.register_buffer("kernel_init", kernel_init)
        self.register_buffer("kernel", kernel_init)

    def update_t(self):
        self.t += 1
        t_scalar = 1 / (1 + self.t * self.time_constant)
        if self.kernel_type == "gaussian":
            self.kernel = (self.kernel_init).pow(torch.exp(t_scalar**2))
        else:
            self.kernel = self.kernel_init * t_scalar


    def forward(self, cb_onehot, update_t = False):
        _, dim, _ = cb_onehot.shape
        cb_reshaped = self.codebook_to_grid(cb_onehot)
        kernel = self.kernel[None, None, ...].repeat(dim, dim, 1, 1)
        cb_blurred = torch.nn.functional.conv2d(cb_reshaped, kernel, padding = self.padding)
        new_cb = self.grid_to_codebook(cb_blurred)

        if update_t:
            self.update_t()
        
        return new_cb

