import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt


def causal_functional_conv1d(x, 
                             weight, 
                             bias = None, 
                             stride = 1, 
                             dilation = 1,
                             groups = 1):
    """A 1D convolution which masks future inputs."""
    kernel_size = weight.shape[-1]

    # causal pad
    pad = dilation * (kernel_size - 1) - stride + 1

    length = x.shape[-1]
    next_length = (length - kernel_size + pad) / stride + 1
    target_length = (ceil(next_length) - 1) * stride + kernel_size - pad
    extra_pad = target_length - length

    x = torch.nn.functional.pad(x, (pad, extra_pad))
    return torch.nn.functional.conv1d(x, weight, bias = bias, stride = stride, dilation = dilation, groups = groups)

def causal_functional_conv_t1d(x, weight, bias = None, stride = 1, dilation = 1):
    kernel_size = weight.shape[-1]
    #TODO : think about this more
    right_pad = dilation * (kernel_size - 1) - stride + 1

    pad = x.shape[-1] - right_pad

    x = torch.nn.functional.conv_transpose1d(x, weight, bias = bias, stride = stride, padding = pad, dilation = dilation)
    return x[..., :pad]
    
class CausalMultiresConv1d(torch.nn.Module):
    """Multi-resolution convolutions from here:
    https://arxiv.org/abs/2305.01638
    Updated to use causal convolutions.

    Parameters
    ----------
    channels : int
        Number of output channels.
    kernel_size : int
        Size of the kernel.
    depth : int
        Number of layers.
    dropout : float
        Dropout rate.
    activation : torch.nn.Module
        Activation function.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 depth,
                 dropout = 0.0,
                 activation = torch.nn.GELU()):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.dropout = dropout
        self.activation = activation

        scalar = sqrt(2.0) / (kernel_size * 2)

        # weights as described in paper
        self.h0 = torch.nn.Parameter(torch.empty(channels, 1, kernel_size).uniform_(-1., 1.) * scalar)
        self.h1 = torch.nn.Parameter(torch.empty(channels, 1, kernel_size).uniform_(-1., 1.) * scalar)
        w = torch.empty(channels, depth + 2).uniform_(-1., 1.) * sqrt(2.0 / (2 * depth + 4))
        self.w = torch.nn.Parameter(w)

        self.dropout_layer = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        residual_low = x.clone()
        y = 0
        dilation = 1
        for i in range(self.depth, 0, -1):
            residual_high = causal_functional_conv1d(residual_low, self.h1, 
                                                     dilation = dilation, 
                                                     groups = residual_low.shape[1])
            residual_low = causal_functional_conv1d(residual_low, self.h0, 
                                                    dilation = dilation, 
                                                    groups = residual_low.shape[1])

            y += self.w[:, i:i + 1] * residual_high
            dilation *= 2

        y += self.w[:, :1] * residual_low
        y += x * self.w[:, -1:]
        return self.dropout_layer(self.activation(y))
    
class MultiresScaleBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor = 2,
                 kernel_size = 3,
                 multires_depth = 6,
                 dropout = 0.0,
                 activation = torch.nn.GELU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.activation = activation

        self.multires_conv = CausalMultiresConv1d(in_channels, kernel_size, multires_depth, dropout, activation)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.multires_conv(x)
        x = torch.nn.functional.interpolate(x, scale_factor = self.scale_factor)
        x = self.conv(x)
        return self.activation(x)

class WaveletLayer(torch.nn.Module):
    """A layer that uses convolutions to project an input to a wavelet frequency basis, generating wavelets.
    Then, it uses a convolution to modify the output wavelet.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    out_channels : int
        Number of output channels, defaults to in_channels.
    wavelet_kernel_size : int
        Size of the wavelet kernel.
    out_conv_kernel_size : int
        Size of the output convolution kernel.
    scale_factor : int
        The scale factor of the wavelet. The wavelet will be scaled by this factor.
    n_points : int
       The number of points used for the wavelet. Must be divisible by scale_factor.
    interval : tuple
        The interval over which the wavelet is defined. 
    wavelet_scale : float
        The scale of the wavelet. This can be learned.
    multires_depth : int
        The depth of the multires convolutions. If 0, no multires convolutions are used.
    channelwise_scale : bool
        If true, then wavelet_scale is per channel. Otherwise, it is shared across channels.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels = None,
                 wavelet_kernel_size = 13,
                 out_conv_kernel_size = 3,
                 scale_factor = 2,
                 n_points = 16,
                 interval = (-10, 10),
                 wavelet_scale = 40, # this qualitatively balances wave and particle like properties
                 multires_depth = 0,
                 channelwise_scale = True,
                 ):
        
        super().__init__()
        assert n_points % scale_factor == 0, "n_points must be divisible by scale_factor"

        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        self.wavelet_kernel_size = wavelet_kernel_size
        self.out_conv_kernel_size = out_conv_kernel_size

        self.n_points = n_points
        self.scale_factor = scale_factor
        self.fold_dim = self.n_points // self.scale_factor

        self.hidden_channels = hidden_channels

        if multires_depth > 0:
            self.multires = True
            self.multires_block = CausalMultiresConv1d(self.hidden_channels,
                                                kernel_size = self.wavelet_kernel_size,
                                                depth = multires_depth)
        else:
            self.multires = False
        
        self.conv_in = torch.nn.Conv1d(self.in_channels, 
                                        self.hidden_channels, 
                                        self.wavelet_kernel_size, 
                                        padding = "same")

        self.conv_out = torch.nn.Conv1d(self.hidden_channels, 
                                        self.out_channels, 
                                        self.out_conv_kernel_size, 
                                        padding = "same")

        self.register_buffer("space", 
                             einops.rearrange(torch.linspace(*interval, n_points), "n -> 1 1 1 n"))

        # learnable parameter controlling how gaussian vs wavelike the wavelet is
        wavelet_scale = torch.tensor(wavelet_scale).float()
        if channelwise_scale:
            wavelet_scale = wavelet_scale.repeat(self.hidden_channels)
            wavelet_scale = einops.rearrange(wavelet_scale, "n -> 1 n 1 1")
        self.wavelet_scale = torch.nn.Parameter(wavelet_scale)
        # the cos component of the wavelet is constant, so we can precompute it
        self.register_buffer("cos_kernel", torch.cos(self.space))

    def forward(self, x):
        # this conv converts the input to frequency space
        x = self.conv_in(x).unsqueeze(-1)

        if self.multires:
            x = self.multires_block(x)

        # multiply conv by wavelet kernel (shape is (batch, channels, length, space))
        y = self.cos_kernel * torch.exp(-(self.space**2) / self.wavelet_scale) * x
        
        # blend wavelet space and length
        y = einops.rearrange(y, "b c l s -> b c (l s)") 
        expected_length = y.shape[-1] // self.fold_dim
        y_out = y.unfold(-1, self.n_points, self.fold_dim).sum(dim = -1)

        # unfortunately have to do some annoying padding
        size_diff = y_out.shape[-1] - expected_length
        if size_diff < 0:
            y_out = torch.cat([y_out, y[..., size_diff:]], dim = -1)
        
        y_out = self.conv_out(y_out)
        return y_out
    
    def export_in_conv_image(self, path):
        """Exports an image of the in convolution kernel"""
        kernel = self.conv_in.weight
        kernel = kernel.view(-1, kernel.shape[-1]).detach().cpu().numpy()
        plt.imshow(kernel, cmap = "seismic")
        plt.savefig(path)
        plt.close()


def simple_mixed_sin(num_freqs, interval, freq_range = 20, shift = 5, device = "cpu"):
    """Generates a signal composed of mixture of sinusoids."""
    freqs = torch.rand(num_freqs, device = device) * freq_range + shift
    freqs = torch.sort(freqs)[0]
    sins = torch.sin(2 * torch.pi * freqs.unsqueeze(-1) * interval.unsqueeze(0))
    sins = sins.mean(dim = 0, keepdim = True).unsqueeze(0)
    return freqs, sins


def test_wavelet(test_iterations = 1000, 
                 scale_factor = 8,
                 kernel_size = 13,
                 hidden_channels = 32,
                 num_freqs = 20, 
                 interval = torch.arange(-1, 1, 0.01),
                 channelwise = True):
    import os
    from utils import losses_to_running_loss as ltrl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    waver = WaveletLayer(1, 
                         hidden_channels,
                         wavelet_kernel_size = kernel_size,
                         scale_factor = scale_factor,
                         channelwise_scale = channelwise).to(device)

    os.makedirs("tmp", exist_ok = True)
    waver.export_in_conv_image("tmp/in_conv_start.png")

    x_hats = []
    losses_us = []
    losses = []

    optimizer = torch.optim.Adam(waver.parameters(), lr = 1e-3)
    for i in tqdm(range(test_iterations)):
        optimizer.zero_grad()

        _, sins = simple_mixed_sin(num_freqs, interval)
        sins = sins.to(device)

        sins_ds = torch.nn.functional.interpolate(sins, scale_factor = 1 / scale_factor)
        # this is counterfactual to see what the loss would be with regular upsampling
        sins_us = torch.nn.functional.interpolate(sins_ds, scale_factor = scale_factor)

        x_hat = waver(sins_ds)

        loss = torch.nn.functional.mse_loss(x_hat, sins)
        loss_us = torch.nn.functional.mse_loss(sins_us, sins).detach()

        loss.backward()
        optimizer.step()

        x_hats.append(x_hat.squeeze().squeeze().detach().cpu().numpy())
        losses.append(loss.item())
        losses_us.append(loss_us.item())


    fig, ax = plt.subplots(2, figsize = (10, 10))
    ax[0].plot(ltrl(losses, 0.99))
    ax[0].plot(ltrl(losses_us, 0.99), linestyle = "--")
    ax[0].set_title("loss")

    ax[1].plot(x_hats[-1], label = "reconstructed")
    ax[1].plot(sins.squeeze().squeeze().detach().cpu().numpy(), label = "original")
    ax[1].legend()
    ax[1].set_title("reconstruction")
    plt.show()

    waver.export_in_conv_image(f"tmp/in_conv_end.png")
    return waver


def test_multires(test_iterations = 500, num_freqs = 6, interval = torch.arange(-1, 1, 0.01)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Sequential(torch.nn.Conv1d(1, num_freqs, 1, padding = "same"),
                                torch.nn.GELU(),
                                CausalMultiresConv1d(num_freqs, 21, 6),
                                ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    fig, ax = plt.subplots()
    losses = []

    for i in tqdm(range(test_iterations)):
        optimizer.zero_grad()
        # randomly sample frequencies
        freqs, sins = simple_mixed_sin(num_freqs, interval)

        x_hat = model(sins).sum(dim = (0, -1))

        loss = torch.nn.functional.mse_loss(x_hat, freqs)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.99

    log_losses = np.log(losses)
    plt.plot(losses_to_running_loss(log_losses))
    plt.show()

if __name__ == "__main__":
    # testing on what is effectively fourier decomposition
    from tqdm import tqdm
    import torchaudio
    from utils import losses_to_running_loss
    wavelet_iters = 10000
    multires_iters = 0

    if wavelet_iters:
        waver = test_wavelet(test_iterations = wavelet_iters, scale_factor = 4)

    if multires_iters:
        test_multires(test_iterations = multires_iters)


