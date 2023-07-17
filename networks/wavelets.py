import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt

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
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels = None,
                 wavelet_kernel_size = 7,
                 out_conv_kernel_size = 7,
                 scale_factor = 2,
                 n_points = 16,
                 interval = (-2, 2),):
        
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

        self.conv_in = torch.nn.Conv1d(self.in_channels, 
                                       self.hidden_channels, 
                                       self.wavelet_kernel_size, 
                                       padding = "same")

        self.conv_out = torch.nn.Conv1d(self.hidden_channels, 
                                        self.out_channels, 
                                        self.out_conv_kernel_size, 
                                        padding = "same")

        space = einops.rearrange(torch.linspace(*interval, n_points), "n -> 1 1 1 n")
        # the spatial extent of the wavelet is constant, so we can precompute it
        self.register_buffer("wavelet_kernel", torch.cos(space) * torch.exp((-space**2)))

    def forward(self, x):
        # this conv converts the input to frequency space
        x = self.conv_in(x).unsqueeze(-1)

        # multiply conv by wavelet kernel (shape is (batch, channels, length, space))
        y = self.wavelet_kernel * torch.abs(x)
        
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
    
def test_wavelet(test_iterations = 1000, scale_factor = 2, plot_losses = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    waver = WaveletLayer(1, 2, scale_factor = scale_factor).to(device)
    optimizer = torch.optim.Adam(waver.parameters(), lr = 1e-3)

    om = torchaudio.load(r"om.wav")[0]
    om = om.mean(dim = 0, keepdim = True).unsqueeze(0).to(device)

    om = torch.nn.functional.interpolate(om, size = 2 ** 14)
    x_ds = torch.nn.functional.interpolate(om, 
                                           scale_factor = 1 / scale_factor)

    fig, ax = plt.subplots()
    x_hats = []
    losses = []
    for i in tqdm(range(test_iterations)):
        optimizer.zero_grad()
        noise = torch.randn_like(x_ds) * 1e-3
        x_i = x_ds + noise

        x_hat = waver(x_i)

        loss = torch.nn.functional.mse_loss(x_hat, om)
        loss.backward()
        optimizer.step()

        x_hats.append(x_hat.squeeze().squeeze().detach().cpu().numpy())
        losses.append(loss.item())

    if plot_losses:
        plt.plot(losses)
    else:
        ax.plot(x_hats[-1], label = "reconstructed")
        ax.plot(om.squeeze().squeeze().detach().cpu().numpy(), label = "original")
    plt.show()
    

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
        freqs = torch.rand(num_freqs) * 4
        freqs = torch.sort(freqs)[0]
        sins = torch.sin(2 * torch.pi * freqs.unsqueeze(-1) * interval.unsqueeze(0))
        sins = sins.sum(dim = 0, keepdim = True).unsqueeze(0)

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
    wavelet_iters = 1000
    multires_iters = 0

    if wavelet_iters:
        test_wavelet(test_iterations = wavelet_iters, scale_factor = 4)

    if multires_iters:
        test_multires(test_iterations = multires_iters)


