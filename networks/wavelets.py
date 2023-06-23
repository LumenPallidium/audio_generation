import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt

class WaveletLayer(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels = None,
                 wavelet_kernel_size = 13,
                 center_kernel_size = 13,
                 out_conv_kernel_size = 8,
                 scale_factor = 2,
                 n_points = 10,
                 interval = (-5, 5),):
        
        super().__init__()

        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        self.wavelet_kernel_size = wavelet_kernel_size
        self.center_kernel_size = center_kernel_size
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
                                        padding = (self.out_conv_kernel_size // 2))

        self.register_buffer("space", einops.rearrange(torch.linspace(*interval, n_points), "n -> 1 1 1 n"))
        self.register_buffer("f_i", 1 / torch.sqrt(torch.log(torch.tensor(2))))

    def forward(self, x):
        x = self.conv_in(x).unsqueeze(-1)

        y = torch.cos(self.space) * torch.exp((-self.space**2) * torch.abs(x))
        
        # blend wavelet space and length
        y = einops.rearrange(y, "b c l s -> b c (l s)")
        y = y.unfold(-1, self.n_points, self.fold_dim).sum(dim = -1)
        
        y = self.conv_out(y)
        return y

    
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
    def __init__(self,
                 out_channels,
                 kernel_size,
                 depth,
                 dropout = 0.0,
                 activation = torch.nn.GELU()):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.dropout = dropout
        self.activation = activation

        scalar = sqrt(2.0 / (kernel_size * 2))

        # weights as described in paper
        self.h0 = torch.nn.Parameter(torch.empty(out_channels, 1, kernel_size).uniform_(-1., 1.) * scalar)
        self.h1 = torch.nn.Parameter(torch.empty(out_channels, 1, kernel_size).uniform_(-1., 1.) * scalar)
        w = torch.empty(out_channels, depth + 2).uniform_(-1., 1.) * sqrt(2.0 / (2 * depth + 4))
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

if __name__ == "__main__":
    from tqdm import tqdm
    import torchaudio
    test_iterations = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_wavelet = False

    om = torchaudio.load(r"om.wav")[0]
    om = om.mean(dim = 0, keepdim = True).unsqueeze(0).to(device)

    if test_wavelet:
        model= WaveletLayer(1, 2).to(device)
    else:
        model = torch.nn.Sequential(torch.nn.Conv1d(1, 32, 1, padding = "same"),
                                    CausalMultiresConv1d(32, 21, 4),
                                    torch.nn.Upsample(scale_factor = 2),
                                    torch.nn.Conv1d(32, 1, 1, padding = "same")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    om = torch.nn.functional.interpolate(om, size = 224)
    x_ds = torch.nn.functional.interpolate(om, scale_factor = 0.5)

    fig, ax = plt.subplots()
    x_hats = []
    losses = []
    for i in tqdm(range(test_iterations)):
        noise = 0#torch.randn_like(x_ds) * 0.1
        x_i = x_ds + noise

        x_hat = model(x_i)

        loss = torch.nn.functional.mse_loss(x_hat, om)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    plt.plot(losses)
    


