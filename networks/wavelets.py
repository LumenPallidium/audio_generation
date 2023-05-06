import torch
import einops
import numpy as np
import matplotlib.pyplot as plt

class WaveletLayer(torch.nn.Module):
    """A layer that takes an input, uses a 1x1 convolution to generate a scale
    for a morlet wavelet, then generates a morlet wavelet with that scale. This
    layer does upsample the input and is intended to be used in a audio-based
    decoders."""
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 out_conv_kernel_size = 7,
                 scale_factor = 2,
                 n_points = 10,
                 interval = (-5, 5),):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_points = n_points
        self.scale_factor = scale_factor

        self.conv = torch.nn.Conv1d(in_channels, hidden_channels, 1, padding = "same")
        self.conv_center = torch.nn.Conv1d(hidden_channels, hidden_channels, 1, padding = "same")
        self.conv_out = torch.nn.Conv1d(hidden_channels, out_channels, out_conv_kernel_size, padding = "same")

        self.space = einops.rearrange(torch.linspace(*interval, n_points), "n -> 1 1 1 n")
        self.f_i = 1 / torch.sqrt(torch.log(torch.tensor(2)))

    def forward(self, x):
        fold_dim = self.n_points // self.scale_factor

        x = self.conv(x)
        x_center = self.conv_center(x).unsqueeze(-1)
        x = x.unsqueeze(-1)
        y = torch.cos(2 * torch.pi * (x - x_center)) * torch.exp(-self.space**2 / (2 * x**2))
        #y = einops.rearrange(y, "b c (n f) d -> b c n (d f)", f = n_folds)
        y = einops.reduce(y, "b c (h block) (w f) -> b c (h block w)", "sum", 
                          block = self.scale_factor, f=fold_dim)
        
        y = self.conv_out(y)
        return y

if __name__ == "__main__":
    from tqdm import tqdm
    test_iterations = 1000
    waver = WaveletLayer(1, 3, 1)
    optimizer = torch.optim.Adam(waver.parameters(), lr = 1e-3)

    x = torch.linspace(0, 2 * torch.pi, 100)
    x = torch.sin(x).unsqueeze(0).unsqueeze(0)

    x_ds = torch.nn.functional.interpolate(x, scale_factor = 0.5)

    fig, ax = plt.subplots()
    x_hats = []
    losses = []
    for i in tqdm(range(test_iterations)):
        noise = torch.randn_like(x_ds) * 0.1
        x_i = x_ds + noise

        x_hat = waver(x_i)

        loss = torch.nn.functional.mse_loss(x_hat, x)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        ax.plot(x_hat.squeeze(0).squeeze(0).detach().numpy(), alpha = 0.01)

    ax.plot(x.squeeze(0).squeeze(0).detach().numpy(), color = "black")

    #plt.plot(x_hat.squeeze(0).squeeze(0).detach().numpy())

