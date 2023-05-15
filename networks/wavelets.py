import torch
import einops
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    from tqdm import tqdm
    import torchaudio
    test_iterations = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    waver = WaveletLayer(1, 2).to(device)
    optimizer = torch.optim.Adam(waver.parameters(), lr = 1e-3)

    om = torchaudio.load(r"om.wav")[0]
    om = om.mean(dim = 0, keepdim = True).unsqueeze(0).to(device)


    om = torch.nn.functional.interpolate(om, size = 224)
    x_ds = torch.nn.functional.interpolate(om, scale_factor = 0.5)

    fig, ax = plt.subplots()
    x_hats = []
    losses = []
    for i in tqdm(range(test_iterations)):
        noise = torch.randn_like(x_ds) * 0.1
        x_i = x_ds + noise

        x_hat = waver(x_i)

        loss = torch.nn.functional.mse_loss(x_hat, om)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    plt.plot(losses)
    #ax.plot(x_hat.squeeze(0).squeeze(0).detach().cpu().numpy(), alpha = 0.05 * (i / test_iterations))
    #ax.plot(om.squeeze(0).squeeze(0).detach().cpu().numpy(), color = "black")

    #plt.plot(x_hat.squeeze(0).squeeze(0).detach().numpy())

