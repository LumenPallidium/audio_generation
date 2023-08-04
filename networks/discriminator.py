import torch
import einops
from utils import tuple_checker, add_util_norm, Snek

#TODO : look into nearly constant discriminator losses

class WaveformDiscriminatorBlock(torch.nn.Module):
    """Waveform discriminator block as described here:
    https://arxiv.org/pdf/1910.06711.pdf

    (See appendix A)
    """
    def __init__(self,
                 in_channels,
                 channel_sizes = [16, 64, 256, 512, 1024, 1024, 1024],
                 kernel_sizes = [15, 41, 41, 41, 41, 5, 3],
                 strides = [1, 4, 4, 4, 4, 1, 1],
                 groups = [1, 4, 16, 64, 256, 1, 1],
                 activation = torch.nn.LeakyReLU(0.2),
                 scale = 1,
                 norm = "spectral",
                 apply_sigmoid = True):
        super().__init__()

        n_steps = len(channel_sizes)

        self.channel_sizes = [in_channels] + channel_sizes
        self.kernel_sizes = tuple_checker(kernel_sizes, n_steps)
        self.strides = tuple_checker(strides, n_steps)
        self.groups = tuple_checker(groups, n_steps)

        layers = [torch.nn.AvgPool1d(2 * scale, stride = scale, padding = scale)]
        layers += [torch.nn.Sequential(add_util_norm(torch.nn.Conv1d(self.channel_sizes[i], self.channel_sizes[i + 1], 
                                                      kernel_sizes[i], stride = strides[i], 
                                                      groups = groups[i]),
                                                      norm = norm),
                                      activation) for i in range(n_steps - 1)]
        
        # last layer does not have activation
        layers.append(add_util_norm(torch.nn.Conv1d(channel_sizes[-1], 1, kernel_sizes[-1], stride = strides[-1], groups = groups[-1]),
                                    norm = norm))

        self.layers = torch.nn.ModuleList(layers)

        if apply_sigmoid:
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = torch.nn.Identity()

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        x = self.final_activation(x)
        return x, features

class WaveFormDiscriminator(torch.nn.Module):
    """As in the paper cited above, use three blocks, each acting on different scales"""
    def __init__(self,
                 in_channels,
                 name = "waveform_discriminator",
                 n_blocks = 3,
                 scalefactor_per_block = 2,
                 norm = "spectral"):
        
        super().__init__()
        self.name = name
        scales = [scalefactor_per_block**i for i in range(n_blocks)]

        self.layers = torch.nn.ModuleList([WaveformDiscriminatorBlock(in_channels, scale = scale, norm = norm) for scale in scales])

    def forward(self, x):
        features = []
        outputs = []

        for layer in self.layers:
            out, layer_features = layer(x)

            outputs.append(out)
            features.extend(layer_features)
        return outputs, features


class STFTDiscriminatorBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 channel_multiplier,
                 stride,
                 kernel_size = None,
                 padding = None,
                 activation = torch.nn.LeakyReLU(0.2),
                 norm = "spectral"):
        
        super().__init__()

        if kernel_size is None:
            kernel_size = (stride[0] + 2, stride[1] + 2)
        if padding is None:
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        self.layers = torch.nn.Sequential(add_util_norm(torch.nn.Conv2d(in_channels, 
                                                                        in_channels, 
                                                                        kernel_size = 3,
                                                                        padding = 1),
                                                        norm = norm),
                                           activation,
                                           add_util_norm(torch.nn.Conv2d(in_channels, 
                                                                         in_channels * channel_multiplier, 
                                                                         stride = stride, 
                                                                         kernel_size = kernel_size,
                                                                         padding = padding),
                                                         norm = norm),)
        

    def forward(self, x):
        return self.layers(x)# + x

class STFTDiscriminator(torch.nn.Module):
    """This is a discriminator based on the Short-Time Fourier Transform. Works in the complex
    domain."""
    def __init__(self,
                 in_channels = 2,
                 first_channel_size = 32,
                 channel_multipliers = [2, 2, 1, 2, 1, 2],
                 strides = [(1, 2), (2, 2)] * 3,
                 win_length = 1024,
                 n_fft = None,
                 hop_length = None,
                 feature_multiplier = 1,
                 normalize_stft = True,
                 norm = "spectral",
                 base_name = "stft_discriminator",
                 apply_sigmoid = True):
        super().__init__()

        self.win_length = win_length
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 4
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize_stft = normalize_stft
        
        self.feature_multiplier = feature_multiplier
        self.name = f"{base_name}_{win_length}"

        self.num_blocks = len(channel_multipliers)

        self.first_conv = add_util_norm(torch.nn.Conv2d(in_channels, 
                                          first_channel_size, 
                                          kernel_size = 7,
                                          padding = 3),
                                        norm = norm)

        blocks = []
        new_channel = first_channel_size
        for i, (multiplier, stride) in enumerate(zip(channel_multipliers, strides)):
            blocks.append(STFTDiscriminatorBlock(new_channel, multiplier, stride, norm = norm))
            new_channel = new_channel * multiplier

        self.blocks = torch.nn.ModuleList(blocks)

        final_kernel_size = win_length // (2 ** (self.num_blocks + 1))
        self.final_conv = add_util_norm(torch.nn.Conv2d(new_channel, 1, 
                                          kernel_size = (1, final_kernel_size),
                                          padding = (0, (final_kernel_size - 1) // 2)),
                                        norm = norm)
        
        if apply_sigmoid:
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = torch.nn.Identity()

    def forward(self, x):
        # need to remove and re-add channel dimension
        x = x.squeeze(1)
        x = torch.stft(x, n_fft = self.n_fft, 
                       hop_length = self.hop_length, 
                       win_length = self.win_length,
                       normalized = self.normalize_stft,
                       return_complex = False,
                       onesided = False)
        # flip time and frequency, convert real/imaginary part to channels
        x = einops.rearrange(x, "b f t c -> b c t f")

        x = self.first_conv(x)

        features = [x]

        for block in self.blocks:
            x = block(x)
            features.append(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return [x], features

def discriminator_generator_loss(original, 
                                 reconstruction, 
                                 discriminator, 
                                 feature_multipier = 100, 
                                 scale_feature_loss = True):
    """A function that is mostly generic for types of dicriminators. Feature multiplier controls
    how much discrimination at different scales are weighted. For reference, the paper uses 
    100 for this multiplier."""
    # get discrimination on the real and fake waveform
    original_d, original_features = discriminator(original.clone().requires_grad_())
    reconstruction_d, reconstruction_features = discriminator(reconstruction)
    # why a second time? we need one copy for the generator loss through the full disc + generator graph, and one for the discriminator loss
    reconstruction_d2, _ = discriminator(reconstruction.detach().clone().requires_grad_())

    # k = number of levels in the discriminator
    k = len(original_d)
    l1_f = torch.nn.functional.l1_loss

    # hinge loss for GAN
    discriminator_loss = 0
    generation_loss = 0
    for x, y, y_disc in zip(original_d, reconstruction_d, reconstruction_d2):
        real_d_loss = -torch.minimum(x - 1, torch.zeros_like(x)).mean()
        fake_d_loss = -torch.minimum(-y_disc - 1, torch.zeros_like(y_disc)).mean()
        discriminator_loss += (real_d_loss + fake_d_loss) / k

        generation_loss += -(y.mean() / k)

    # feature wise loss - generated samples should "look-like" original at all scales
    feature_loss = 0
    n_features = len(original_features)
    for x, y in zip(original_features, reconstruction_features):
        feature_loss_i = l1_f(x, y) / n_features
        if scale_feature_loss:
            feature_loss_i /= torch.abs(x + 1e-3).mean()

        feature_loss += feature_loss_i

    generator_loss = generation_loss + feature_multipier * feature_loss

    return generator_loss, discriminator_loss

if __name__ == "__main__":
    # test by seeing if we can get a generator to implictly approximate frequency distribution of a signal
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from wavelets import simple_mixed_sin
    n_iters = 1000
    batch_size = 32
    hidden_dim = 32
    n_freqs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interval = torch.arange(-1, 1, 2 / 32768, device = device)
    disc = WaveFormDiscriminator(1).to(device)
    generator = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, n_freqs)).to(device)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr = 1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr = 1e-3)

    losses_g = []
    losses_d = []

    for iter in tqdm(range(n_iters)):
        opt_g.zero_grad()
        opt_d.zero_grad()

        _, input_t = simple_mixed_sin(20, interval, device = device)
        
        z = torch.randn(hidden_dim, device = device)

        # generation steps
        x = generator(z)
        x = torch.sin(2 * torch.pi * x.unsqueeze(-1) * interval.unsqueeze(0))
        x = x.mean(dim = 0, keepdim = True).unsqueeze(0)

        loss_g, loss_d = discriminator_generator_loss(input_t, x, disc,
                                                      feature_multipier = 0)

        loss_g.backward(retain_graph = True)
        opt_g.step()

        loss_d.backward()
        opt_d.step()

        losses_g.append(loss_g.item())
        losses_d.append(loss_d.item())
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(losses_g, label = "generator")
    ax[0].plot(losses_d, label = "discriminator")
    ax[0].legend()

    ax[1].plot(input_t[0][0].detach().cpu().numpy(), label = "original")
    ax[1].plot(x[0][0].detach().cpu().numpy(), label = "generated")
    ax[1].legend()