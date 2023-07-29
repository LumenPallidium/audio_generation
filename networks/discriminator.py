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
                 norm = "spectral"):
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

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features

class WaveFormDiscriminator(torch.nn.Module):
    """As in the paper cited above, use three blocks, each acting on different scales"""
    def __init__(self,
                 in_channels,
                 name = "waveform_discriminator",
                 n_blocks = 3,
                 scalefactor_per_block = 2,
                 feature_multiplier = 100,
                 norm = "spectral"):
        
        super().__init__()
        self.feature_multiplier = feature_multiplier
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
                 base_name = "stft_discriminator"):
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
    # why a second time? we need one copy for the generator loss throught the full disc + generator graph, and one for the discriminator loss
    reconstruction_d2, _ = discriminator(reconstruction.detach().clone().requires_grad_())

    # k = number of levels in the discriminator
    k = len(original_d)

    relu_f = torch.nn.functional.relu
    l1_f = torch.nn.functional.l1_loss

    # general hinge loss for GAN
    discriminator_loss = 0
    generation_loss = 0
    for x, y, y_disc in zip(original_d, reconstruction_d, reconstruction_d2):
        discriminator_loss += (relu_f(1 - x).mean() + relu_f(1 + y_disc).mean()) / k

        generation_loss += relu_f(1 - y).mean() / k

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
    input_t = torch.randn(1, 1, 72000)
    disc = STFTDiscriminator(win_length = 128)
    disc(input_t)