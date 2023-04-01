import torch
import einops
from utils import tuple_checker


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
                 scale = 1):
        super().__init__()

        n_steps = len(channel_sizes)

        self.channel_sizes = [in_channels] + channel_sizes
        self.kernel_sizes = tuple_checker(kernel_sizes, n_steps)
        self.strides = tuple_checker(strides, n_steps)
        self.groups = tuple_checker(groups, n_steps)

        self.activation = activation

        layers = [torch.nn.AvgPool1d(2 * scale, stride = scale, padding = scale)]
        layers += [torch.nn.Sequential(torch.nn.Conv1d(self.channel_sizes[i], self.channel_sizes[i + 1], 
                                                      kernel_sizes[i], stride = strides[i], 
                                                      groups = groups[i]),
                                      activation) for i in range(n_steps - 1)]
        
        # last layer does not have activation
        layers.append(torch.nn.Conv1d(channel_sizes[-1], 1, kernel_sizes[-1], stride = strides[-1], groups = groups[-1]))

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
                 n_blocks = 3,
                 scalefactor_per_block = 2,
                 feature_multiplier = 100):
        
        super().__init__()
        self.feature_multiplier = feature_multiplier
        scales = [scalefactor_per_block**i for i in range(n_blocks)]

        self.layers = torch.nn.ModuleList([WaveformDiscriminatorBlock(in_channels, scale = scale) for scale in scales])

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
                 activation = torch.nn.Identity()):
        
        super().__init__()

        if kernel_size is None:
            kernel_size = (stride[0] + 2, stride[1] + 2)

        self.layers = torch.nn.Sequential(torch.nn.Conv2d(in_channels, 
                                                          in_channels, 
                                                          kernel_size = 3),
                                           activation,
                                           torch.nn.Conv2d(in_channels, 
                                                           in_channels * channel_multiplier, 
                                                           stride = stride, 
                                                           kernel_size = kernel_size))
        

    def forward(self, x):
        return self.layers(x)# + x

class STFTDiscriminator(torch.nn.Module):
    """This is a discriminator based on the Short-Time Fourier Transform. In the paper,
    it uses the complex STFT, but here we use the real STFT, mostly because optimization
    of the complex STFT takes MUCH longer."""
    def __init__(self,
                 in_channels = 2,
                 first_channel_size = 32,
                 channel_multipliers = [2, 2, 1, 2, 1, 2],
                 strides = [(1, 2), (2, 2)] * 3,
                 n_fft = 1024,
                 hop_length = 256,
                 win_length = 1024,
                 feature_multiplier = 1):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.feature_multiplier = feature_multiplier

        self.num_blocks = len(channel_multipliers)

        self.first_conv = torch.nn.Conv2d(in_channels, 
                                          first_channel_size, 
                                          kernel_size = 7)

        blocks = []
        new_channel = first_channel_size
        for i, (multiplier, stride) in enumerate(zip(channel_multipliers, strides)):
            blocks.append(STFTDiscriminatorBlock(new_channel, multiplier, stride))
            new_channel = new_channel * multiplier

        self.blocks = torch.nn.ModuleList(blocks)

        self.final_conv = torch.nn.Conv2d(new_channel, 1, 
                                          kernel_size = (1, win_length // (2 ** (self.num_blocks + 1))))

    def forward(self, x):
        # need to remove and re-add channel dimension
        x = x.squeeze(1)
        x = torch.stft(x, n_fft = self.n_fft, 
                       hop_length = self.hop_length, 
                       win_length = self.win_length, 
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
        return x, features

def discriminator_generator_loss(original, reconstruction, discriminator, feature_multipier = 24):
    """A function that is mostly generic for types of dicriminators. Feature multiplier controls
    how much discrimination at different scales are weighted. For reference, the paper uses 
    100 for this multiplier. Another point of reference is that in general, the generation_loss
    value tends to be about 12x larger than feature loss."""
    # get discrimination on the real and fake waveform
    original_d, original_features = discriminator(original.clone().requires_grad_())
    reconstruction_d, reconstruction_features = discriminator(reconstruction)

    relu_f = torch.nn.functional.relu
    l1_f = torch.nn.functional.l1_loss

    # general hinge loss for GAN
    discriminator_loss = 0
    generation_loss = 0
    for x, y in zip(original_d, reconstruction_d):
        discriminator_loss += (relu_f(1 - x) + relu_f(1 + y.detach())).mean()
        generation_loss += relu_f(1 - y).mean()

    # feature wise loss - generated samples should "look-like" original at all scales
    feature_loss = 0
    for x, y in zip(original_features, reconstruction_features):
        feature_loss += l1_f(x, y)

    generator_loss = generation_loss + feature_multipier * feature_loss

    # k = number of levels in the discriminator
    k = len(original_d)

    return generator_loss / k, discriminator_loss / k