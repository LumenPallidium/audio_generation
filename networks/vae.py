import torch
import torchaudio
import einops
from math import ceil
from quantizer import ResidualQuantizer, tuple_checker
from utils import add_util_norm

# the causal convolution layers were initially modified from:
# https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py
# and then updated based on:
# https://github.com/facebookresearch/encodec/blob/main/encodec/modules/conv.py
class CausalConv1d(torch.nn.Module):
    """A 1D convolution which masks future inputs."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation=1, 
                 stride = 1, 
                 bias=True,
                 groups = 1,
                 norm = "weight"):
        super().__init__()
        self.conv = add_util_norm(torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                                    stride = stride, dilation=dilation, bias=bias,
                                    groups = groups),
                                  norm = norm)
        self.dilation = dilation

        self.pad = self.dilation * (self.conv.kernel_size[0] - 1) - stride + 1

    def forward(self, x):
        extra_pad = self._calc_extra_pad(x)
        x = torch.nn.functional.pad(x, (self.pad, extra_pad))
        return self.conv(x)
    
    def _calc_extra_pad(self, x):
        length = x.shape[-1]
        next_length = (length - self.conv.kernel_size[0] + self.pad) / self.conv.stride[0] + 1
        target_length = (ceil(next_length) - 1) * self.conv.stride[0] + self.conv.kernel_size[0] - self.pad
        return target_length - length
    
class CausalConvT1d(torch.nn.Module):
    """A 1D transposed convolution which masks future inputs."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 bias=True,
                 norm = "weight"):
        super().__init__()
        self.conv = add_util_norm(torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 
                                             stride = stride, bias=bias),
                                  norm = norm)
        self.right_pad = kernel_size - stride


    def forward(self, x):
        x = self.conv(x)
        pad = x.shape[-1] - self.right_pad
        return x[..., :pad]
    
class CausalResidualBlock1d(torch.nn.Module):
    """A residual block with causal convolutions. Standard convolution followed by kernel = 1 convolution."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size = 7, 
                 dilation=1, 
                 bias=True, 
                 activation=torch.nn.LeakyReLU(negative_slope=0.3),
                 dropout = 0.0,
                 depthwise = False):
        super().__init__()
        if depthwise:
            self.conv1 = torch.nn.Sequential(CausalConv1d(in_channels, in_channels, 1, bias=bias, groups = in_channels),
                                             CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias))
        else:
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

        self.conv2 = CausalConv1d(out_channels, out_channels, 1, bias=bias)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout)

    
    def forward(self, x):
        x_p = self.activation(self.conv1(x))
        x_p = self.conv2(x_p)
        x_p = self.dropout(x_p)
        return x + x_p
    
class CausalEncoderBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 n_layers = 4,
                 activation = torch.nn.LeakyReLU(negative_slope=0.3),
                 depthwise = False):
        super().__init__()
        dilations = [3**i for i in range(n_layers - 1)]

        layers = [torch.nn.Sequential(CausalResidualBlock1d(in_channels, 
                                                            in_channels, 
                                                            dilation=dilation,
                                                            depthwise = depthwise),
                                      activation) for dilation in dilations]
        layers.append(torch.nn.Sequential(CausalConv1d(in_channels, out_channels, 2 * stride, stride=stride),
                                          activation))

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class CausalDecoderBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 n_layers = 4,
                 activation = torch.nn.LeakyReLU(negative_slope=0.3),
                 depthwise = False):
        super().__init__()
        dilations = [3**i for i in range(n_layers - 1)]
        self.in_conv = torch.nn.Sequential(CausalConvT1d(in_channels, out_channels, 2 * stride, stride=stride),
                                           activation)
        layers = [torch.nn.Sequential(CausalResidualBlock1d(out_channels, 
                                                            out_channels, 
                                                            dilation=dilation,
                                                            depthwise = depthwise),
                                      activation) for dilation in dilations]
        
        self.layers = torch.nn.ModuleList(layers)


    def forward(self, x):
        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class CausalVQAE(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 n_blocks = 4,
                 n_layers_per_block = 4,
                 first_block_channels = 32,
                 num_quantizers = 8,
                 codebook_size = 1024,
                 codebook_dim = 512,
                 vq_type = "base",
                 strides = (2, 4, 5, 8),
                 input_format = "b l c",
                 channel_multiplier = 2,
                 norm = torch.nn.Identity,
                 zero_center = False,
                 depthwise = False):
        
        super().__init__()
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.num_quantizers = num_quantizers
        self.zero_center = zero_center

        self.codebook_dim = codebook_dim
        self.codebook_size = tuple_checker(codebook_size, n_blocks)
        self.strides = tuple_checker(strides, n_blocks)

        self.quantizer = ResidualQuantizer(num_quantizers = num_quantizers, 
                                           dim = codebook_dim, 
                                           quantizer_class = vq_type, 
                                           codebook_sizes = codebook_size)

        channel_sizes = [first_block_channels * channel_multiplier**i for i in range(n_blocks + 1)]

        # init encoders
        encoders = [torch.nn.Sequential(norm(),
                                        CausalConv1d(in_channels, first_block_channels, 7))]

        for i in range(self.n_blocks):
            encoders.append(CausalEncoderBlock(channel_sizes[i], 
                                               channel_sizes[i + 1], 
                                               self.strides[i], 
                                               self.n_layers_per_block, 
                                               depthwise = depthwise))
            
        encoders.append(CausalConv1d(channel_sizes[-1], codebook_dim, 3))

        # init decoders
        decoders = [CausalConvT1d(codebook_dim, channel_sizes[-1], 7)]

        for i in range(self.n_blocks, 0 , -1):
            decoders.append(CausalDecoderBlock(channel_sizes[i], 
                                               channel_sizes[i - 1], 
                                               self.strides[i - 1], 
                                               n_layers = self.n_layers_per_block,
                                               depthwise = depthwise))
            
        # last decoder layer
        decoders.append(CausalConv1d(first_block_channels, in_channels, 7))

        if input_format == "b l c":
            self.rearrange_in = einops.layers.torch.Rearrange("b l c -> b c l")
            self.rearrange_in = einops.layers.torch.Rearrange("b c l -> b l c")
        else:
            self.rearrange_in = torch.nn.Identity()
            self.rearrange_out = torch.nn.Identity()

        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)

    def forward(self, x, update_codebook = False, codebook_n = None, multiscale = False, prioritize_early = False):
        if self.zero_center:
            mean = x.mean(dim = -1, keepdim = True).detach()
            x = x - mean

        x = self.rearrange_in(x)

        for encoder in self.encoders:
            x = encoder(x)

        x = einops.rearrange(x, "b c l -> b l c") # maybe inefficient

        x_quantized, index, commit_loss = self.quantizer(x, 
                                                        codebook_n, 
                                                        update_codebook = update_codebook,
                                                        prioritize_early = prioritize_early)

        x_quantized = einops.rearrange(x_quantized,
                                        "b l c -> b c l" )

        multiscales = []
        for decoder in self.decoders:
            # each decoder takes the quantized from the previous decoder
            x_quantized = decoder(x_quantized)
            # multiscale forces the average (over channels) to be close to the original
            if multiscale:
                multiscales.append(x_quantized.mean(dim = 1, keepdim = True))

        x_quantized = self.rearrange_out(x_quantized)

        if self.zero_center:
            x_quantized = x_quantized + mean
 
        return x_quantized, commit_loss, index, multiscales


#TODO : try varying codebooks sizes and dims
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalVQAE(1, input_format = "n c l").to(device)

    x = torch.randn(8, 1, 72000).to(device)
    y, commit_loss, index, multiscales = model(x, codebook_n = model.quantizer.num_quantizers)



