import torch
import torchaudio
import einops
import numpy as np
from math import ceil
from quantizer import ResidualQuantizer, tuple_checker
from wavelets import MultiresScaleBlock, WaveletLayer
from utils import add_util_norm, animate_sound, Snek

# the causal convolution layers were initially modified from:
# https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py
# and then updated based on:
# https://github.com/facebookresearch/encodec/blob/main/encodec/modules/conv.py
class CausalConv1d(torch.nn.Module):
    """A 1D convolution which preserves number of timesteps."""
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
    """A 1D transposed convolution which preserves number of timesteps."""
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
    
class CausalUpsampleConv1d(torch.nn.Module):
    """A 1D upsampling convolution which preserves number of timesteps.
    Avoids "checkerboard" artifacts by upsampling before convolution."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 bias=True,
                 norm = "weight"):
        super().__init__()
        self.scale_factor = stride
        self.conv = add_util_norm(torch.nn.Conv1d(in_channels, 
                                                  out_channels, 
                                                  kernel_size,
                                                  padding = "same",
                                                  bias=bias),
                                  norm = norm)


    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor = self.scale_factor)
        x = self.conv(x)
        return x

class CausalResidualBlock1d(torch.nn.Module):
    """A residual block with causal convolutions. Standard convolution followed by kernel = 1 convolution."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size = 7, 
                 dilation=1, 
                 bias=True, 
                 activation=torch.nn.LeakyReLU(0.1),
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
                 activation = torch.nn.LeakyReLU(0.1),
                 depthwise = False):
        super().__init__()
        dilations = [3**i for i in range(n_layers - 1)]

        layers = [torch.nn.Sequential(CausalResidualBlock1d(in_channels, 
                                                            in_channels, 
                                                            dilation=dilation,
                                                            depthwise = depthwise),
                                      activation
                                      ) for dilation in dilations]
        layers.append(torch.nn.Sequential(CausalConv1d(in_channels, 
                                                       out_channels, 
                                                       2 * stride + 1, 
                                                       stride=stride),
                                          activation,
                                          ))

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
                 activation = torch.nn.LeakyReLU(0.1),
                 depthwise = False,
                 upsample = True,
                 wavelet = False,
                 wavelet_hidden_ratio = 4,
                 channelwise = True):
        super().__init__()
        self.wavelet = wavelet

        dilations = [3**i for i in range(n_layers - 1)]
        if self.wavelet:
            conv_layer = WaveletLayer(in_channels, 
                                      out_channels * wavelet_hidden_ratio, # hidden channels
                                      out_channels = out_channels, 
                                      scale_factor = stride,
                                      wavelet_kernel_size = 2 * stride + 1,
                                      n_points = 2 * stride * wavelet_hidden_ratio,
                                      channelwise_scale = channelwise)

        elif upsample:
            conv_layer = CausalUpsampleConv1d(in_channels,
                                              out_channels,
                                              2 * stride + 1,
                                              stride = stride)
        else:
            conv_layer = CausalConvT1d(in_channels, 
                                       out_channels, 
                                       2 * stride + 1, 
                                       stride = stride)
        self.in_conv = torch.nn.Sequential(conv_layer,
                                           activation,
                                            )
        layers = [torch.nn.Sequential(CausalResidualBlock1d(out_channels, 
                                                            out_channels, 
                                                            dilation = dilation,
                                                            depthwise = depthwise),
                                      activation,
                                      ) for dilation in dilations]
        
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        x = self.in_conv(x)

        for layer in self.layers:
            x = layer(x)
        return x
    
class CausalVQAE(torch.nn.Module):
    def __init__(self,
                 in_channels = 1,
                 n_blocks = 5,
                 n_layers_per_block = 4,
                 first_block_channels = 32,
                 num_quantizers = 8,
                 codebook_size = 1024,
                 codebook_dim = 512,
                 vq_cutoff_freq = 1,
                 vq_type = "ema",
                 strides = (2, 3, 4, 4, 5),
                 input_format = "b l c",
                 channel_multiplier = 2,
                 norm = torch.nn.Identity,
                 depthwise = False,
                 use_som = True,
                 som_kernel_type = "gaussian",
                 wavelet_decoders = [False, False, True, False, False],
                 ):
        
        super().__init__()
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.num_quantizers = num_quantizers
        self.vq_cutoff_freq = vq_cutoff_freq

        self.codebook_dim = codebook_dim
        self.codebook_size = tuple_checker(codebook_size, num_quantizers)
        self.strides = tuple_checker(strides, n_blocks)
        self.scale_factor = np.prod(self.strides)

        if isinstance(wavelet_decoders, (list, tuple)):
            assert len(wavelet_decoders) == n_blocks, "Number of wavelet decoders must match number of blocks."
            # reverse the list since we iterate backwards
            wavelet_decoders = wavelet_decoders[::-1]
            self.wavelet_decoders = wavelet_decoders
        else:
            self.wavelet_decoders = [wavelet_decoders] * n_blocks

        self.quantizer = ResidualQuantizer(num_quantizers = num_quantizers, 
                                            dim = codebook_dim, 
                                            quantizer_class = vq_type, 
                                            codebook_sizes = codebook_size,
                                            vq_cutoff_freq = vq_cutoff_freq,
                                            use_som = use_som,
                                            som_kernel_type = som_kernel_type)

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
            wavelet_bool = self.wavelet_decoders[i - 1]
            decoders.append(CausalDecoderBlock(channel_sizes[i], 
                                               channel_sizes[i - 1], 
                                               self.strides[i - 1], 
                                               n_layers = self.n_layers_per_block,
                                               depthwise = depthwise,
                                               wavelet = wavelet_bool))
            
        # last decoder layer
        decoders.append(CausalConv1d(first_block_channels, in_channels, 7))

        if input_format == "b l c":
            self.rearrange_in = einops.layers.torch.Rearrange("b l c -> b c l")
            self.rearrange_out = einops.layers.torch.Rearrange("b c l -> b l c")
        else:
            self.rearrange_in = torch.nn.Identity()
            self.rearrange_out = torch.nn.Identity()

        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)

    def forward(self, x, update_codebook = False, codebook_n = None, prioritize_early = False):

        x_quantized, commit_loss, index = self.encode(x, update_codebook, codebook_n, prioritize_early)

        x = x_quantized

        for decoder in self.decoders:
            # each decoder takes the quantized from the previous decoder
            x = decoder(x)

        x = self.rearrange_out(x)
 
        return x, commit_loss, index
    
    def encode(self, x, update_codebook = False, codebook_n = None, prioritize_early = False):
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
        return x_quantized, commit_loss, index
    
    def sample(self, length = 225, device = "cuda", normal_var = 5e3, n_iters = 12):
        self.to(device)
        self.eval()
        with torch.no_grad():

            x = 0
            for i in range(self.num_quantizers):
                x_i= torch.randint(0, self.codebook_size[0],
                                    (1, length)).to(device)
                x_i = self.quantizer.quantizers[i].dequantize(x_i)
                x += x_i

            x = einops.rearrange(x, "b l c -> b c l" )

            for decoder in self.decoders:
                # each decoder takes the quantized from the previous decoder
                x = decoder(x)

            x = self.rearrange_out(x)

        self.train()
        return x
    
    def replace_quantizer(self, new_quantizer):
        self.quantizer = new_quantizer

    def update_cutoff(self, new_cutoff = None, ratio = None):
        self.quantizer.update_cutoff(new_cutoff = new_cutoff, ratio = ratio)
 

# note with input length 72000, latent dim with default params is 225 (stride factor 480)
#TODO : try varying codebooks sizes and dims
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from IPython.display import Audio

    test_sampling = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CausalVQAE(in_channels = 1, 
                       num_quantizers = 8, 
                       codebook_size = 1024, 
                       input_format = "n c l",
                       vq_type = "ema").to(device)
    
    if test_sampling:
        y = model.sample(device = device)
        Audio(y[0].detach().cpu().numpy(), rate = 16000)
    else:
        test_iterations = 50
        om = torchaudio.load(r"om.wav")[0]
        om = om.mean(dim = 0, keepdim = True).unsqueeze(0).to(device)

        # shape needs to be divisible by 480 for current architecture
        om = om[:, :, :65280]

        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

        for i in tqdm(range(test_iterations)):
            optimizer.zero_grad()
            y, commit_loss, index = model(om, update_codebook = True)

            loss = torch.mean((y - om).pow(2)) + commit_loss

            loss.backward()
            optimizer.step()
        animate_sound(om, model)



