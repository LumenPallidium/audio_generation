import torch
from torch.nn import functional as F
import einops

# Module for quantizers. Lots of influence from OpenAI's Jukebox VQ-VAE, rosinality's VQ-VAE, and the OG paper:
# https://arxiv.org/pdf/1711.00937.pdf

# TODO : Add support for different types of quantizers, e.g. Gumbel-Softmax, etc.
# TODO : maybe add a repulsion term to make the codebook vectors fill the data manifold better


def tuple_checker(item, length):
    if isinstance(item, (int, float, str)):
        item = [item] * length
    elif isinstance(item, (tuple, list)):
        assert len(item) == length, f"Expected tuple of length {length}, got {len(item)}"
    return item

class BaseQuantizer(torch.nn.Module):
    """Base class for quantizers.
    Parameters
    ----------
    dim : int
        Dimension of the input.
    codebook_size : int
        Number of entries in the codebook.
    """
    def __init__(self, dim : int, 
                 codebook_size : int,
                 cut_freq : int = 2,
                 alpha : float = 0.95,
                 replace_with_obs : bool = True,
                 init_scale = 1.0):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.alpha = alpha
        self.cut_freq = cut_freq
        self.replace_with_obs = replace_with_obs

        self.stale_clusters = None

        # initialize codebook as parameter to enable optimization
        self.codebook = torch.nn.Parameter(torch.randn(dim, codebook_size) * init_scale)
        # initalize a count that each codebook entry appears
        self.register_buffer("cluster_frequency", torch.ones(codebook_size))


    def quantize(self, input):
        """Quantize the input. Returns the index of the closest codebook entry for each input element."""
        flatten = einops.rearrange(input, "b ... d -> b (...) d")

        # distances, note this is ~~ (flatten - codebook)T @ (flatten - codebook), where T is the transpose
        # TODO : look into replacing this with torch.cdist
        dist = (
            flatten.pow(2).sum(-1, keepdim=True)
            - 2 * flatten @ self.codebook
            + self.codebook.pow(2).sum(0, keepdim=True)
        )

        # get the closest codebook entry
        min_dist, codebook_index = torch.min(dist, dim=-1)

        return codebook_index, flatten

    def dequantize(self, codebook_index):
        """Given the symbol/index of the codebook entry, return the corresponding vector in the codebook."""
        return F.embedding(codebook_index, self.codebook.T)

    def codebook_update_function(self, x, x_quantized, codebook_index, x_flat, codebook_onehot):
        """Default way to update the codebook is by gradient descent on a codebook loss. Accepts extra arguments for
        other types of quantizers that may need them."""
        # codebook loss, must stop/detach the gradient of the non-quantized input
        codebook_loss = (x_quantized - x.detach()).pow(2).mean()
        
        return codebook_loss
    
    def update_codebook_count(self, codebook_index, x_flat, verbose = False, update_codebook = True):
        """Update the count of how many times each codebook entry has been used, useful for avoiding
        codebook collapse."""
        with torch.no_grad():
            # get a one-hot encoding of the codebook index
            codebook_onehot = F.one_hot(codebook_index, self.codebook_size).type(x_flat.dtype)
            # use that to get the count that each codebook entry has been used
            codebook_count = torch.einsum("b l c -> c", codebook_onehot)

            # take ema weighted sum of current code/symbol use frequencies
            self.cluster_frequency.data.mul_(self.alpha).add_(codebook_count, alpha = 1 - self.alpha)

            # this vector leverages both the historical count and more recent counts
            comparison_clusters = torch.maximum(self.cluster_frequency, codebook_count)

            # replace the codebook entries that have been used less than the cut frequency
            low_clusters = comparison_clusters < self.cut_freq
            num_low_clusters = int(low_clusters.sum().item())
            self.stale_clusters = num_low_clusters
            # all clusters will be low clusters on the first run, so only update in between
            if update_codebook & ((low_clusters.any()) and not (low_clusters.all())):
                
                if verbose:
                    print(f"{num_low_clusters} clusters are poorly represented. Updating...")
                if not self.replace_with_obs:
                    high_vectors = self._replace_with_high(low_clusters, num_low_clusters)
                else:
                    # get a sample from x_flat with size num_low_clusters
                    high_vectors = einops.rearrange(x_flat.detach().clone(), "b l d -> (b l) d")
                    # shuffle them  - don't want to add position-based bias - and select num_low_clusters of them
                    high_vectors = high_vectors[torch.randperm(high_vectors.shape[0]), :][:num_low_clusters].T

                # convert low clusters to indices
                low_clusters = low_clusters.nonzero().squeeze(1)
                    
                # replace the low clusters with the new clusters
                
                self.codebook[:, low_clusters] = high_vectors

        return codebook_onehot
    
    def _replace_with_high(self, low_clusters, num_low_clusters):
        if num_low_clusters <= self.codebook_size // 2:

            high_clusters = torch.topk(self.cluster_frequency, 
                                    num_low_clusters,
                                    largest = True, sorted = False)[1]
        else:
            # note that these are indices and not a bool vector like low_clusters
            high_clusters = (~low_clusters).nonzero().squeeze(1)
            num_high_clusters = high_clusters.shape[0]
            # if there are many low clusters, use the highest multiple times
            n_repeats = num_low_clusters // num_high_clusters
            remainder = num_low_clusters % num_high_clusters

            high_clusters = high_clusters.repeat(n_repeats)
            if remainder > 0:
                # repeat the remainder
                high_clusters = torch.cat([high_clusters, high_clusters[:remainder]])

        high_vectors = self.codebook[:, high_clusters].detach().clone()

        # get the high vectors, jitter them
        high_vectors += torch.randn_like(high_vectors) * self.new_code_noise
        return high_vectors

    def forward(self, x, update_codebook : bool = False):
        """Quantize and dequantize the input. Returns the quantized input, the index of the codebook entry for each
        input element, and the commitment loss. Note that update_codebook means to reassign input vectors to 
        codebook entries that are poorly represented; the codebook updates via gradient descent (or
        k-means for the EMAQuantizer) regardless of this flag."""
        codebook_index, x_flat = self.quantize(x)
        x_quantized = self.dequantize(codebook_index)

        # the commitment loss, stop/detach the gradient of the quantized input
        inner_loss = (x_quantized.detach() - x).pow(2).mean()
        if self.training:
            codebook_onehot = self.update_codebook_count(codebook_index, x_flat, update_codebook = update_codebook)
            codebook_loss = self.codebook_update_function(x, x_quantized, codebook_index, x_flat, codebook_onehot)
            if codebook_loss is not None:
                inner_loss += codebook_loss

        # passes the gradient through the quantization for the reconstruction loss
        x_quantized = x + (x_quantized - x).detach()

        return x_quantized, codebook_index, inner_loss

class EMAQuantizer(BaseQuantizer):
    """Quantizer that uses an exponential moving average to update the codebook, 
    based on the Appendix of the VQ-VAE paper.
    Parameters
    ----------
    dim : int
        Dimension of the input.
    codebook_size : int
        Number of entries in the codebook.
    alpha : float
        The exponential moving average coefficient.
    """
    def __init__(self, dim : int, 
                 codebook_size : int, 
                 alpha : float = 0.99, 
                 eps : float = 1e-5,
                 cut_freq : int = 2,
                 new_code_noise : float = 1e-4,
                 replace_with_obs = True,
                 init_scale = 1.0):
        super().__init__(dim, 
                         codebook_size, 
                         alpha = alpha, 
                         cut_freq = cut_freq, 
                         replace_with_obs = replace_with_obs,
                         init_scale = init_scale)
        
        self.eps = eps
        self.new_code_noise = new_code_noise

        # disable grad for the codebook
        self.codebook.requires_grad = False
        
        # many implementations initialize this to 0s, but there are less annoying numerical issues when started as 1s
        # this prior also makes more sense: a uniform distribution over the codebook by default
        self.register_buffer("ema_codebook", self.codebook.clone())

    def codebook_update_function(self, x, x_quantized, codebook_index, x_flat, codebook_onehot):
        """Update the codebook using an exponential moving average. In this case, the vectors in the codebook are updated by 
        averaging the input vectors that are closest to them. This makes the codebook better represent the encoding of the data.
        The description can be found in Appendix A.1 here: https://arxiv.org/pdf/1711.00937.pdf"""

        # projects the input onto the closest codebook entry, taking a mean along the batch and length
        size = x_flat.shape[0] * x_flat.shape[1]
        codebook_sum = torch.einsum("b l d, b l c -> d c", x_flat, codebook_onehot) / size

        # update ema codebook with the input vectors
        self.ema_codebook.data.mul_(self.alpha).add_(codebook_sum, alpha = 1 - self.alpha)

        # normalize the codebook
        n = self.cluster_frequency.sum()
        cluster_frequency_normalized = ((self.cluster_frequency + self.eps) / (n + self.eps) * n)
        codebook_normalized = self.ema_codebook / cluster_frequency_normalized.unsqueeze(0)

        # overwrite codebook
        self.codebook.data.copy_(codebook_normalized)
    

class ResidualQuantizer(torch.nn.Module):
    """Residual vector quantization, as here:
    https://arxiv.org/pdf/2107.03312.pdf
    
    Essentially, we quantize the encoder output, and then quantize the residual iteratively.
    """
    def __init__(self,
                 num_quantizers,
                 dim,
                 codebook_sizes,
                 quantizer_class = "ema",
                 scale_factor = 4.0,
                 priority_n = 24):
        
        super().__init__()

        self.num_quantizers = num_quantizers
        self.dim = dim
        self.codebook_sizes = tuple_checker(codebook_sizes, num_quantizers)
        self.priority_n = priority_n

        # residual gets smaller at each step, so can be helpful to have small quantizer vectors
        scale_factors = [1 / (scale_factor ** i) for i in range(num_quantizers)]

        quantizer_type = EMAQuantizer if quantizer_class == "ema" else BaseQuantizer
        print(f"Initializing residual quantizer with class {quantizer_class}")

        quantizers = [quantizer_type(self.dim, codebook_size, init_scale = scale) for codebook_size, scale in zip(self.codebook_sizes, scale_factors)]

        self.quantizers = torch.nn.ModuleList(quantizers)

    def forward(self, x, n = None, update_codebook : bool = False, prioritize_early : bool = False):
        # can limit to first n quantizers, they call this bitrate dropout in the paper
        # if n is None, use all quantizers, for training n will typically be sampled uniformly from [1, num_quantizers]
        if n is None:
            n = self.num_quantizers + 1
        x_hat = 0
        residual = x
        inner_loss = 0

        indices = []

        for i in range(n):
            x_i, index, inner_loss_i = self.quantizers[i](residual, update_codebook = update_codebook)
            stale_clusters = self.quantizers[i].stale_clusters

            # prioritize early quantizers, switch off update_codebook if stale clusters exceeds threshold
            if prioritize_early and update_codebook and (stale_clusters > self.priority_n):
                update_codebook = False

            x_hat += x_i
            residual -= x_i.detach()
            inner_loss += inner_loss_i
            indices.append(index)
        return x_hat, torch.stack(indices), inner_loss
    
    def get_stale_clusters(self):
        """Get the number of stale clusters from all quantizers"""
        stale_clusters = []
        for quantizer in self.quantizers:
            stale_clusters.append(quantizer.stale_clusters)
        return stale_clusters


if __name__ == "__main__":
    # test the quantizer
    from tqdm import tqdm

    test_base = False
    test_iterations = 3000

    device = "cuda"

    # test the quantizers
    if test_base:
        quantizer = BaseQuantizer(2, 10)
        optimizer = torch.optim.Adam(quantizer.parameters(), lr=1e-2)
    else:
        quantizer = EMAQuantizer(2, 10)

    quantizer.to(device)

    for iter in tqdm(range(test_iterations)):
        x = torch.randn((1, 3, 2)).to(device)

        x_quantized, index, inner_loss = quantizer(x, update_codebook = True)

        if test_base:
            # optimize the base quantizer
            optimizer.zero_grad()
            inner_loss.backward()
            optimizer.step()
