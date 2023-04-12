import torch

# See https://ml-jku.github.io/hopfield-layers/, https://openreview.net/pdf?id=4nrZXPFN1c4 for mathematical reference
# Implementation based on jax here: https://github.com/bhoov/energy-transformer-jax

def value_and_grad(f, x):
    """Compute value and gradient of f at x, analogous to jax.value_and_grad"""
    x = x.detach().requires_grad_()
    y = f(x)
    y.backward()
    return y.detach(), x.grad.detach()

class EnergyMHA(torch.nn.Module):
    def __init__(self,
                 embed_dim,
                 n_heads,
                 beta = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if beta is None:
            # default to the standard scaling factor for attention
            self.beta = torch.nn.Parameter(torch.ones(1)) * 1 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        else:
            self.beta = beta

        self.Wq = torch.nn.Parameter(torch.randn(self.n_heads, self.head_dim, self.embed_dim))
        self.Wk = torch.nn.Parameter(torch.randn(self.n_heads, self.head_dim, self.embed_dim))

    def energy(self, x):
        """Input is (batch, length, embed_dim)"""
        k = torch.einsum("bld,hzd->blhz", x, self.Wk) # (batch, length, n_heads, head_dim)
        q = torch.einsum("bld,hzd->blhz", x, self.Wq)

        # attention, where each head has its own scaling factor
        # (batch, heads, length, length)
        attention = torch.einsum("h,bqhz,bkhz->bhqk", self.beta, k, q) # (batch, length, n_heads, head_dim)

        attention = torch.logsumexp(attention, dim = -1) # (batch, n_heads, length)
        attention = attention.sum(dim = -1) # (batch, n_heads)

        return (-1 / self.beta) * attention.sum(dim = -1) # (batch) 

    
    def forward(self, x):
        return value_and_grad(self.energy, x)