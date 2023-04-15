import torch

# See https://ml-jku.github.io/hopfield-layers/, https://openreview.net/pdf?id=4nrZXPFN1c4 for mathematical reference
# Implementation based on jax here: https://github.com/bhoov/energy-transformer-jax

def value_and_grad(f, x):
    """Compute value and gradient of f at x, analogous to jax.value_and_grad"""
    x = x.detach()
    y = f(x)

    grads = torch.autograd.functional.jacobian(f, x)
    grads = grads.sum(dim = 0) # sum over extra batch dimension
    return y.detach(), grads

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
            scale = 1 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
            self.beta = torch.nn.Parameter(torch.ones(self.n_heads)* scale) 
        else:
            self.beta = torch.nn.Parameter(beta)

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

        return ((-1 / self.beta) * attention).sum(dim = -1) # (batch) 

    
    def forward(self, x):
        return value_and_grad(self.energy, x)
    
class BaseModernHopfield(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 use_bias = False,
                 beta = None):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        self._init_activation(beta = beta)
               
        # in order to match the jax implementation, not doing a Linear layer here
        self.W = torch.nn.Parameter(torch.randn(self.in_dim, self.hidden_dim))

    def _init_activation(self, beta):
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.activation = torch.nn.ReLU()

    def activation_energy(self, x):
        energy = self.activation(x)
        return -0.5*(energy**2).sum()
    
    def energy(self, x):
        h = self.beta * torch.einsum("bld,dh->blh", x, self.W)
        return self.activation_energy(h)
    
    def forward(self, x):
        return value_and_grad(self.energy, x)

class SoftmaxModernHopfield(BaseModernHopfield):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 use_bias = False,
                 beta = None):
        super().__init__(in_dim,
                         hidden_dim,
                         use_bias = use_bias,
                         beta = beta)
        
    def _init_activation(self, beta):
        beta = torch.tensor(0.01) if beta is None else torch.tensor(beta)
        self.beta = torch.nn.Parameter(beta)
        self.activation = torch.nn.Identity()

    def activation_energy(self, x):
        energy = self.activation(x)
        energy = torch.logsumexp(energy, dim = -1)
        return -1 / self.beta * energy.sum()
        
class EnergyTransformer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 n_heads,
                 context_length = 0,
                 n_iters_default = 12,
                 alpha = 0.1,
                 beta = None,
                 hopfield_type = "relu",
                 use_positional_embedding = True,
                 norm = torch.nn.LayerNorm):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.context_length = context_length
        self.n_iters = n_iters_default
        self.alpha = alpha
        if use_positional_embedding and context_length:
            self.pos_embedding = torch.nn.Parameter(torch.randn(context_length, in_dim))
        else:
            self.pos_embedding = 0

        self.mha = EnergyMHA(self.in_dim, self.n_heads, beta = beta)
        self.hopfield = SoftmaxModernHopfield(self.in_dim, self.hidden_dim, beta = beta) if hopfield_type == "softmax" else BaseModernHopfield(self.in_dim, self.hidden_dim, beta = beta)
        self.norm = norm(self.in_dim)

    def energy(self, x):
        mha_energy = self.mha.energy(x)
        hopfield_energy = self.hopfield.energy(x)
        return mha_energy + hopfield_energy
    
    def forward_step(self, x):
        return value_and_grad(self.energy, x)

    def forward(self, x, n_iters = None):
        if n_iters is None:
            n_iters = self.n_iters

        x = x + self.pos_embedding

        energies = []
        features = []
        for i in range(n_iters):
            g = self.norm(x)
            energy, step = self.forward_step(g)
            x = x - self.alpha * step

            energies.append(energy.detach().clone())
            features.append(x.detach().clone())
        return x, energies, features


if __name__ == "__main__":
    ## a very long test on image reconstruction (like the original paper)
    import torchvision
    import ssl
    import os
    import matplotlib.pyplot as plt
    from einops.layers.torch import Rearrange
    from itertools import chain
    from tqdm import tqdm
    ssl._create_default_https_context = ssl._create_unverified_context

    im2tensor = torchvision.transforms.ToTensor()

    def collate(x, im2tensor = im2tensor):
        x = [im2tensor(x_i[0]) for x_i in x]
        return torch.stack(x, dim = 0)
    def tensor2im(x):
        return torchvision.transforms.ToPILImage()(x)
    def save_im(x, path):
        tensor2im(x).save(path)

    def save_features(feature_list, patch_deembedder, depatcher):
        for i, x in enumerate(feature_list):
            x = patch_deembedder(x)
            x = depatcher(x)
            save_im(x[0], f"tmp/feature_{i}.png")
    
    # making a tmp folder to store the images
    os.makedirs("tmp/", exist_ok = True)

    cifar = torchvision.datasets.CIFAR100(root = "C:/Projects/", train = True, download = True)
    dataloader = torch.utils.data.DataLoader(cifar, 
                                             batch_size = 32, 
                                             shuffle = True,
                                             collate_fn = collate)

    patch_size = 8
    patch_dim = patch_size**2 * 3
    n_patches = 16 # adding 1 for CLS token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patcher = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_size, p2 = patch_size).to(device)
    depatcher = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = patch_size, p2 = patch_size, h = 32 // patch_size, w = 32 // patch_size).to(device)

    patch_embedder = torch.nn.Sequential(
        torch.nn.Linear(patch_dim, 256),
        torch.nn.LayerNorm(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256)).to(device)
    patch_deembedder = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.LayerNorm(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, patch_dim)).to(device)
    
    mask_token = torch.nn.Parameter(torch.randn(256)).to(device)

    model = EnergyTransformer(256, 128, 8, context_length = n_patches, n_iters_default = 12, alpha = 0.1).to(device)
    
    optimizer = torch.optim.Adam(chain(
                                       model.parameters(),
                                       patch_embedder.parameters(),
                                       patch_deembedder.parameters()
                                       ), 
                                 lr = 1e-3)
    criterion = torch.nn.MSELoss()

    losses = []
    for i, x in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        x = x.to(device)
        x_orig = x.clone()

        x = patcher(x)
        x = patch_embedder(x)
        x, energies, features = model(x)
        x = patch_deembedder(x)
        x = depatcher(x)

        loss = criterion(x, x_orig)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0:
            save_im(x_orig[0], f"tmp/{i}_orig.png")
            save_im(x[0], f"tmp/{i}_recon.png")
            save_features(features, patch_deembedder, depatcher)

    


    

