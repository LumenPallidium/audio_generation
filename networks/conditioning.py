import torch

class SqueezeExcite(torch.nn.Module):
    def __init__(self,
                 dim,
                 scale_factor = 2,
                 first_activation = torch.nn.ReLU(),
                 second_activation = torch.nn.Sigmoid()):
        super().__init__()
        self.scale_factor = scale_factor

        self.dim = dim
        self.hidden_dim = dim // scale_factor

        self.squeeze = torch.nn.Linear(dim, self.hidden_dim)
        self.excite = torch.nn.Linear(self.hidden_dim, dim)

        self.first_activation = first_activation
        self.second_activation = second_activation
    
    def forward(self, x):
        condition = self.first_activation(self.squeeze(x))
        condition = self.second_activation(self.excite(condition))
        return x * condition
    
class FiLM(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim = None,
                 bias = True,):
        super().__init__()
        self.dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim

        self.gamma = torch.nn.Linear(self.dim, self.out_dim)

        self.bias = bias
        if bias:
            self.beta = torch.nn.Linear(self.dim, self.out_dim)
        
    def forward(self, x, condition):
        if condition is None:
            return x
        gamma = self.gamma(condition)
        if self.bias:
            beta = self.beta(condition)
            beta = beta.unsqueeze(1)
        else:
            beta = 0
        # assume x is (batch, L, dim) and gamma is (batch, dim)
        gamma = gamma.unsqueeze(1)
        return x * gamma + beta

if __name__ == "__main__":
    # testing on generative MNIST
    from einops.layers.torch import Rearrange
    from torchvision import datasets, transforms
    from transformers import Transformer
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    transform = transforms.Compose([transforms.ToTensor()])

    mnist = datasets.MNIST('./data', train=True, download=True,
                            transform=transform)

    dim_latent = 16
    patch_size = 4
    n_epochs = 10

    patch_dim = patch_size ** 2
    num_patches = (28 // patch_size) ** 2
    effective_dim = patch_dim * num_patches

    class Gen(torch.nn.Module):
        def __init__(self,
                     dim = dim_latent,
                     patch_size = patch_size,
                     film_dim = 128,):
            super().__init__()

            patch_dim = patch_size ** 2
            num_patches = (28 // patch_size) ** 2
            effective_dim = patch_dim * num_patches
            
            self.first_block = torch.nn.Sequential(torch.nn.Conv1d(dim, 32, 7, padding = "same"),
                                                   torch.nn.GELU())
            self.film1 = FiLM(film_dim, 32)

            self.second_block = torch.nn.Sequential(torch.nn.Conv1d(32, 64, 3, padding = "same"),
                                                    torch.nn.GELU())
            self.film2 = FiLM(film_dim, 64)

            self.third_block = torch.nn.Sequential(torch.nn.Conv1d(64, 128, 3, padding = "same"),
                                                    torch.nn.GELU())
            self.film3 = FiLM(film_dim, 128)

            self.fourth_block = torch.nn.Sequential(torch.nn.Conv1d(128, 784, 3, padding = "same"),
                                                    torch.nn.GELU())
            self.film4 = FiLM(film_dim, 784)

            self.transformer = torch.nn.Sequential(
                Rearrange('b (l d) -> b l d', l = num_patches, d = patch_dim),
                Transformer(16, 1, 4, 4, context_x = num_patches),
                Rearrange('b (h w) (p1 p2) -> b (h p1) (w p2)', p1 = patch_size, p2 = patch_size, 
                          h = 28 // patch_size, w = 28 // patch_size),
            )

        def forward(self, x, condition = None):
            if len(x.shape) == 2:
                x = x.unsqueeze(-1)
            x = self.first_block(x)
            x = self.film1(x, condition)
            x = self.second_block(x)
            x = self.film2(x, condition)
            x = self.third_block(x)
            x = self.film3(x, condition)
            x = self.fourth_block(x)
            x = self.film4(x, condition)
            x = self.transformer(x.squeeze(-1))
            return x
                

    class Disc(torch.nn.Module):
        def __init__(self,
                     n_classes = 10,):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3, stride = 2, padding = 1),
                                             torch.nn.GELU(),
                                             torch.nn.Conv2d(32, 64, 7, stride = 2, padding = 3),
                                             torch.nn.GELU(),
                                             torch.nn.Conv2d(64, 128, 7, stride = 7, padding = 3),
                                             torch.nn.GELU())
            self.class_layer = torch.nn.Linear(128, n_classes)

        def forward(self, x):
            x = self.net(x).squeeze(-1).squeeze(-1)
            label = self.class_layer(x)
            label = torch.nn.functional.softmax(label, dim = -1)
            return x, label
        
    gen = Gen()
    disc = Disc()
                                             
    latent = torch.randn(1, 16)

    opt_g = torch.optim.Adam(gen.parameters(), lr = 1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr = 1e-3)

    for epoch in range(n_epochs):
        dataloader = DataLoader(mnist, batch_size = 32, shuffle = True)
        for data, label in tqdm(dataloader):
            opt_g.zero_grad()
            opt_d.zero_grad()

            latent = torch.randn(1, 16)
            x, label_est = disc(data.clone().requires_grad_())
            gens = gen(latent, x)

            y_disc = disc(gens.detach().clone().requires_grad_())[0]
            y = disc(gens)[0]

            loss_d = torch.nn.functional.cross_entropy(label_est, label)

            real_d_loss = -torch.minimum(x - 1, torch.zeros_like(x)).mean()
            fake_d_loss = -torch.minimum(-y_disc - 1, torch.zeros_like(y_disc)).mean()

            loss_d += (real_d_loss + fake_d_loss)
            loss_g = -y.mean()

            loss_d.backward(retain_graph = True)
            loss_g.backward()