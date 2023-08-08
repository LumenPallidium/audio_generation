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
        gamma = self.gamma(condition)
        if self.bias:
            beta = self.beta(condition)
            beta = beta.unsqueeze(1)
        else:
            beta = 0
        # assume x is (batch, L, dim) and gamma is (batch, dim)
        gamma = gamma.unsqueeze(1)
        return x * gamma + beta

    