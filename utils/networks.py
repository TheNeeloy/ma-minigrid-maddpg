import torch.nn as nn
import torch.nn.functional as F
from torch import cat, stack, sqrt

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class ConvMLPNetwork(nn.Module):
    """
    Conv + MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ConvMLPNetwork, self).__init__()

        self.in_fn = nn.BatchNorm2d(3)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.mlpnet = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim)
        )

        self.apply(init_params)

    def forward(self, obss, actions=None, critic=False, debug=False):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        if not critic:
            x = obss

            if debug:
                print('----------')
                print('obss:')
                print(x)

            if len(x.shape) < 4:
                x = x.unsqueeze(0).transpose(1, 3).transpose(2, 3)
            else:
                x = x.transpose(1, 3).transpose(2, 3)

            x = self.in_fn(x)
            x = self.image_conv(x)

            x = x.reshape(x.shape[0], -1)

            if debug:
                print('----------')
                print('conv out:')
                print(x)

            out = self.mlpnet(x)

            if debug:
                print('----------')
                print('mlp out:')
                print(out)

            return out

        else:
            x = stack(obss)

            num_agents = x.shape[0]
            num_batches = x.shape[1]

            x = x.reshape(-1, *x.shape[-3:])
            x = x.transpose(1, 3).transpose(2, 3)
            x = self.in_fn(x)
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)
            x = x.reshape(num_agents, num_batches, x.shape[1])

            act = stack(actions)

            concat = cat((*x, *act), dim=1)

            out = self.mlpnet(concat)

            return out

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
