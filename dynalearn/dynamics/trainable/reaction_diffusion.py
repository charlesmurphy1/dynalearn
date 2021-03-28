import numpy as np
import torch

from dynalearn.dynamics.reaction_diffusion import ReactionDiffusion
from dynalearn.nn.models import ReactionDiffusionGNN
from dynalearn.config import Config


class TrainableReactionDiffusion(ReactionDiffusion):
    def __init__(self, config=None, **kwargs):
        self.config = config or Config(**kwargs)
        ReactionDiffusion.__init__(self, config, config.num_states)
        self.nn = ReactionDiffusionGNN(config)

    def reaction(self, x):
        raise ValueError("This method is invalid for Trainable models")

    def diffusion(self, x):
        raise ValueError("This method is invalid for Trainable models")

    def initial_state(self):
        x = np.random.randn(self.num_nodes, self.num_states, self.lag)
        return self.nn.transformers["t_inputs"].backward(x)

    def is_dead(self):
        return False

    def sample(self, x):
        return self.predict(x)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)

        assert x.ndim == 3
        assert x.shape[1] == self.num_states
        assert x.shape[2] == self.lag
        x = self.nn.transformers["t_inputs"].forward(x)
        g = self.nn.transformers["t_networks"].forward(self.network)
        y = self.nn.transformers["t_targets"].backward(self.nn.forward(x, g))
        return y.detach().numpy()
