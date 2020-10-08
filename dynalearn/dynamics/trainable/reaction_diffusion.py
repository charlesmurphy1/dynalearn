import numpy as np
import torch

from dynalearn.dynamics.reaction_diffusion import (
    ReactionDiffusion,
    WeightedReactionDiffusion,
    MultiplexReactionDiffusion,
    WeightedMultiplexReactionDiffusion,
)
from dynalearn.nn.models import (
    ReactionDiffusionGNN,
    ReactionDiffusionWGNN,
    ReactionDiffusionMGNN,
    ReactionDiffusionWMGNN,
)
from dynalearn.config import Config


class SimpleTrainableReactionDiffusion(ReactionDiffusion):
    def __init__(self, config=None, **kwargs):
        self.config = config or Config(**kwargs)
        ReactionDiffusion.__init__(self, config, config.num_states)
        self.nn = ReactionDiffusionGNN(config)
        # self.window_size = config.window_size
        # self.window_step = config.window_step

    def reaction(self, x):
        return x

    def diffusion(self, x):
        return x

    def initial_state(self):
        x = np.random.randn(self.num_nodes, self.num_states, self.window_size)
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
        assert x.shape[2] == self.window_size
        x = self.nn.transformers["t_inputs"].forward(x)
        g = self.nn.transformers["t_networks"].forward(self.network)
        y = self.nn.transformers["t_targets"].backward(self.nn.forward(x, g))
        return y.detach().numpy()


class WeightedTrainableReactionDiffusion(
    SimpleTrainableReactionDiffusion, WeightedReactionDiffusion
):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        WeightedReactionDiffusion.__init__(self, config, config.num_states)
        SimpleTrainableReactionDiffusion.__init__(self, config=config, **kwargs)
        self.nn = ReactionDiffusionWGNN(config)


class MultiplexTrainableReactionDiffusion(
    SimpleTrainableReactionDiffusion, MultiplexReactionDiffusion
):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        MultiplexReactionDiffusion.__init__(self, config, config.num_states)
        SimpleTrainableReactionDiffusion.__init__(self, config=config, **kwargs)
        self.nn = ReactionDiffusionMGNN(config)


class WeightedMultiplexTrainableReactionDiffusion(
    SimpleTrainableReactionDiffusion, WeightedMultiplexReactionDiffusion
):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        WeightedMultiplexReactionDiffusion.__init__(self, config, config.num_states)
        SimpleTrainableReactionDiffusion.__init__(self, config=config, **kwargs)
        self.nn = ReactionDiffusionWMGNN(config)


def TrainableReactionDiffusion(config=None, **kwargs):
    config = config or Config(**kwargs)
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False

    if is_weighted and is_multiplex:
        return WeightedMultiplexTrainableReactionDiffusion(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedTrainableReactionDiffusion(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexTrainableReactionDiffusion(config=config, **kwargs)
    else:
        return SimpleTrainableReactionDiffusion(config=config, **kwargs)
