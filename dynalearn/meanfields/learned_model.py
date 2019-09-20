from .ame import AME
from .pa import PA
from .mf import MF
import tqdm
from scipy.special import binom


def all_combinations(k, d):
    if d == 1:
        return [[k]]
    return [(*j, k - i) for i in range(k + 1) for j in all_combinations(i, d - 1)]


class LearnedModelAME(AME):
    def __init__(self, p_k, model, max_iter=100, tol=1e-3, verbose=1):
        self.model = model
        self.model.num_nodes = p_k.values.max()
        super(LearnedModelAME, self).__init__(self.model.num_states, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self, ):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        state_degree = np.array(all_combinations(k, self.s_dim))
        for k in self.p_k.values():
            adj = np.zeros((self.model.num_nodes, self.model.num_nodes))
            adj[1 : k + 1, 0] = 1
            adj[0, 1 : k + 1] = 1
            for s in state_degree:
                neighbors_states = np.concatenate([ss * np.ones(l) for ss, l in enumerate(s)])
                inputs = np.zeros(k + 1)
                inputs[1:] = neighbors_states
                for i in range(self.s_dim):
                    inputs[0] = i
                    p = self.model.predict(inputs, adj)[0]
                    for j in range(self.s_dim):
                        ltp[(i,j, k, *l)] = p[j]
        return ltp


class LearnedModelPA(PA):
    def __init__(self, p_k, model, max_iter=100, tol=1e-3, verbose=1):
        self.model = model
        self.model.num_nodes = p_k.values.max()
        super(LearnedModelPA, self).__init__(self.model.num_states, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self, ):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        if self.verbose:
            num_iter = int(
                d
                * np.sum(
                    [
                        binom(k + d - 1, d - 1)
                        if binom(k + d - 1, d - 1) < max_num_sample
                        else max_num_sample
                        for k in self.degree_class
                    ]
                )
            )
            pb = tqdm.tqdm(range(num_iter), "Computing model LTP")
        for k in self.p_k.values:
            state_degree = np.array(all_combinations(k, self.s_dim))
            adj = np.zeros((self.model.num_nodes, self.model.num_nodes))
            adj[1 : k + 1, 0] = 1
            adj[0, 1 : k + 1] = 1
            for s in state_degree:
                neighbors_states = np.concatenate([ss * np.ones(l) for ss, l in enumerate(s)])
                inputs = np.zeros(k + 1)
                inputs[1:] = neighbors_states
                for i in range(self.s_dim):
                    inputs[0] = i
                    p = self.model.predict(inputs, adj)[0]
                    for j in range(self.s_dim):
                        ltp[(i,j, k, *l)] = p[j]
        return ltp


class LearnedModelMF(MF):
    def __init__(self, p_k, model, max_iter=100, tol=1e-3, verbose=1):
        self.model = model
        self.model.num_nodes = p_k.values.max()
        super(LearnedModelMF, self).__init__(self.model.num_states, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self, ):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        state_degree = np.array(all_combinations(k, self.s_dim))
        for k in self.p_k.values():
            adj = np.zeros((self.model.num_nodes, self.model.num_nodes))
            adj[1 : k + 1, 0] = 1
            adj[0, 1 : k + 1] = 1
            for s in state_degree:
                neighbors_states = np.concatenate([ss * np.ones(l) for ss, l in enumerate(s)])
                inputs = np.zeros(k + 1)
                inputs[1:] = neighbors_states
                for i in range(self.s_dim):
                    inputs[0] = i
                    p = self.model.predict(inputs, adj)[0]
                    for j in range(self.s_dim):
                        ltp[(i,j, k, *l)] = p[j]
        return ltp
