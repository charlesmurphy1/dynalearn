import numpy as np


class Sampler(object):
    def __init__(self):
        self.available_index = set()
        self.used_index = set()

        self.graphs = dict()
        self.inputs = dict()
        self.targets = dict()

        self.adj_container = dict()

        self.net_sample_size = 0


    def __call__(self):

        ind = random.sample(self.avail_index, 1)[0]
        
        ind = self.get_index()

        g_name = self.graphs[ind]
        
        adj = self.adj_container[g_name]
        inputs = self.inputs[ind]
        targets = self.targets[ind]
        weights = self.get_weights(inputs, adj)

        return [inputs, adj], targets, weights


    def add_samples(self, samples):
        g_name = samples[0]
        adj = samples[1]
        inputs = samples[2]
        targets = samples[3]

        num_samples = inputs.shape[0]

        begin = len(self.index)
        end = len(self.index) + num_samples

        self.avail_index = self.avail_index.union(range(begin, end))
        self.adj_container[g_name] = adj

        for i in range(num_samples):
            ind = begin + i
            self.graphs[ind] = g_name
            self.inputs[ind] = inputs[i]
            self.targets[ind] = targets[i]
            self.update_counts(inputs[i], adj)

        return

    def reset_index(self):
        if len(self.avail_index) == 0:
            self.avail_index = self.avail_index.union(self.used_index)
            self.used_index = set()
        return


    def get_index(self):
        raise NotImplementedError()

    def get_weights(self, states, adj):
        raise NotImplementedError()

    def update_counts(self, states, adj):
        raise NotImplementedError()




class SequentialSampler(Sampler):
    def __init__(self):
        super(SequentialSampler, self).__init__()

    def get_index(self):
        ind = self.avail_index.pop()
        self.used_index.add(ind)
        self.reset_index()
        return ind
        
    def get_weights(self, states, adj):
        states = data
        num_nodes = states.shape[0]
        return np.ones(num_nodes)

    def update_counts(self, states, adj):
        return


class RandomSampler(Sampler):
    def __init__(self):
        super(RandomSampler, self).__init__()

    def get_index(self):
        ind = random.sample(self.avail_index, 1)[0]
        self.avail_index.remove(ind)
        self.used_index.add(ind)
        self.reset_index()
        return ind
        
    def get_weights(self, states, adj):
        states = data
        num_nodes = states.shape[0]
        return np.ones(num_nodes)

    def update_counts(self, states, adj):
        return

class DegreeSampler(RandomSampler):
    def __init__(self, kmax, gamma=0, sample_from_weight=True):
        super(RandomSampler, self).__init__()
        self.kmax = kmax
        self.gamma = gamma
        self.sample_from_weight = sample_from_weight

        self.counts = np.zeros(kmax)
        self.avg_count = 1

    def update_counts(self, states, adj):
        k = np.sum(adj, axis=0).astype('int')

        num_nodes = len(states)

        for i in range(num_nodes): self.counts[k[i]] += 1
        
        self.avg_count = np.mean(self.counts[self.counts > 0])
        self.net_sample_size = np.sum(self.counts > 0)
        return 


    def get_weights(self, states, adj):
        k = np.sum(adj, axis=0).astype('int')
        count = self.counts[k]

        weights = self.avg_count / count**self.gamma
        if self.sample_from_weight:
            p = min(weights, 1)
            weights = np.random.binomial(1, p)

        return weights

class LocalStateSampler(RandomSampler):
    def __init__(self, kmax, num_states, gamma=0, sample_from_weight=True):
        super(RandomSampler, self).__init__()
        self.gamma = gamma
        self.kmax = kmax
        self.num_states = num_states
        self.sample_from_weight = sample_from_weight

        self.counts = np.zeros((kmax, kmax, num_states))
        self.avg_count = 1


    def update_counts(self, states, adj):
        k = np.sum(adj, axis=0).astype('int')
        l = np.matmul(states, adj).astype('int')
        s = states * 1

        num_nodes = len(states)

        for i in range(num_nodes): self.counts[k[i], l[i], s[i]] += 1

        self.avg_count = np.mean(self.counts[self.counts > 0])
        self.net_sample_size = np.sum(self.counts > 0)
        return 


    def get_weights(self, states, adj):
        k = np.sum(adj, axis=0).astype('int')
        l = np.matmul(states, adj).astype('int')
        s = states.astype('int')
        count = self.counts[k, l, s]

        weights = self.avg_count / count**self.gamma
        if self.sample_from_weight:
            p = min(weights, 1)
            weights = np.random.binomial(1, p)

        return weights


