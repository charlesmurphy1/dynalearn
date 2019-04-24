import utilities as u
import numpy as np
import matplotlib.pyplot as plt

import dynalearn as dl

N = 1000
p = 0.01
inf_prob = 0.4
rec_prob = 0.8
T = 1000
num_states = 2

dynamics = dl.dynamics.SISDynamics(inf_prob, rec_prob, init_state=0.1)
graph = dl.graphs.ERGraph(N, p)
name, g = graph.generate()
dynamics.graph = g

TS = np.zeros((T, N))
for i in range(T):
	dynamics.update()
	TS[i, :] = dynamics.states

num_sample = 50
delays = [1, 2, 3, 4, 5, 6]
avg_mi = []
var_mi = []
for d in delays:
	_mi = []
	for i in range(num_sample):
		print(d, i)
		nodes = np.random.choice(range(N), 2, replace=False)
		x = u.mutual_information(TS[:, nodes[0]], TS[:, nodes[1]], d, num_states)
		_mi.append(x)
	avg_mi.append(np.mean(_mi))
	var_mi.append(np.std(_mi))
plt.errorbar(delays, avg_mi, yerr=var_mi, marker='o', linestyle='-')
plt.show()
	



