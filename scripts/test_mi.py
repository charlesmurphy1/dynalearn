import utilities as u
import numpy as np
import matplotlib.pyplot as plt

import dynalearn as dl

N = 10
p = 0.5
inf_prob = 0.08
rec_prob = 0.08
T = 10000
num_states = 2

dynamics = dl.dynamics.SISDynamics(inf_prob, rec_prob)
graph = dl.graphs.ERGraph(N, p)

name, g = graph.generate()
dynamics.graph = g
TS1 = np.zeros((T, N))
for i in range(T):
	dynamics.update()
	TS1[i, :] = dynamics.states

name, g = graph.generate()
dynamics.graph = g
TS2 = np.zeros((T, N))
dynamics.initialize_states()
for i in range(T):
	dynamics.update()
	TS2[i, :] = dynamics.states

p1, p2 = np.mean(TS1, axis=0), np.mean(TS2, axis=0)

mi = u.mutual_information(TS1, TS2, num_states)
inf1 = u.information(TS1, num_states)
inf2 = u.information(TS2, num_states)
# print(mi, inf1, inf2)
print(mi, inf1, inf2)
# print(TS1[:, 1], TS1[:, 3], x)
# plt.errorbar(delays, avg_mi, yerr=var_mi, marker='o', linestyle='-')
# plt.show()
	



