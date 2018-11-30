import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dynalearn.dynamics.nonmark_epidemic import SIkSNetwork
import os

N = 500
avgk = 10
M = N * avgk / 2
g = nx.gnm_random_graph(N, M)
print(g.number_of_nodes())
init_active = 0.1
dt = 0.01
step = 0.01
T = 10
K = 3
gamma = 0.
rate = 1.
save = True

def plot_activity(model_class, file_states, file_net, overwrite=True):
    ## Testing SIRNetwork sub-class

    if (not os.path.exists(file_states) or not os.path.exists(file_net) or overwrite):
        if save:
            net_file = open(file_net, "wb")
            nx.write_edgelist(g, net_file)
            net_file.close()
        else:
            file_states = None

        m_net = model_class(g, rate, K, init_active=init_active, dt=dt,
                             filename=file_states)

        avg, err = m_net.get_avg_activity()
        t = [0]
        avg_act = [list(avg.values())]
        err_act = [list(err.values())]


        while(t[-1] < T and m_net.continu_simu):
            m_net.update(step=step, record=True)

            avg, err = m_net.get_avg_activity()

            t.append(m_net.t[-1])
            avg_act.append(list(avg.values()))
            err_act.append(list(err.values()))

        if save:
            m_net.close()

        t = np.array(t)
        avg_act = np.array(avg_act)
        err_act = np.array(err_act)

        plt.plot(t, avg_act)
        plt.show()

plot_activity(SIkSNetwork, "testdata/SIkS_states.b", "testdata/SIkS_net.txt")