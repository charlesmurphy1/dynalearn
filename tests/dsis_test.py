import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dynalearn.dynamics.multidim_epidemic import DSISNetwork
import os

N = 1000
avgk = 10
M = N * avgk / 2
g = nx.gnm_random_graph(N, M)
print(g.number_of_nodes())
init_active = 0.1
init_num = 1
dt = 0.01
step = 0.01
T = 1
inf_rate = 0.2

D = 3
in_coupling = 0.5
out_coupling = 1.

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

        m_net = model_class(g, inf_rate, 1, D, in_coupling, out_coupling, 
                            init_active=init_active, init_num=init_num, dt=dt,
                            filename=file_states)
        avg, std = m_net.get_avg_activity()

        avg, err = m_net.get_avg_activity()
        t = [0]
        avg_act = [list(avg.values())]
        err_act = [list(err.values())]

        i = 0
        while(t[-1] < T and m_net.continu_simu):
            m_net.update(step=step, record=True)

            avg, err = m_net.get_avg_activity()

            t.append(m_net.t[-1])
            i += 1
            print(t[-1], avg)
            avg_act.append(list(avg.values()))
            err_act.append(list(err.values()))

        if save:
            m_net.close()

        t = np.array(t)
        avg_act = np.array(avg_act)
        err_act = np.array(err_act)

        for i, k in enumerate(list(avg.keys())):
            plt.plot(t, avg_act[:, i], label=rf"${k}$")

        # legend_linetype = [Line2D([0], [0], marker='s', linestyle='None',
        #                     color=mycolors["b"], lw=2, alpha=1),
        #                 Line2D([0], [0], marker='s', linestyle='None',
        #                     color=mycolors["o"], lw=2, alpha=1)
        #                ]
        plt.xlim([0, T])
        plt.ylim([0, 1])
        plt.legend(frameon=False, ncol=1, fontsize=10)
        plt.show()

plot_activity(DSISNetwork, "testdata/DSIS_states.b", "testdata/DSIS_net.txt")