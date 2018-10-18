from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_data_1 = int(1e5)
    num_data_2 = int(2e5)

    batch_step = 1000

    X1 = stats.norm.rvs(size=num_data_1, loc=0., scale=1)
    X2 = stats.norm.rvs(size=num_data_2, loc=0.5, scale=1.5)

    fig, axes = plt.subplots(1, 2)

    for n in np.logspace(1,5,10):
        n = int(n)

        x1 = np.random.choice(X1, size=n)
        x2 = np.random.choice(X2, size=n)
        ks_stat, p_value = stats.ks_2samp(x1, x2)

        # sampling x1, x2 in batches
        avg_ks_stat, avg_p_value = 0, 0
        i, j=0, 0
        while (i < n):
            xx1 = np.random.choice(x1, size=batch_step)
            # xx2 = np.random.choice(x2, size=batch_step)
            # partial_ks_stat, partial_p_value = stats.ks_2samp(xx1, xx2)
            partial_ks_stat, partial_p_value = stats.ks_2samp(xx1, x2)
            avg_ks_stat += partial_ks_stat
            avg_p_value += partial_p_value
            i += batch_step

        avg_ks_stat /= (i / batch_step)
        avg_p_value /= (i / batch_step)


        axes[0].plot(n, ks_stat, 'bo')
        axes[1].plot(n, p_value, 'bo')
        axes[0].plot(n, avg_ks_stat, 'rx')
        axes[1].plot(n, avg_p_value, 'rx')

    axes[0].set_xscale('log')
    axes[1].set_xscale('log')

    plt.show()



if __name__ == '__main__':
    main()
