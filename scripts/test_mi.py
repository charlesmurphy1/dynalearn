import utilities as u
import numpy as np
import matplotlib.pyplot as plt


N = 100
delay = 1
num_states = 2
x = np.random.randint(0, num_states, N)
y = np.random.randint(0, num_states, N)

num_sample = 1

perturbed_MI = np.zeros(N)
Z = u.information(x, delay, num_states) / 2

for i in range(N):
	print(i)
	for s in range(num_sample):
		y_perturbed = y * 1
		index = np.random.choice(range(N), size=i, replace=False)
		y_perturbed[index] = x[index] * 1
		perturbed_MI[i] += u.mutual_information(x, y_perturbed, delay, num_states) / num_sample

# plt.plot(perturbed_MI, '-', color='#1f77b4')
plt.plot(perturbed_MI / Z, 'o', color='#1f77b4', alpha=0.3)
plt.ylabel('Normalized mutual information')
plt.xlabel('Number of common bits')
plt.yscale('log')
plt.show()


