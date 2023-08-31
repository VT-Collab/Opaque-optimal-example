import numpy as np
from matplotlib import pyplot as plt
import pickle

def get_data(filename):
    data = pickle.load(open(filename, 'rb'))
    return data

def process_file(filename):
    data = get_data(filename)
    total_states = 21 * 9
    fully_opaque_count = 0.
    rationally_opaque_count = 0.
    transparent_count = 0.
    for key in data:
        augmented_state, r_opaque, f_opaque = data[key]
        rationally_opaque_count += r_opaque
        fully_opaque_count += f_opaque
        transparent_count += 1 - max([r_opaque, f_opaque])
    r_perc = rationally_opaque_count / total_states
    f_perc = fully_opaque_count / total_states
    t_perc = transparent_count / total_states
    return (r_perc, f_perc, t_perc)

def plot_file(filename):
    data = get_data(filename)
    for key in data:
        augmented_state, r_opaque, f_opaque = data[key]
        if r_opaque:
            plt.plot(augmented_state[1], augmented_state[2], 'bo', markersize=10)
        if f_opaque:
            plt.plot(augmented_state[1], augmented_state[2], 'ro', markersize=5)
    plt.axis([-0.1, 2.1, 0, 1])
    plt.show()

# plots for the basic approach
T = range(5, 16)
LR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Zr = np.zeros((len(T), len(LR)))
Zf = np.zeros((len(T), len(LR)))
for idx, t in enumerate(T):
    for jdx, lr in enumerate(LR):
        filename = "basic-t-" + str(t) + "-lr-" + str(lr) + ".pkl"
        (r_perc, f_perc, t_perc) = process_file(filename)
        Zr[idx, jdx] = r_perc
        Zf[idx, jdx] = f_perc

# rationally opaque plot
fig, ax = plt.subplots()
cmap = ax.pcolormesh(np.transpose(Zr), cmap="Purples", vmin=0.0, vmax=1.0)
fig.colorbar(cmap)
plt.show()
# fully opaque plot
fig, ax = plt.subplots()
cmap = ax.pcolormesh(np.transpose(Zf), cmap="Oranges", vmin=0.0, vmax=1.0)
fig.colorbar(cmap)
plt.show()

# plots for the bayes human
T = range(5, 16)
Zr = np.array([0.] * len(T))
Zf = np.array([0.] * len(T))
for idx, t in enumerate(T):
    filename = "bayes-t-" + str(t) + ".pkl"
    (r_perc, f_perc, t_perc) = process_file(filename)
    Zr[idx] = r_perc
    Zf[idx] = f_perc
plt.plot(T, Zr)
plt.plot(T, Zf)
plt.axis([5, 15, 0, 1.0])
plt.show()

# plots for the memory human
T = range(5, 16)
Zr = np.array([0.] * len(T))
Zf = np.array([0.] * len(T))
for idx, t in enumerate(T):
    filename = "memory-t-" + str(t) + "-lr-0.3.pkl"
    (r_perc, f_perc, t_perc) = process_file(filename)
    Zr[idx] = r_perc
    Zf[idx] = f_perc
plt.plot(T, Zr)
plt.plot(T, Zf)

T = range(5, 16)
Zr = np.array([0.] * len(T))
Zf = np.array([0.] * len(T))
for idx, t in enumerate(T):
    filename = "memory-t-" + str(t) + "-lr-0.7.pkl"
    (r_perc, f_perc, t_perc) = process_file(filename)
    Zr[idx] = r_perc
    Zf[idx] = f_perc
plt.plot(T, Zr)
plt.plot(T, Zf)
plt.axis([5, 15, 0, 1.0])
plt.show()
