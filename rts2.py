import matplotlib.pyplot as plt
import numpy as np

import time
import concurrent.futures
from math import pi, cos, sin
from funcs import generate_signal

ampl = 20  # Max. amplitude - A
phi = 2 * pi  # Max. phase - phi
n = 10  # Number of harmonics - n
w = 2700  # Limit frequency - Wgr
points = 256  # Number of points - N
a, b = 0, 0.1  # Generation range

# Generating signal and creating graph
t, x = generate_signal(n, ampl, phi, w, points, a, b)


def dft(x):
    def coeff(p, k, N):
        res = cos((2 * pi * p * k) / N) - 1j * sin((2 * pi * p * k) / N)
        return complex(round(res.real, 2), round(res.imag, 2))

    p = np.arange(len(x))

    return np.array([sum([x[k] * coeff(p_i, k, len(x))
                          for k in range(len(x))]) for p_i in p])


def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])


def plot(function):
    plt.figure(figsize=(10, 5))
    plt.plot(function)
    plt.grid(True)
    plt.show()


def multi_plot(y1, y2, y3):
    plt.figure(figsize=(10, 5))
    plt.plot(y1, 'b', label="F_real")
    plt.plot(y2, 'k', label="F_im")
    plt.plot(y3, 'r', label="F sum")
    plt.grid(True)
    plt.legend(prop={'size': 16}, loc='upper right', borderaxespad=0.)
    plt.show()


#additional task
def add():
    F = np.zeros(points)
    F_new = np.zeros(points)
    F_real = np.zeros(points)
    F_im = np.zeros(points)

    F_real_w = np.zeros(points)
    F_im_w = np.zeros(points)

    # table
    w_coeff = np.zeros(shape=(points, points))
    for p in range(points):
        for k in range(points):
            w_coeff[p][k] = cos(2 * pi / points * p * k) + sin(2 * pi / points * p * k)

    # without
    start1 = time.time()
    for p in range(points):
        for k in range(points):
            F_real[p] += x[k] * cos(2 * pi / points * p * k)
            F_im[p] += x[k] * sin(2 * pi / points * p * k)
    for i in range(points):
        F[i] += F_real[i] + F_im[i]
    time1 = time.time() - start1
    print(f"without table: {time1}")

    plot(F)

    multi_plot(F_real, F_im, F)

    # with table
    start2 = time.time()
    for p in range(points):
        for k in range(points):
            F_real_w[p] += x[k] * w_coeff[p][k]
            F_im_w[p] += x[k] * w_coeff[p][k]
    for i in range(points):
        F_new[i] += F_real_w[i] + F_im_w[i]
    time2 = time.time() - start2

    print(f"with table: {time2}")

    multi_plot(F_real_w, F_im_w, F_new)


if __name__ == "__main__":
    add()