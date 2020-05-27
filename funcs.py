import random
import math
import numpy as np


def generate_signal(harm_num, max_amp, max_ph, lim_frequency, point_num, a, b) \
        -> (np.ndarray, np.ndarray):
    """
    Generates random signal with given parameters:
        harm_num - number of harmonics,
        max_amp - maximal amplitude value,
        max_ph - maximal phase value,
        lim_frequency - limit for frequency value,
        point_num - number of discrete points for generating signal,
        a - start value on X axis,
        b - end value on X axis
    Returns:
        Tuple of two numpy arrays for two axis
    """
    w_step = lim_frequency / harm_num
    rand_val = [
        (random.randint(0, max_amp), random.uniform(0, max_ph), w_step * (i + 1))
        for i in range(harm_num)
    ]

    time_axis = np.linspace(a, b, point_num)
    sign_axis = sum([np.array([val[0] * math.sin(2 * math.pi * (val[2] * t + val[1])) for t in time_axis])
                     for val in rand_val])

    return (time_axis, sign_axis)


def expected_value(arr):
    return round(sum(arr) / len(arr), 2)


def dispersion(arr):
    m_x = expected_value(arr)
    return round(sum([(x_i - m_x) ** 2 for x_i in arr]) / (len(arr) - 1), 2)


def corelation(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError('arr1 length must be equal to arr2 length')

    m_arr1 = expected_value(arr1)
    m_arr2 = expected_value(arr2)

    tau_range = np.arange(int(len(arr1) / 2))
    corelation_arr = np.array([])
    for tau in tau_range:
        r_12 = [(arr1[i] - m_arr1) * (arr2[i + tau] - m_arr2)
                for i in range(int(len(arr1)/2))]
        r_12 = sum(r_12) / (len(r_12) - 1)
        corelation_arr = np.append(corelation_arr, r_12)

    return (tau_range, corelation_arr)