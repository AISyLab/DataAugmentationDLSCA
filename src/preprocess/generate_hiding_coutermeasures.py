from numpy.random import RandomState
from tqdm import tqdm
from tsaug import TimeWarp
import numpy as np
import matplotlib.pyplot as plt


def make_desync(dataset, desync_level):
    print("adding desynchronization countermeasure")
    # set fixed seed to allow reproducibility
    np.random.seed(12345)

    # parameters for Gaussian distribution
    std_dict = {
        25: 3.5,
        50: 7,
        75: 11.5,
        100: 14,
        125: 17.5,
        150: 21,
        175: 24.5,
        200: 28
    }

    # parameters for Gaussian distribution
    mean = 0
    std = std_dict[desync_level]

    # add desynchronization to profiling traces
    normal_dist_numbers = np.random.normal(mean, std, dataset.x_profiling.shape[0])
    normal_dist_numbers_int = np.round(normal_dist_numbers)
    shifts_profiling = np.array([int(s) + int(desync_level / 2) for s in normal_dist_numbers_int])  # add desync_level / 2 to only have positive shifts
    shifts_profiling[shifts_profiling < 0] = int(desync_level / 2)
    shifts_profiling[shifts_profiling > desync_level] = int(desync_level / 2)

    plt.hist(shifts_profiling, bins=desync_level*2)
    plt.show()

    dataset.x_profiling = dataset.x_profiling.reshape(dataset.x_profiling.shape[0], dataset.x_profiling.shape[1])
    for trace_index in tqdm(range(dataset.n_profiling)):
        trace_tmp_shifted = np.zeros(dataset.ns)
        trace_tmp_shifted[0:dataset.ns - int(shifts_profiling[trace_index])] = dataset.x_profiling[trace_index][
                                                                               int(shifts_profiling[trace_index]):dataset.ns]
        trace_tmp_shifted[dataset.ns - int(shifts_profiling[trace_index]):dataset.ns] = dataset.x_profiling[trace_index][
                                                                                        0:int(shifts_profiling[trace_index])]
        dataset.x_profiling[trace_index] = trace_tmp_shifted

    normal_dist_numbers = np.random.normal(mean, std, dataset.x_attack.shape[0])
    normal_dist_numbers_int = np.round(normal_dist_numbers)
    shifts_attack = np.array(
        [int(s) + int(desync_level / 2) for s in normal_dist_numbers_int])  # add desync_level / 2 to only have positive shifts
    shifts_attack[shifts_attack < 0] = int(desync_level / 2)
    shifts_attack[shifts_attack > desync_level] = int(desync_level / 2)

    dataset.x_attack = dataset.x_attack.reshape(dataset.x_attack.shape[0], dataset.x_attack.shape[1])
    for trace_index in tqdm(range(dataset.n_attack)):
        trace_tmp_shifted = np.zeros(dataset.ns)
        trace_tmp_shifted[0:dataset.ns - int(shifts_attack[trace_index])] = dataset.x_attack[trace_index][
                                                                            int(shifts_attack[trace_index]):dataset.ns]
        trace_tmp_shifted[dataset.ns - int(shifts_attack[trace_index]):dataset.ns] = dataset.x_attack[trace_index][
                                                                                     0:int(shifts_attack[trace_index])]
        dataset.x_attack[trace_index] = trace_tmp_shifted
    return dataset


def make_gaussian_noise(dataset, std):
    print("adding gaussian noise countermeasure")

    # parameters for Gaussian distribution
    mean = 0
    std = std

    # set fixed seed to allow reproducibility
    np.random.seed(12345)

    # add Gaussian noise to profiling traces
    noise = np.random.normal(mean, std, np.shape(dataset.x_profiling))
    dataset.x_profiling = np.add(dataset.x_profiling, noise)

    # set fixed seed to allow reproducibility
    np.random.seed(67890)

    # add Gaussian noise to profiling traces
    noise = np.random.normal(mean, std, np.shape(dataset.x_attack))
    dataset.x_attack = np.add(dataset.x_attack, noise)

    return dataset
