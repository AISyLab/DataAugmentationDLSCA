from numpy.random import RandomState
from tqdm import tqdm
from tsaug import TimeWarp
import numpy as np


def make_desync(dataset):

    print("adding desynchronization countermeasure")
    # set fixed seed to allow reproducibility
    np.random.seed(12345)

    # parameters for Gaussian distribution
    mean = 0
    std = 7

    # add desynchronization to profiling traces
    nums = np.random.normal(mean, std, dataset.x_profiling.shape[0])
    bins = np.linspace(-25, 25, 50, dtype='int')
    digitized = bins[np.digitize(np.squeeze(nums.reshape(1, -1)), bins) - 1].reshape(len(nums), -1)
    shifts_profiling = [s[0] + 25 for s in digitized]  # add 25 to only have positive shifts

    dataset.x_profiling = dataset.x_profiling.reshape(dataset.x_profiling.shape[0], dataset.x_profiling.shape[1])
    for trace_index in tqdm(range(dataset.n_profiling)):
        trace_tmp_shifted = np.zeros(dataset.ns)
        trace_tmp_shifted[0:dataset.ns - int(shifts_profiling[trace_index])] = dataset.x_profiling[trace_index][
                                                                               int(shifts_profiling[trace_index]):dataset.ns]
        trace_tmp_shifted[dataset.ns - int(shifts_profiling[trace_index]):dataset.ns] = dataset.x_profiling[trace_index][
                                                                                        0:int(shifts_profiling[trace_index])]
        dataset.x_profiling[trace_index] = trace_tmp_shifted

    # add desynchronization to attack traces
    nums = np.random.normal(mean, std, dataset.x_attack.shape[0])
    bins = np.linspace(-25, 25, 50, dtype='int')
    digitized = bins[np.digitize(np.squeeze(nums.reshape(1, -1)), bins) - 1].reshape(len(nums), -1)
    shifts_attack = [s[0] + 25 for s in digitized]  # add 25 to only have positive shifts

    dataset.x_attack = dataset.x_attack.reshape(dataset.x_attack.shape[0], dataset.x_attack.shape[1])
    for trace_index in tqdm(range(dataset.n_attack)):
        trace_tmp_shifted = np.zeros(dataset.ns)
        trace_tmp_shifted[0:dataset.ns - int(shifts_attack[trace_index])] = dataset.x_attack[trace_index][
                                                                            int(shifts_attack[trace_index]):dataset.ns]
        trace_tmp_shifted[dataset.ns - int(shifts_attack[trace_index]):dataset.ns] = dataset.x_attack[trace_index][
                                                                                     0:int(shifts_attack[trace_index])]
        dataset.x_attack[trace_index] = trace_tmp_shifted
    return dataset


def make_gaussian_noise(dataset):
    print("adding gaussian noise countermeasure")

    # parameters for Gaussian distribution
    mean = 0
    std = 5

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


def make_time_warping(dataset):
    print("adding time warping countermeasure")
    # set fixed seed to allow reproducibility
    np.random.seed(12345)

    # add time warping to profiling traces
    dataset.x_profiling = TimeWarp(n_speed_change=200, max_speed_ratio=20, seed=12345).augment(dataset.x_profiling)

    # set fixed seed to allow reproducibility
    np.random.seed(67890)

    # add time warping to profiling traces
    dataset.x_attack = TimeWarp(n_speed_change=200, max_speed_ratio=20, seed=67890).augment(dataset.x_attack)

    return dataset
