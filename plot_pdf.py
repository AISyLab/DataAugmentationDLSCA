import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# parameters for Gaussian distribution
std_dict = {
    12: 1.75,
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
desync_level = 25
mean = 0
std = std_dict[desync_level]

# add desynchronization to profiling traces
normal_dist_numbers = np.random.normal(mean, std, 20000)
normal_dist_numbers_int = np.round(normal_dist_numbers)
shifts_profiling = np.array(
    [int(s) + int(desync_level / 2) for s in normal_dist_numbers_int])  # add desync_level / 2 to only have positive shifts
shifts_profiling[shifts_profiling < 0] = int(desync_level / 2)
shifts_profiling[shifts_profiling > desync_level] = int(desync_level / 2)

sns.kdeplot(np.array(shifts_profiling), bw=0.5, label="$\delta_{hid} = $" + f"{desync_level}")

# parameters for Gaussian distribution

for desync_level_aug in [12, 25, 50]:
    # desync_level_aug = 12
    mean = 0
    std = std_dict[desync_level_aug]

    # add desynchronization to profiling traces
    normal_dist_numbers = np.random.normal(mean, std, 20000)
    normal_dist_numbers_int = np.round(normal_dist_numbers)
    shifts_profiling_aug = np.array(
        [int(s) + int(desync_level_aug / 2) + shifts_profiling[i] for i, s in
         enumerate(normal_dist_numbers_int)])  # add desync_level / 2 to only have positive shifts


    sns.kdeplot(np.array(shifts_profiling_aug), bw=0.5, label="$\delta_{hid}$ + $\delta_{aug} = $" + f"{desync_level} + " + f"{desync_level_aug}")

plt.ylabel("Density")
plt.xlabel("Shifts")
plt.legend(fontsize=8)
plt.xlim([0, 75])
plt.savefig("D:/postdoc/paper_data_augmentation/hid25.png", dpi=100)
