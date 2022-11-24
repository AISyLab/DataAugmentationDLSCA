import numpy as np
import random


def get_original_batch(x_profiling, y_profiling, batch_count, batch_prof_count, batch_size):
    x_mini_batch = x_profiling[batch_prof_count * batch_size:batch_prof_count * batch_size + batch_size]
    y_mini_batch = y_profiling[batch_prof_count * batch_size:batch_prof_count * batch_size + batch_size]

    # print(f"prof({batch_count}) - {batch_prof_count * batch_size}:{batch_prof_count * batch_size + batch_size}")

    return x_mini_batch, y_mini_batch


def get_augmented_batch(x_profiling, y_profiling, ns, batch_size, batches_rnd, batch_count, data_augmentation_per_epoch, desync,
                        desync_level_augmentation, gaussian_noise, std_augmentation, model_name):
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
    std = std_dict[desync_level_augmentation]
    mean = 0

    # take a random batch from training traces
    x_mini_batch_augmented = np.zeros((batch_size, ns))
    if data_augmentation_per_epoch:
        rnd = random.randint(0, len(x_profiling) - batch_size)
    else:
        rnd = batches_rnd[batch_count]

    x_mini_batch = x_profiling[rnd:rnd + batch_size]
    x_mini_batch = x_mini_batch.reshape(x_mini_batch.shape[0], x_mini_batch.shape[1])

    shifts = None
    if not data_augmentation_per_epoch:
        # add desynchronization to mini-batch
        normal_dist_numbers = np.random.normal(mean, std, x_mini_batch.shape[0])
        normal_dist_numbers_int = np.round(normal_dist_numbers)
        shifts = np.array(
            [int(s) + int(desync_level_augmentation / 2) for s in
             normal_dist_numbers_int])  # add desync_level / 2 to only have positive shifts
        shifts[shifts < 0] = int(desync_level_augmentation / 2)
        shifts[shifts > desync_level_augmentation] = int(desync_level_augmentation / 2)

    # print(f"augmented({batch_count}) - {rnd}:{rnd + batch_size}")

    if desync:

        if data_augmentation_per_epoch:
            # add desynchronization to mini-batch
            normal_dist_numbers = np.random.normal(mean, std, x_mini_batch.shape[0])
            normal_dist_numbers_int = np.round(normal_dist_numbers)
            shifts = np.array(
                [int(s) + int(desync_level_augmentation / 2) for s in
                 normal_dist_numbers_int])  # add desync_level / 2 to only have positive shifts
            shifts[shifts < 0] = int(desync_level_augmentation / 2)
            shifts[shifts > desync_level_augmentation] = int(desync_level_augmentation / 2)

        for trace_index in range(batch_size):
            x_mini_batch_augmented[trace_index] = x_mini_batch[trace_index]
            x_mini_batch_augmented[trace_index][0:ns - shifts[trace_index]] = x_mini_batch[trace_index][shifts[trace_index]:ns]
            x_mini_batch_augmented[trace_index][ns - shifts[trace_index]:ns] = x_mini_batch[trace_index][0:shifts[trace_index]]

        x_mini_batch = x_mini_batch_augmented

    if gaussian_noise:
        # define statistical parameters
        mean = 0
        std = std_augmentation

        # add gaussian noise to batch
        noise = np.random.normal(mean, std, np.shape(x_mini_batch))
        x_mini_batch = np.add(x_mini_batch, noise)

    y_mini_batch = y_profiling[rnd:rnd + batch_size]

    # return batch
    if model_name == "cnn":
        x_mini_batch = x_mini_batch.reshape((x_mini_batch.shape[0], x_mini_batch.shape[1], 1))

    return x_mini_batch, y_mini_batch


def generate_data_augmentation(x_profiling, y_profiling, batch_size, model_name, n_batches_prof, n_batches_augmented,
                               desync_level_augmentation, std_augmentation, data_augmentation_per_epoch=True,
                               augmented_traces_only=False, desync=False, gaussian_noise=False):
    ns = len(x_profiling[0])
    batch_count = 0
    batch_prof_count = 0

    batches_prof_id = np.random.randint(0, 1, n_batches_prof)
    batches_augmented_id = np.random.randint(1, 2, n_batches_augmented)
    if augmented_traces_only:
        batches_id = np.random.permutation(np.array(batches_augmented_id))
    else:
        batches_id = np.concatenate((batches_prof_id, batches_augmented_id), axis=0)
        batches_id = np.random.permutation(np.array(batches_id))

    batches_rnd = None
    if not data_augmentation_per_epoch:
        if augmented_traces_only:
            batches_rnd = np.random.randint(0, len(x_profiling) - batch_size, n_batches_augmented)
        else:
            batches_rnd = np.random.randint(0, len(x_profiling) - batch_size, n_batches_augmented + n_batches_prof)

    while True:
        if batches_id[batch_count] == 0:
            x_mini_batch, y_mini_batch = get_original_batch(x_profiling, y_profiling, batch_count, batch_prof_count, batch_size)

            batch_prof_count += 1
            if batch_prof_count == n_batches_prof:
                batch_prof_count = 0
        else:
            x_mini_batch, y_mini_batch = get_augmented_batch(x_profiling, y_profiling, ns, batch_size, batches_rnd, batch_count,
                                                             data_augmentation_per_epoch, desync, desync_level_augmentation, gaussian_noise,
                                                             std_augmentation, model_name)

        batch_count += 1
        if batch_count == len(batches_id):
            batch_count = 0
            batches_id = np.random.permutation(np.array(batches_id))

        yield x_mini_batch, y_mini_batch

#
# def generate_data_augmentation(x_profiling, y_profiling, batch_size, model_name, n_batches_prof, n_batches_augmented,
#                                desync_level_augmentation, std_augmentation, data_augmentation_per_epoch=True,
#                                augmented_traces_only=False, desync=False, gaussian_noise=False):
#     ns = len(x_profiling[0])
#
#     batch_count = 0
#     batch_prof_count = 0
#
#     batches_prof_id = np.random.randint(0, 1, n_batches_prof)
#     batches_augmented_id = np.random.randint(1, 2, n_batches_augmented)
#     if augmented_traces_only:
#         batches_id = np.random.permutation(np.array(batches_augmented_id))
#     else:
#         batches_id = np.concatenate((batches_prof_id, batches_augmented_id), axis=0)
#         batches_id = np.random.permutation(np.array(batches_id))
#
#     batches_rnd = None
#     if not data_augmentation_per_epoch:
#         if augmented_traces_only:
#             batches_rnd = np.random.randint(0, len(x_profiling) - batch_size, n_batches_augmented)
#         else:
#             batches_rnd = np.random.randint(0, len(x_profiling) - batch_size, n_batches_augmented + n_batches_prof)
#
#     while True:
#
#         if batches_id[batch_count] == 0:
#
#             x_mini_batch = x_profiling[batch_prof_count * batch_size:batch_prof_count * batch_size + batch_size]
#             y_mini_batch = y_profiling[batch_prof_count * batch_size:batch_prof_count * batch_size + batch_size]
#
#             # print(f"prof({batch_count}) - {batch_prof_count * batch_size}:{batch_prof_count * batch_size + batch_size}")
#
#             batch_prof_count += 1
#
#             if batch_prof_count == n_batches_prof:
#                 batch_prof_count = 0
#
#         else:
#
#             # take a random batch from training traces
#             x_mini_batch_augmented = np.zeros((batch_size, ns))
#             if data_augmentation_per_epoch:
#                 rnd = random.randint(0, len(x_profiling) - batch_size)
#             else:
#                 rnd = batches_rnd[batch_count]
#             x_mini_batch = x_profiling[rnd:rnd + batch_size]
#             x_mini_batch = x_mini_batch.reshape(x_mini_batch.shape[0], x_mini_batch.shape[1])
#
#             # print(f"augmented({batch_count}) - {rnd}:{rnd + batch_size}")
#
#             if desync:
#                 # parameters for Gaussian distribution
#                 std_dict = {
#                     25: 3.5,
#                     50: 7,
#                     75: 11.5,
#                     100: 14,
#                     125: 17.5,
#                     150: 21,
#                     175: 24.5,
#                     200: 28
#                 }
#                 std = std_dict[desync_level_augmentation]
#                 mean = 0
#
#                 # add desynchronization to mini-batch
#                 normal_dist_numbers = np.random.normal(mean, std, x_mini_batch.shape[0])
#                 normal_dist_numbers_int = np.round(normal_dist_numbers)
#                 shifts = np.array(
#                     [int(s) + int(desync_level_augmentation / 2) for s in
#                      normal_dist_numbers_int])  # add desync_level / 2 to only have positive shifts
#                 shifts[shifts < 0] = int(desync_level_augmentation / 2)
#                 shifts[shifts > desync_level_augmentation] = int(desync_level_augmentation / 2)
#
#                 for trace_index in range(batch_size):
#                     x_mini_batch_augmented[trace_index] = x_mini_batch[trace_index]
#                     if shifts[trace_index] > 0:
#                         x_mini_batch_augmented[trace_index][0:ns - shifts[trace_index]] = x_mini_batch[trace_index][shifts[trace_index]:ns]
#                         x_mini_batch_augmented[trace_index][ns - shifts[trace_index]:ns] = x_mini_batch[trace_index][0:shifts[trace_index]]
#                     else:
#                         x_mini_batch_augmented[trace_index][0:abs(shifts[trace_index])] = x_mini_batch[trace_index][
#                                                                                           ns - abs(shifts[trace_index]):ns]
#                         x_mini_batch_augmented[trace_index][abs(shifts[trace_index]):ns] = x_mini_batch[trace_index][
#                                                                                            0:ns - abs(shifts[trace_index])]
#
#                 x_mini_batch = x_mini_batch_augmented
#
#             if gaussian_noise:
#                 # define statistical parameters
#                 mean = 0
#                 std = std_augmentation
#
#                 # add gaussian noise to batch
#                 noise = np.random.normal(mean, std, np.shape(x_mini_batch))
#                 x_mini_batch = np.add(x_mini_batch, noise)
#
#             y_mini_batch = y_profiling[rnd:rnd + batch_size]
#
#             # return batch
#             if model_name == "cnn":
#                 x_mini_batch = x_mini_batch.reshape((x_mini_batch.shape[0], x_mini_batch.shape[1], 1))
#
#         batch_count += 1
#         if batch_count == len(batches_id):
#             batch_count = 0
#             batches_id = np.random.permutation(np.array(batches_id))
#
#         yield x_mini_batch, y_mini_batch
