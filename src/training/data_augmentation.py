import random
import numpy as np
from tsaug import TimeWarp
import random


def generate_data_augmentation(data_set_samples, data_set_labels, batch_size, model_name, n_batches_prof, n_batches_augmented,
                               desync=False, gaussian_noise=False, time_warping=False):
    ns = len(data_set_samples[0])

    r = random.randint(0, 100000)
    np.random.seed(r)

    batches_prof_id = np.random.randint(0, 1, n_batches_prof)
    batches_augmented_id = np.random.randint(1, 2, n_batches_augmented)
    batches_id = np.concatenate((batches_prof_id, batches_augmented_id), axis=0)
    batches_id = np.random.permutation(np.array(batches_id))

    batch_count = 0
    batch_prof_count = 0

    while True:

        if batches_id[batch_count] == 0:

            x_mini_batch = data_set_samples[batch_prof_count * batch_size:batch_prof_count * batch_size + batch_size]
            y_mini_batch = data_set_labels[batch_prof_count * batch_size:batch_prof_count * batch_size + batch_size]

            # print(f"prof({batch_count}) - {batch_prof_count * batch_size}:{batch_prof_count * batch_size + batch_size}")

            batch_prof_count += 1

            if batch_prof_count == n_batches_prof:
                batch_prof_count = 0

        else:

            # take a random batch from training traces
            x_train_augmented = np.zeros((batch_size, ns))
            rnd = random.randint(0, len(data_set_samples) - batch_size)
            x_mini_batch = data_set_samples[rnd:rnd + batch_size]

            x_mini_batch = x_mini_batch.reshape(x_mini_batch.shape[0], x_mini_batch.shape[1])

            # print(f"augmented({batch_count}) - {rnd}:{rnd + batch_size}")

            if desync:
                # parameters for Gaussian distribution
                mean = 0
                std = 7

                # add desynchronization to profiling traces
                nums = np.random.normal(mean, std, x_mini_batch.shape[0])
                bins = np.linspace(-25, 25, 50, dtype='int')
                digitized = bins[np.digitize(np.squeeze(nums.reshape(1, -1)), bins) - 1].reshape(len(nums), -1)
                shifts = [s[0] for s in digitized]

                # add random shift to batch
                for trace_index in range(batch_size):
                    x_train_augmented[trace_index] = x_mini_batch[trace_index]
                    if shifts[trace_index] > 0:
                        x_train_augmented[trace_index][0:ns - shifts[trace_index]] = x_mini_batch[trace_index][shifts[trace_index]:ns]
                        x_train_augmented[trace_index][ns - shifts[trace_index]:ns] = x_mini_batch[trace_index][0:shifts[trace_index]]
                    else:
                        x_train_augmented[trace_index][0:abs(shifts[trace_index])] = x_mini_batch[trace_index][
                                                                                     ns - abs(shifts[trace_index]):ns]
                        x_train_augmented[trace_index][abs(shifts[trace_index]):ns] = x_mini_batch[trace_index][
                                                                                      0:ns - abs(shifts[trace_index])]

                x_mini_batch = x_train_augmented

            if gaussian_noise:
                # define statistical parameters
                mean = 0
                std = 1

                # add gaussian noise to batch
                noise = np.random.normal(mean, std, np.shape(x_mini_batch))
                x_mini_batch = np.add(x_mini_batch, noise)

            if time_warping:
                # add time warping to batch
                x_mini_batch = TimeWarp(n_speed_change=200, max_speed_ratio=20).augment(x_mini_batch)

            y_mini_batch = data_set_labels[rnd:rnd + batch_size]

            # return batch
            if model_name == "cnn":
                x_mini_batch = x_mini_batch.reshape((x_mini_batch.shape[0], x_mini_batch.shape[1], 1))

        batch_count += 1
        if batch_count == len(batches_id):
            batch_count = 0

        yield x_mini_batch, y_mini_batch
