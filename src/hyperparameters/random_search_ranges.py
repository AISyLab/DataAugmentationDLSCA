import itertools
import numpy as np


def hp_list(model_type):
    if model_type == "mlp":
        hp = {
            "layers": [1, 2, 3, 4, 5, 6],
            "neurons": [20, 40, 50, 100, 150, 200, 300, 400],
            "batch_size": [100, 200, 400],
            "activation": ["elu", "selu", "relu"],
            "learning_rate": [0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001],
            "weight_init": ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"],
        }
    else:
        hp = {
            "neurons": [20, 50, 100, 200],
            "batch_size": [100, 200, 400],
            "layers": [1, 2],
            "filters": [4, 8, 12, 16],
            "kernel_size": [10, 20, 30, 40],
            "strides": [5, 10, 15, 20],
            "pool_size": [2],
            "pool_strides": [2],
            "conv_layers": [1, 2, 3, 4],
            "activation": ["elu", "selu", "relu"],
            "learning_rate": [0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001],
            "weight_init": ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]
        }

    keys, value = zip(*hp.items())
    search_hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*value)]
    print(f"Total search space for {model_type}: {len(search_hp_combinations)}")

    hp_id = np.random.randint(len(search_hp_combinations))
    return search_hp_combinations[hp_id]
