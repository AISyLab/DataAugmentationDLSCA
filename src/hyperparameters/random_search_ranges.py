import random


def hp_list(model_type):
    if model_type == "mlp":
        hp = {
            "neurons": random.choice([20, 40, 50, 100, 150, 200, 300, 400]),
            "batch_size": random.choice([100, 200, 400]),
            "layers": random.choice([1, 2, 3, 4, 5, 6]),
            "activation": random.choice(["elu", "selu", "relu"]),
            "learning_rate": random.choice([0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001]),
            "weight_init": random.choice(
                ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
            "optimizer": random.choice(["Adam", "RMSprop"])
        }
    else:
        hp = {
            "neurons": random.choice([20, 40, 50, 100, 150, 200, 300, 400]),
            "batch_size": random.choice([100, 200, 400]),
            "layers": random.choice([1, 2]),
            "filters": random.choice([4, 8, 12, 16]),
            "kernel_size": random.choice([10, 20, 30, 40]),
            "strides": random.choice([5, 10, 15, 20]),
            "pool_type": random.choice(["Average", "Max"]),
            "pool_size": random.choice([2]),
            "conv_layers": random.choice([1, 2, 3, 4]),
            "activation": random.choice(["elu", "selu", "relu"]),
            "learning_rate": random.choice([0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001]),
            "weight_init": random.choice(
                ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
            "optimizer": random.choice(["Adam", "RMSprop"])
        }
        hp["pool_strides"] = hp["pool_size"]
        conv_stride_options = [1, 2, 3, 4, 5, 10, 15, 20]
        possible_stride_options = []
        for i, st in enumerate(conv_stride_options):
            if st <= hp["kernel_size"]:
                possible_stride_options.append(st)
        hp["strides"] = random.choice(possible_stride_options)

    return hp
