import os
import glob


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def get_filename(folder_results, dataset_name, model_type, leakage_model, desync=False, gaussian_noise=False, time_warping=False):
    da_str = "_desync" if desync else ""
    da_str += "_gaussian_noise" if gaussian_noise else ""
    da_str += "_time_warping" if time_warping else ""

    file_count = 0
    for name in glob.glob(f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{da_str}_*.npz"):
        file_count += 1
    new_filename = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{da_str}_{file_count + 1}.npz"

    return new_filename
