import os
import glob


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def get_filename_augmentation(folder_results, dataset_name, model_type, leakage_model, desync_level, desync_level_augmentation, std,
                              std_augmentation, file_id, n_prof, n_augmented, desync=False, gaussian_noise=False):
    if gaussian_noise:
        file_count = 0
        for name in glob.glob(
                f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_std_{std}_std_augmentation_{std_augmentation}_n_prof_{n_prof}_n_augmented_{n_augmented}_gaussian_noise_file_id_{file_id}_*.npz"):
            file_count += 1
        new_filename = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_std_{std}_std_augmentation_{std_augmentation}_n_prof_{n_prof}_n_augmented_{n_augmented}_gaussian_noise_file_id_{file_id}_{file_count + 1}.npz"

    else:
        file_count = 0
        for name in glob.glob(
                f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_desync_level_{desync_level}_desync_level_augmentation_{desync_level_augmentation}_n_prof_{n_prof}_n_augmented_{n_augmented}_desync_file_id_{file_id}_*.npz"):
            file_count += 1
        new_filename = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_desync_level_{desync_level}_desync_level_augmentation_{desync_level_augmentation}_n_prof_{n_prof}_n_augmented_{n_augmented}_desync_file_id_{file_id}_{file_count + 1}.npz"

    return new_filename


def get_filename(folder_results, dataset_name, model_type, leakage_model, desync_level, desync_level_augmentation, std, std_augmentation,
                 desync=False, gaussian_noise=False):
    if gaussian_noise:
        file_count = 0
        for name in glob.glob(
                f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_std_{std}_std_augmentation_{std_augmentation}_gaussian_noise_*.npz"):
            file_count += 1
        new_filename = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_std_{std}_std_augmentation_{std_augmentation}_gaussian_noise_{file_count + 1}.npz"

    else:
        file_count = 0
        for name in glob.glob(
                f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_desync_level_{desync_level}_desync_level_augmentation_{desync_level_augmentation}_desync_*.npz"):
            file_count += 1
        new_filename = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_desync_level_{desync_level}_desync_level_augmentation_{desync_level_augmentation}_desync_{file_count + 1}.npz"

    return new_filename
