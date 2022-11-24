import sys
from src.datasets.load_ascadr import *
from src.datasets.load_dpav42 import *
from src.preprocess.generate_hiding_coutermeasures import *
from src.neural_networks.models import *
from src.metrics.guessing_entropy import *
from src.metrics.perceived_information import *
from src.hyperparameters.random_search_ranges import *
from src.datasets.paths import *
from src.training.train import *
from src.utils.utils import *
import gc
import os
import random
import argparse

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.path.append("/home/nfs/gperin/paper_1_data_augmentation_paper")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", help="Dataset Name")
    parser.add_argument("--leakage_model", help="Leakage Model")
    parser.add_argument("--desync", help="desync", default=0)
    parser.add_argument("--desync_level", help="desync_level", default=0)
    parser.add_argument("--desync_level_augmentation", help="desync_level_augmentation", default=0)
    parser.add_argument("--gaussian_noise", help="gaussian_noise", default=0)
    parser.add_argument("--std", help="std", default=0)
    parser.add_argument("--std_augmentation", help="std_augmentation", default=0)
    parser.add_argument("--file_id", help="file_id")
    parser.add_argument("--n_prof", help="n_prof")
    parser.add_argument("--n_augmented", help="n_augmented")
    parser.add_argument("--data_augmentation_per_epoch", help="data_augmentation_per_epoch")
    parser.add_argument("--augmented_traces_only", help="augmented_traces_only")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    leakage_model = args.leakage_model
    desync = True if int(args.desync) == 1 else False
    desync_level = int(args.desync_level)
    desync_level_augmentation = int(args.desync_level_augmentation)
    gaussian_noise = True if int(args.gaussian_noise) == 1 else False
    std = float(args.std)
    std_augmentation = float(args.std_augmentation)
    file_id = int(args.file_id)
    n_prof = int(args.n_prof)
    n_augmented = int(args.n_augmented)

    """ Define if augmented traces are different (True) for each epoch """
    data_augmentation_per_epoch = True if int(args.data_augmentation_per_epoch) == 1 else False
    """ Define if original traces are also included during training (False)"""
    augmented_traces_only = True if int(args.augmented_traces_only) == 1 else False

    trace_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme"
    folder_results = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_1_data_augmentation_results/random_search"
    if data_augmentation_per_epoch:
        if augmented_traces_only:
            save_results = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_1_data_augmentation_results/data_augmentation/data_augmentation_per_epoch/augmented_traces_only"
        else:
            save_results = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_1_data_augmentation_results/data_augmentation/data_augmentation_per_epoch/augmented_and_original_traces"
    else:
        if augmented_traces_only:
            save_results = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_1_data_augmentation_results/data_augmentation/data_augmentation_same_all_epochs/augmented_traces_only"
        else:
            save_results = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_1_data_augmentation_results/data_augmentation/data_augmentation_same_all_epochs/augmented_and_original_traces"

    # dataset_name = "ascad-variable"
    # leakage_model = "ID"
    # desync = True
    # desync_level = 25
    # desync_level_augmentation = 12
    # gaussian_noise = False
    # std = 1
    # std_augmentation = 1
    # file_id = 521
    # n_prof = 200000
    # n_augmented = 20000
    # trace_folder = "D:/traces"
    # folder_results = "D:/postdoc/paper_data_augmentation/random_search"
    # data_augmentation_per_epoch = False
    # augmented_traces_only = True
    #
    # if data_augmentation_per_epoch:
    #     if augmented_traces_only:
    #         save_results = "D:/postdoc/paper_data_augmentation/data_augmentation/data_augmentation_per_epoch/augmented_traces_only"
    #     else:
    #         save_results = "D:/postdoc/paper_data_augmentation/data_augmentation/data_augmentation_per_epoch/augmented_and_original_traces"
    # else:
    #     if augmented_traces_only:
    #         save_results = "D:/postdoc/paper_data_augmentation/data_augmentation/data_augmentation_same_all_epochs/augmented_traces_only"
    #     else:
    #         save_results = "D:/postdoc/paper_data_augmentation/data_augmentation/data_augmentation_same_all_epochs/augmented_and_original_traces"

    model_type = "cnn"

    dataset_parameters = None
    class_name = None

    if dataset_name == "dpa_v42":
        dataset_parameters = {
            "n_profiling": n_prof,
            "n_profiling_augmented": n_augmented,
            "n_attack": 5000,
            "n_attack_ge": 3000,
            "target_byte": 12,
            "npoi": 2000,
            "epochs": 100
        }
        class_name = ReadDPAV42
    if dataset_name == "ascad-variable":
        dataset_parameters = {
            "n_profiling": n_prof,
            "n_profiling_augmented": n_augmented,
            "n_attack": 5000,
            "n_attack_ge": 3000,
            "target_byte": 2,
            "npoi": 1400,
            "epochs": 100
        }
        class_name = ReadASCADr

    """ Create dataset """
    dataset = class_name(
        dataset_parameters["n_profiling"],
        dataset_parameters["n_attack"],
        file_path=get_dataset_filepath(trace_folder, dataset_name, dataset_parameters["npoi"]),
        target_byte=dataset_parameters["target_byte"],
        leakage_model=leakage_model,
        first_sample=0,
        number_of_samples=dataset_parameters["npoi"]
    )

    """ Add hiding countermeasures """
    if desync:
        dataset = make_desync(dataset, desync_level)
    if gaussian_noise:
        dataset = make_gaussian_noise(dataset, std)

    """ Rescale and reshape (if CNN) """
    dataset.rescale(True if model_type == "cnn" else False)

    """ Generate key guessing table """
    labels_key_guess = dataset.labels_key_hypothesis_attack

    random.seed(None)
    np.random.seed(None)

    """ Run training with augmentation """

    """ get hyperparameters """
    if desync:
        filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_desync_level_{desync_level}_desync_level_augmentation_0_desync_{file_id}.npz"
    else:
        filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_std_{desync_level}_std_augmentation_0_gaussian_noise_{file_id}.npz"

    if os.path.exists(filepath):
        npz_file = np.load(filepath, allow_pickle=True)
        hp_values = dict(enumerate(npz_file["hp_values"].flatten()))[0]
        print(hp_values)

        n_batches_prof = int(dataset_parameters["n_profiling"] / hp_values["batch_size"])
        n_batches_augmented = int(dataset_parameters["n_profiling_augmented"] / hp_values["batch_size"])
        steps_per_epoch = n_batches_augmented if augmented_traces_only else n_batches_prof + n_batches_augmented

        """ Create model """
        baseline_model = cnn(dataset.classes, dataset.ns, hp_values) if model_type == "cnn" else mlp(dataset.classes, dataset.ns,
                                                                                                     hp_values)
        """ Train model """
        model, history = train_model_augmentation(baseline_model, model_type, dataset, dataset_parameters["epochs"],
                                                  hp_values["batch_size"],
                                                  steps_per_epoch, n_batches_prof, n_batches_augmented, desync_level_augmentation,
                                                  std_augmentation,
                                                  data_augmentation_per_epoch=data_augmentation_per_epoch,
                                                  augmented_traces_only=augmented_traces_only, desync=desync, gaussian_noise=gaussian_noise)

        """ Compute guessing entropy and perceived information """
        predictions = model.predict(dataset.x_attack)
        GE, NT = guessing_entropy(predictions, labels_key_guess, dataset.correct_key, dataset_parameters["n_attack_ge"])
        PI = information(predictions, dataset.attack_labels, dataset.classes)

        """ Save results """
        new_filename = get_filename_augmentation(save_results, dataset_name, model_type, leakage_model, desync_level,
                                                 desync_level_augmentation, std, std_augmentation, file_id,
                                                 dataset_parameters["n_profiling"], dataset_parameters["n_profiling_augmented"],
                                                 desync=desync, gaussian_noise=gaussian_noise)
        np.savez(new_filename,
                 GE=GE,
                 NT=NT,
                 PI=PI,
                 hp_values=hp_values,
                 history=history.history,
                 dataset=dataset_parameters,
                 desync=desync,
                 desync_level=desync_level,
                 desync_level_augmentation=desync_level_augmentation,
                 gaussian_noise=gaussian_noise,
                 std=std,
                 std_augmentation=std_augmentation
                 )

        del model
        gc.collect()
