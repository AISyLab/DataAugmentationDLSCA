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

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.path.append("/home/nfs/gperin/paper_1_data_augmentation_paper")

if __name__ == "__main__":

    dataset_name = sys.argv[1]
    leakage_model = sys.argv[2]
    desync = True if int(sys.argv[3]) == 1 else False
    desync_level = int(sys.argv[4])
    desync_level_augmentation = int(sys.argv[5])
    gaussian_noise = True if int(sys.argv[6]) == 1 else False
    std = float(sys.argv[7])
    std_augmentation = float(sys.argv[8])
    trace_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme"
    folder_results = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_1_data_augmentation_results/random_search"

    # dataset_name = "ascad-variable"
    # leakage_model = "ID"
    # desync = False
    # desync_level = 0
    # desync_level_augmentation = 0
    # gaussian_noise = True
    # std = 1.2
    # std_augmentation = 0
    # trace_folder = "D:/traces"
    # folder_results = "D:/postdoc/paper_data_augmentation/random_search"

    dataset_parameters = None
    class_name = None

    model_type = "cnn"

    if dataset_name == "dpa_v42":
        dataset_parameters = {
            "n_profiling": 70000,
            "n_attack": 5000,
            "n_attack_ge": 3000,
            "target_byte": 12,
            "npoi": 2000,
            "epochs": 100
        }
        class_name = ReadDPAV42
    if dataset_name == "ascad-variable":
        dataset_parameters = {
            "n_profiling": 200000,
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

    """ Run random search """
    for search_index in range(100):
        """ generate hyperparameters """
        hp_values = hp_list(model_type)
        hp_values["seed"] = np.random.randint(1048576)
        print(hp_values)

        """ Create model """
        baseline_model = cnn(dataset.classes, dataset.ns, hp_values) if model_type == "cnn" else mlp(dataset.classes, dataset.ns,
                                                                                                     hp_values)
        """ Train model """
        model, history = train_model(baseline_model, dataset, dataset_parameters["epochs"], hp_values["batch_size"])

        """ Compute guessing entropy and perceived information """
        predictions = model.predict(dataset.x_attack)
        GE, NT = guessing_entropy(predictions, labels_key_guess, dataset.correct_key, dataset_parameters["n_attack_ge"])
        PI = information(predictions, dataset.attack_labels, dataset.classes)

        """ Save results """
        new_filename = get_filename(folder_results, dataset_name, model_type, leakage_model, desync_level, 0, std,
                                    0, desync=desync, gaussian_noise=gaussian_noise)
        np.savez(new_filename,
                 GE=GE,
                 NT=NT,
                 PI=PI,
                 hp_values=hp_values,
                 history=history.history,
                 dataset=dataset_parameters,
                 desync=desync,
                 desync_level=desync_level,
                 desync_level_augmentation=0,
                 gaussian_noise=gaussian_noise,
                 std=std,
                 std_augmentation=0
                 )

        del model
        gc.collect()
