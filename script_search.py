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

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.path.append("/home/nfs/gperin/paper_1_data_augmentation_paper")

if __name__ == "__main__":

    dataset_name = sys.argv[1]
    model_type = sys.argv[2]
    leakage_model = sys.argv[3]
    desync = True if int(sys.argv[4]) == 1 else False
    gaussian_noise = True if int(sys.argv[5]) == 1 else False
    time_warping = True if int(sys.argv[6]) == 1 else False
    trace_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme"
    folder_results = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_1_data_augmentation_results/random_search"

    data_augmentation = True

    dataset_parameters = None
    class_name = None

    if dataset_name == "dpa_v42":
        dataset_parameters = {
            "n_profiling": 70000,
            "n_profiling_augmented": 70000,
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
            "n_profiling_augmented": 100000,
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
        dataset = make_desync(dataset)
    if gaussian_noise:
        dataset = make_gaussian_noise(dataset)
    if time_warping:
        dataset = make_time_warping(dataset)

    """ Rescale and reshape (if CNN) """
    dataset.rescale(True if model_type == "cnn" else False)

    """ Generate key guessing table """
    labels_key_guess = dataset.labels_key_hypothesis_attack

    """ Run random search """
    for search_index in range(100):
        """ generate hyperparameters """
        hp_values = hp_list(model_type)
        hp_values["seed"] = np.random.randint(1048576)
        print(hp_values)

        steps_per_epoch = int((dataset_parameters["n_profiling"] + dataset_parameters["n_profiling_augmented"]) / hp_values["batch_size"])
        n_batches_prof = int(dataset_parameters["n_profiling"] / hp_values["batch_size"])
        n_batches_augmented = int(dataset_parameters["n_profiling_augmented"] / hp_values["batch_size"])

        """ Create model """
        baseline_model = cnn(dataset.classes, dataset.ns, hp_values) if model_type == "cnn" else mlp(dataset.classes, dataset.ns,
                                                                                                     hp_values)
        """ Train model """
        model, history = train_model(baseline_model, model_type, dataset, dataset_parameters["epochs"], hp_values["batch_size"],
                                     steps_per_epoch, n_batches_prof, n_batches_augmented, data_augmentation=data_augmentation,
                                     desync=desync, gaussian_noise=gaussian_noise, time_warping=time_warping)

        """ Compute guessing entropy and perceived information """
        predictions = model.predict(dataset.x_attack)
        GE, NT = guessing_entropy(predictions, labels_key_guess, dataset.correct_key, dataset_parameters["n_attack_ge"])
        PI = information(predictions, dataset.attack_labels, dataset.classes)

        """ Save results """
        new_filename = get_filename(folder_results, dataset_name, model_type, leakage_model, desync=desync, gaussian_noise=gaussian_noise,
                                    time_warping=time_warping)
        np.savez(new_filename,
                 GE=GE,
                 NT=NT,
                 PI=PI,
                 hp_values=hp_values,
                 history=history.history,
                 dataset=dataset_parameters
                 )

        del model
        gc.collect()
