from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from src.metrics.guessing_entropy import *
from src.metrics.perceived_information import *
import numpy as np
import glob


class CallbackBOMetric(Callback):
    def __init__(self, x_attack, n_attack, attack_labels, project_folder, labels_key_guess_attack, correct_key, number_of_batches,
                 key_rank_attack_traces, folder, dataset_name, model_type, leakage_model, npoi, classes, bottleneck=False):
        super().__init__()
        self.project_folder = project_folder
        self.n_attack = n_attack
        self.x_attack = x_attack
        self.attack_labels = attack_labels
        self.labels_key_guess_attack = labels_key_guess_attack
        self.correct_key = correct_key
        self.number_of_batches = number_of_batches
        self.key_rank_attack_traces = key_rank_attack_traces
        self.folder = folder
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.leakage_model = leakage_model
        self.npoi = npoi
        self.classes = classes
        self.bottleneck = bottleneck
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

    def on_batch_end(self, batch, logs=None):
        if batch == self.number_of_batches - 1:
            predictions = self.model.predict(self.x_attack[:self.n_attack])
            GE, NT = guessing_entropy(predictions, self.labels_key_guess_attack, self.correct_key, self.key_rank_attack_traces)
            np.savez(f"{self.project_folder}/nt_sum.npz", nt_sum=NT * GE[len(GE) - 1])

    def on_train_end(self, logs=None):

        predictions = self.model.predict(self.x_attack[:self.n_attack])
        GE, NT = guessing_entropy(predictions, self.labels_key_guess_attack, self.correct_key, self.key_rank_attack_traces)
        PI = information(predictions, self.attack_labels[:self.n_attack], self.classes)

        if GE[len(GE) - 1] < 2:

            mlp_type = "bo" if self.bottleneck else "flat"

            file_count = 0
            for name in glob.glob(
                    f"{self.folder}/{self.dataset_name}_{self.model_type}_{self.leakage_model}_*_{self.npoi}_poi_search_{mlp_type}.npz"):
                file_count += 1
            new_filename = f"{self.folder}/{self.dataset_name}_{self.model_type}_{self.leakage_model}_{file_count + 1}_{self.npoi}_poi_search_{mlp_type}.npz"

            npz_file = np.load(f"{self.project_folder}/hp_values.npz", allow_pickle=True)
            hp_values = npz_file["hp_values"]

            np.savez(new_filename,
                     PI=PI,
                     GE=GE,
                     NT=NT,
                     hp_values=hp_values,
                     accuracy=self.accuracy,
                     val_accuracy=self.val_accuracy,
                     loss=self.loss,
                     val_loss=self.val_loss
                     )
