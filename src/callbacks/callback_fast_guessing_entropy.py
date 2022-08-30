from tensorflow.keras.callbacks import *
from src.metrics.guessing_entropy import *
import numpy as np


class GetFastGuessingEntropy(Callback):
    def __init__(self, x_attack, labels_key_guess, correct_key, epochs, num_outputs, key_rank_attack_traces):
        super().__init__()
        self.x_attack = x_attack
        self.labels_key_guess = labels_key_guess
        self.correct_key = correct_key
        self.num_outputs = num_outputs
        self.ge_dict_epochs = np.zeros(epochs)
        self.ge_dict_epochs_multiple = np.zeros((epochs, num_outputs))
        self.key_rank_attack_traces = key_rank_attack_traces

    def on_epoch_end(self, epoch, logs=None):
        if self.num_outputs > 1:
            multi_output_predictions = self.model.predict(self.x_attack)
            self.ge_dict_epochs_multiple[epoch] = fast_guessing_entropy_multiple(multi_output_predictions, self.labels_key_guess,
                                                                                 self.correct_key, self.key_rank_attack_traces)
        else:
            predictions = self.model.predict(self.x_attack)
            self.ge_dict_epochs[epoch] = fast_guessing_entropy(predictions, self.labels_key_guess, self.correct_key,
                                                               self.key_rank_attack_traces)

    def get_ge_multiple(self):
        return self.ge_dict_epochs_multiple

    def get_ge(self):
        return self.ge_dict_epochs
