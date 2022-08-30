from tensorflow.keras.callbacks import *
from src.metrics.guessing_entropy import *
import numpy as np


class GetGuessingEntropy(Callback):
    def __init__(self, x_attack, labels_key_guess, correct_key, num_outputs, key_rank_attack_traces):
        super().__init__()
        self.x_attack = x_attack
        self.labels_key_guess = labels_key_guess
        self.correct_key = correct_key
        self.num_outputs = num_outputs
        self.ge_epochs = []
        self.nt_epochs = []
        self.key_rank_attack_traces = key_rank_attack_traces

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.x_attack)
        ge, nt = guessing_entropy(predictions, self.labels_key_guess, self.correct_key, self.key_rank_attack_traces)
        self.ge_epochs.append(ge)
        self.nt_epochs.append(nt)

    def get_ge(self):
        return self.ge_epochs

    def get_nt(self):
        return self.nt_epochs
