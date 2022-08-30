from tensorflow.keras.callbacks import *
from src.metrics.perceived_information import *
import numpy as np


class GetPerceivedInformation(Callback):
    def __init__(self, x_attack, labels, classes, epochs, num_outputs):
        super().__init__()
        self.x_attack = x_attack
        self.labels = labels
        self.classes = classes
        self.num_outputs = num_outputs
        self.pi_dict_epochs = np.zeros(epochs)
        self.pi_dict_epochs_multiple = np.zeros((epochs, num_outputs))

    def on_epoch_end(self, epoch, logs=None):
        if self.num_outputs > 1:
            multi_output_predictions = self.model.predict(self.x_attack)
            for m_i, m in enumerate(multi_output_predictions):
                self.pi_dict_epochs_multiple[epoch][m_i] = information(m, self.labels[m_i], self.classes)
        else:
            predictions = self.model.predict(self.x_attack)
            self.pi_dict_epochs[epoch] = information(predictions, self.labels, self.classes)

    def get_pi_multiple(self):
        return self.pi_dict_epochs_multiple

    def get_pi(self):
        return self.pi_dict_epochs
