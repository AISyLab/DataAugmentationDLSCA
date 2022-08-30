from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from src.callbacks.callback_perceived_information import *
from src.neural_networks.models import *
import numpy as np
import gc


class GetWeights(Callback):
    def __init__(self):
        super().__init__()
        self.weight_epochs = []
        self.weight_mean = {}
        self.weight_std = {}
        self.bias_mean = {}
        self.bias_std = {}

    def on_epoch_end(self, epoch, logs=None):

        if epoch == 0:
            for index, layer in enumerate(self.model.layers):
                if len(layer.get_weights()) > 0:
                    if layer.name != "output":
                        self.weight_mean[layer.name] = []
                        self.weight_std[layer.name] = []
                        self.bias_mean[layer.name] = []
                        self.bias_std[layer.name] = []

        # loop over each layer and get weights and biases
        for index, layer in enumerate(self.model.layers):
            if len(layer.get_weights()) > 0:
                if layer.name != "output":
                    w = layer.get_weights()[0]
                    b = layer.get_weights()[1]

                    self.weight_mean[layer.name].append(np.mean(w))
                    self.bias_mean[layer.name].append(np.mean(b))
                    self.weight_std[layer.name].append(np.std(w))
                    self.bias_std[layer.name].append(np.std(b))
