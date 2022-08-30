from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from src.callbacks.callback_perceived_information import *
from src.neural_networks.models import *
import numpy as np
import gc


class GetActivations(Callback):
    def __init__(self, dataset, n_prof):
        super().__init__()
        self.n_prof = n_prof
        self.dataset = dataset
        self.layer_names = None
        self.pi_epochs_r = {}
        self.pi_epochs_masked_sbox = {}
        self.pi_epochs_sboxout = {}
        self.history_shares = {}
        self.epoch_step = 5

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.epoch_step == 0:

            layer_name = None
            outputs = [[layer_i.output, layer_i.name] for layer_i in self.model.layers if layer_i.name == layer_name or layer_name is None]
            self.layer_names = []
            for layer_i in self.model.layers:
                if layer_i.name not in ["input_layer", "flatten", "output"]:
                    self.layer_names.append(layer_i.name)

            for output_index, output in enumerate(outputs):
                if output_index > 0:

                    if "input_layer" not in output[1] and "flatten" not in output[1] and "output" not in output[1]:
                        intermediate_model = Model(inputs=self.model.input, outputs=output[0])
                        layer_activations_profiling = intermediate_model.predict(self.dataset.x_profiling[:self.n_prof])
                        layer_activations_attack = intermediate_model.predict(self.dataset.x_attack)
                        layer_name = output[1]
                        self.compute_pi_layers(layer_activations_profiling, layer_activations_attack, epoch, layer_name)

    def compute_pi_layers(self, layer_activations_profiling, layer_activations_attack, epoch, layer_name):

        print(f"Layer: {layer_name}, epoch: {epoch}")

        if epoch == self.epoch_step - 1:
            self.history_shares[layer_name] = {}
            self.pi_epochs_r[layer_name] = {}
            self.pi_epochs_masked_sbox[layer_name] = {}
            self.pi_epochs_sboxout[layer_name] = {}

        p_outputs = []
        a_outputs = []

        p_outputs.append(to_categorical(self.dataset.share1_profiling[:self.n_prof], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.share1_attack)
        p_outputs.append(to_categorical(self.dataset.share2_profiling[:self.n_prof], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.share2_attack)
        p_outputs.append(to_categorical(self.dataset.profiling_labels[:self.n_prof], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.attack_labels)

        epoch_masks = 10
        callback_pi_encoded = GetPerceivedInformation(layer_activations_attack, a_outputs, self.dataset.classes, epoch_masks, 3)

        flatten = True if len(np.shape(layer_activations_profiling)) > 2 else False
        shape = None
        if flatten:
            shape = (layer_activations_profiling.shape[1], layer_activations_profiling.shape[2])
        model_encoded = mlp_encoded_multiple(self.dataset.classes, layer_activations_profiling.shape[1], flatten=flatten, shape=shape)
        history_shares = model_encoded.fit(
            x=layer_activations_profiling,
            y=p_outputs,
            batch_size=2000,
            verbose=2,
            epochs=epoch_masks,
            shuffle=True,
            callbacks=[callback_pi_encoded])

        self.history_shares[layer_name][epoch] = history_shares.history
        self.pi_epochs_r[layer_name][epoch] = callback_pi_encoded.get_pi_multiple()[:, 0]
        self.pi_epochs_masked_sbox[layer_name][epoch] = callback_pi_encoded.get_pi_multiple()[:, 1]
        self.pi_epochs_sboxout[layer_name][epoch] = callback_pi_encoded.get_pi_multiple()[:, 2]

        del model_encoded
        del layer_activations_profiling
        del layer_activations_attack
        gc.collect()
