from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from src.callbacks.callback_perceived_information import *
from src.neural_networks.models import *
import numpy as np
import gc


class GetExplainCompression(Callback):
    def __init__(self, dataset, nt, methods=None, hp_values=None):
        super().__init__()
        self.dataset = dataset
        self.layer_names = None
        self.epoch_step = 1
        self.methods = methods
        self.nt = nt

        self.mlp_epochs_mask = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        self.mlp_epochs_masked_sbox = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

        self.hp_values = hp_values

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.epoch_step == 0:

            layer_name = None
            outputs = [[layer_i.output, layer_i.name] for layer_i in self.model.layers if layer_i.name == layer_name or layer_name is None]
            self.layer_names = []
            for layer_i in self.model.layers:
                if layer_i.name not in ["input_layer", "pool", "batch_normalization", "flatten", "output"]:
                    self.layer_names.append(layer_i.name)

            for output_index, output in enumerate(outputs):
                if output_index > 0:

                    if "input_layer" not in output[1] and "pool" not in output[1] and "batch_normalization" not in output[
                        1] and "flatten" not in output[1] and "output" not in output[1]:
                        intermediate_model = Model(inputs=self.model.input, outputs=output[0])
                        layer_activations_profiling = intermediate_model.predict(self.dataset.x_profiling[:self.nt])
                        layer_activations_attack = intermediate_model.predict(self.dataset.x_attack)
                        layer_name = output[1]

                        self.explain_layers_mlp(layer_activations_profiling, layer_activations_attack, epoch, layer_name)

                        del layer_activations_profiling
                        del layer_activations_attack
                        gc.collect()

    def explain_layers_mlp(self, layer_activations_profiling, layer_activations_attack, epoch, layer_name):

        print(f"Layer: {layer_name}, epoch: {epoch}")

        if epoch == self.epoch_step - 1:
            for byte in range(16):
                self.mlp_epochs_mask[byte][layer_name] = {}
                self.mlp_epochs_masked_sbox[byte][layer_name] = {}

        p_outputs = []
        a_outputs = []

        for byte in range(16):
            p_outputs.append(to_categorical(self.dataset.share1_profiling[byte, :self.nt], num_classes=self.dataset.classes))
            a_outputs.append(self.dataset.share1_attack[byte])
            p_outputs.append(to_categorical(self.dataset.share2_profiling[byte, :self.nt], num_classes=self.dataset.classes))
            a_outputs.append(self.dataset.share2_attack[byte])

        if len(np.shape(layer_activations_profiling)) > 2:
            layer_activations_profiling = layer_activations_profiling.transpose(0, 2, 1)
            layer_activations_profiling = layer_activations_profiling.reshape(layer_activations_profiling.shape[0],
                                                                              layer_activations_profiling.shape[1] *
                                                                              layer_activations_profiling.shape[2])
        if len(np.shape(layer_activations_attack)) > 2:
            layer_activations_attack = layer_activations_attack.transpose(0, 2, 1)
            layer_activations_attack = layer_activations_attack.reshape(layer_activations_attack.shape[0],
                                                                        layer_activations_attack.shape[1] *
                                                                        layer_activations_attack.shape[2])

        epoch_masks = 10
        callback_pi_encoded = GetPerceivedInformation(layer_activations_attack, a_outputs, self.dataset.classes, epoch_masks, 32)

        model_encoded = mlp_encoded_multiple(self.dataset.classes, layer_activations_profiling.shape[1], flatten=False, shape=None,
                                             hp_values=self.hp_values, num_outputs=32)
        model_encoded.fit(
            x=layer_activations_profiling,
            y=p_outputs,
            batch_size=2000,
            verbose=2,
            epochs=epoch_masks,
            shuffle=True,
            callbacks=[callback_pi_encoded])

        for byte in range(16):
            self.mlp_epochs_mask[byte][layer_name][epoch] = np.max(callback_pi_encoded.get_pi_multiple()[:, byte * 2])
            self.mlp_epochs_masked_sbox[byte][layer_name][epoch] = np.max(callback_pi_encoded.get_pi_multiple()[:, byte * 2 + 1])

            print(
                f"MLP: {self.mlp_epochs_mask[byte][layer_name][epoch]}, {self.mlp_epochs_masked_sbox[byte][layer_name][epoch]}")

        del model_encoded
