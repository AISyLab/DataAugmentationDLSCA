from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import *
from src.metrics.mutual_information import *
from scalib.metrics import SNR
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class GetMutualInformation(Callback):
    def __init__(self, dataset, nt, num_outputs):
        super().__init__()
        self.dataset = dataset
        self.nt = nt
        self.labels = to_categorical(dataset.attack_labels, num_classes=dataset.classes)
        self.classes = dataset.classes
        self.num_outputs = num_outputs
        self.mi_xt_profiling = {}
        self.mi_ty_profiling = {}
        self.mi_xt_attack = {}
        self.mi_ty_attack = {}
        self.layer_names = []

    def on_epoch_end(self, epoch, logs=None):
        if self.num_outputs > 1:
            multi_output_predictions = self.model.predict(self.dataset.x_attack[:self.nt])
            # self.mi_dict_epochs_multiple[epoch] = mutual_information(multi_output_predictions, self.labels)
        else:
            layer_name = None
            outputs = [[layer_i.output, layer_i.name] for layer_i in self.model.layers if layer_i.name == layer_name or layer_name is None]
            self.layer_names = []
            for layer_i in self.model.layers:
                if layer_i.name not in ["input_layer", "flatten", "output"]:
                    self.layer_names.append(layer_i.name)

            bin_size = 30

            for output_index, output in enumerate(outputs):
                if output_index > 0:
                    if output[1] not in ["input_layer", "flatten", "output"]:
                        intermediate_model = Model(inputs=self.model.input, outputs=output[0])
                        layer_activations_profiling = intermediate_model.predict(self.dataset.x_profiling[:self.nt])
                        layer_activations_attack = intermediate_model.predict(self.dataset.x_attack[:self.nt])
                        layer_name = output[1]

                        reshape = False
                        if len(np.shape(self.dataset.x_profiling)) > 2:
                            self.dataset.x_profiling = self.dataset.x_profiling.reshape(self.dataset.x_profiling.shape[0],
                                                                                        self.dataset.x_profiling.shape[1])
                            self.dataset.x_attack = self.dataset.x_attack.reshape(self.dataset.x_attack.shape[0],
                                                                                  self.dataset.x_attack.shape[1])
                            reshape = True

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

                        if epoch == 0:
                            self.mi_xt_profiling[layer_name] = {}
                            self.mi_ty_profiling[layer_name] = {}
                            self.mi_xt_attack[layer_name] = {}
                            self.mi_ty_attack[layer_name] = {}

                        # self.mi_ty_profiling[layer_name][epoch] = calc_ity(self.dataset.y_profiling[:self.nt],
                        #                                                    layer_activations_profiling[:, peaks_prof_y],
                        #                                                    bin_size)
                        # self.mi_xt_profiling[layer_name][epoch] = calc_ixt(self.dataset.x_profiling[:self.nt, [62, 74, 188, 199, 200]],
                        #                                                    layer_activations_profiling[:, peaks_prof_m1], bin_size)
                        # self.mi_ty_attack[layer_name][epoch] = calc_ity(self.dataset.y_attack[:self.nt],
                        #                                                 layer_activations_attack[:, peaks_attack_y], bin_size)
                        # self.mi_xt_attack[layer_name][epoch] = calc_ixt(self.dataset.x_attack[:self.nt, [62, 74, 188, 199, 200]],
                        #                                                 layer_activations_attack[:, peaks_attack_m1], bin_size)

                        # self.mi_ty_profiling[layer_name][epoch] = mutual_information(layer_activations_profiling,
                        #                                                              self.dataset.profiling_labels[:self.nt],
                        #                                                              self.dataset.classes, bin_size)
                        # self.mi_xt_profiling[layer_name][epoch] = mutual_information(layer_activations_profiling,
                        #                                                              self.dataset.share1_profiling[:self.nt],
                        #                                                              self.dataset.classes, bin_size)
                        # self.mi_ty_attack[layer_name][epoch] = mutual_information(layer_activations_attack,
                        #                                                           self.dataset.attack_labels[:self.nt],
                        #                                                           self.dataset.classes, bin_size)
                        # self.mi_xt_attack[layer_name][epoch] = mutual_information(layer_activations_attack,
                        #                                                           self.dataset.share1_attack[:self.nt],
                        #                                                           self.dataset.classes, bin_size)

                        # infoplane = ZivInformationPlane(self.dataset.x_profiling[:self.nt],
                        #                                 to_categorical(self.dataset.profiling_labels[:self.nt], 256),
                        #                                 bins=np.linspace(-1, 1, 100))
                        # ixt_ity = infoplane.mutual_information(layer_activations_profiling)
                        # self.mi_xt_profiling[layer_name][epoch] = ixt_ity[0]
                        # self.mi_ty_profiling[layer_name][epoch] = ixt_ity[1]
                        # infoplane = ZivInformationPlane(self.dataset.x_attack,
                        #                                 to_categorical(self.dataset.attack_labels, 256),
                        #                                 bins=np.linspace(-1, 1, 100))
                        # ixt_ity = infoplane.mutual_information(layer_activations_attack)
                        # self.mi_xt_attack[layer_name][epoch] = ixt_ity[0]
                        # self.mi_ty_attack[layer_name][epoch] = ixt_ity[1]

                        infoplane = ZivInformationPlane(self.dataset.x_profiling[:self.nt],
                                                        to_categorical(self.dataset.profiling_labels[:self.nt], 256),
                                                        bins=np.linspace(-1, 1, 100))
                        self.mi_ty_profiling[layer_name][epoch] = infoplane.mutual_information(layer_activations_profiling)[1]

                        infoplane = ZivInformationPlane(self.dataset.x_profiling[:self.nt],
                                                        to_categorical(self.dataset.share2_profiling[:self.nt], 256),
                                                        bins=np.linspace(-1, 1, 100))
                        self.mi_xt_profiling[layer_name][epoch] = infoplane.mutual_information(layer_activations_profiling)[1]

                        infoplane = ZivInformationPlane(self.dataset.x_attack,
                                                        to_categorical(self.dataset.attack_labels, 256),
                                                        bins=np.linspace(-1, 1, 100))
                        self.mi_ty_attack[layer_name][epoch] = infoplane.mutual_information(layer_activations_attack)[1]

                        infoplane = ZivInformationPlane(self.dataset.x_attack,
                                                        to_categorical(self.dataset.share2_attack[:self.nt], 256),
                                                        bins=np.linspace(-1, 1, 100))
                        self.mi_xt_attack[layer_name][epoch] = infoplane.mutual_information(layer_activations_attack)[1]

                        print(f"{layer_name}: ({self.mi_xt_profiling[layer_name][epoch]}, {self.mi_ty_profiling[layer_name][epoch]})")
                        print(f"{layer_name}: ({self.mi_xt_attack[layer_name][epoch]}, {self.mi_ty_attack[layer_name][epoch]})")

                        if reshape:
                            self.dataset.x_profiling = self.dataset.x_profiling.reshape(self.dataset.x_profiling.shape[0],
                                                                                        self.dataset.x_profiling.shape[1], 1)
                            self.dataset.x_attack = self.dataset.x_attack.reshape(self.dataset.x_attack.shape[0],
                                                                                  self.dataset.x_attack.shape[1], 1)
