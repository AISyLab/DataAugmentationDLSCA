from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from src.callbacks.callback_perceived_information import *
from src.neural_networks.models import *
from scalib.metrics import SNR
import numpy as np
import gc
from scipy.stats import multivariate_normal
import time


def template_training(X, Y, pool=False):
    num_clusters = int(max(Y) + 1)
    classes = np.unique(Y)
    # assign traces to clusters based on lables
    HW_category_for_traces = [[] for _ in range(num_clusters)]
    for i in range(len(X)):
        HW = int(Y[i])
        HW_category_for_traces[HW].append(X[i])
    HW_category_for_traces = [np.array(HW_category_for_traces[HW]) for HW in range(num_clusters)]

    # calculate Covariance Matrices
    # step 1: calculate mean matrix of POIs
    meanMatrix = np.zeros((num_clusters, len(X[0])))
    for i in range(num_clusters):
        meanMatrix[i] = np.mean(HW_category_for_traces[i], axis=0)
    # step 2: calculate covariance matrix
    covMatrix = np.zeros((num_clusters, len(X[0]), len(X[0])))
    for HW in range(num_clusters):
        for i in range(len(X[0])):
            for j in range(len(X[0])):
                x = HW_category_for_traces[HW][:, i]
                y = HW_category_for_traces[HW][:, j]
                covMatrix[HW, i, j] = np.cov(x, y)[0][1]
    if pool:
        covMatrix[:] = np.mean(covMatrix, axis=0)
    return meanMatrix, covMatrix, classes


# Calculate probability of the most possible cluster for each traces
def template_attacking_proba_fast(meanMatrix, covMatrix, X_test, classes, labels):
    labels = np.array(labels, dtype=np.uint8)
    p_k = np.ones(len(classes), dtype=np.float64)
    for k in range(len(classes)):
        p_k[k] = np.count_nonzero(labels == k)
    p_k /= len(labels)

    number_traces = X_test.shape[0]
    rv_array = []
    m = 1e-6
    for idx in range(len(classes)):
        rv_array.append(multivariate_normal(meanMatrix[idx], covMatrix[idx], allow_singular=True))

    l_pk = len(p_k)
    p_k = np.repeat(p_k, number_traces)
    p_k = p_k.reshape(l_pk, number_traces)
    p_k = p_k.T

    proba = [o.pdf(X_test) for o in rv_array]
    proba = np.array(proba).T
    proba = np.divide(np.multiply(proba, p_k).T, np.sum(np.multiply(proba, p_k), axis=1)).T

    return proba


def CalculateSNRFast(l, l_i):
    trace_length = l.shape[1]
    mean = np.zeros([256, trace_length])
    var = np.zeros([256, trace_length])
    cpt = np.zeros(256)

    for i in range(256):
        mean[i] = np.sum(l[l_i[i]], axis=0)
        var[i] = np.sum(np.square(l[l_i[i]]), axis=0)
        cpt[i] = len(l_i[i])

    for i in range(256):
        # average the traces for each SBox output
        mean[i] = mean[i] / cpt[i]
        # variance  = mean[x^2] - (mean[x])^2
        var[i] = var[i] / cpt[i] - np.square(mean[i])
    # Compute mean [var_cond] and var[mean_cond] for the conditional variances and means previously processed
    # calculate the trace variation in each points
    varMean = np.var(mean, 0)
    # evaluate if current point is stable for all S[p^k] cases
    MeanVar = np.mean(var, 0)
    return varMean / MeanVar


def CalculateSNR(l, IntermediateData):
    trace_length = l.shape[1]
    mean = np.zeros([256, trace_length])
    var = np.zeros([256, trace_length])
    cpt = np.zeros(256)
    i = 0

    for trace in l:
        # classify the traces based on its SBox output
        # then add the classified traces together
        mean[IntermediateData[i]] += trace
        var[IntermediateData[i]] += np.square(trace)
        # count the trace number for each SBox output
        cpt[IntermediateData[i]] += 1
        i += 1

    for i in range(256):
        # average the traces for each SBox output
        mean[i] = mean[i] / cpt[i]
        # variance  = mean[x^2] - (mean[x])^2
        var[i] = var[i] / cpt[i] - np.square(mean[i])
    # Compute mean [var_cond] and var[mean_cond] for the conditional variances and means previously processed
    # calculate the trace variation in each points
    varMean = np.var(mean, 0)
    # evaluate if current point is stable for all S[p^k] cases
    MeanVar = np.mean(var, 0)
    return varMean / MeanVar


class GetExplainHiddenLayers(Callback):
    def __init__(self, dataset, nt, methods=None, hp_values=None):
        super().__init__()
        self.dataset = dataset
        self.layer_names = None
        self.epoch_step = 5
        self.methods = methods
        self.nt = nt

        self.l1_m1_prof = []
        self.l1_m2_prof = []
        self.l1_y_prof = []
        for i in range(256):
            self.l1_m1_prof.append(np.where(np.array(self.dataset.share1_profiling[:self.nt]) == i)[0])
            self.l1_m2_prof.append(np.where(np.array(self.dataset.share2_profiling[:self.nt]) == i)[0])
            self.l1_y_prof.append(np.where(np.array(self.dataset.profiling_labels[:self.nt]) == i)[0])

        self.l1_m1_attack = []
        self.l1_m2_attack = []
        self.l1_y_attack = []
        for i in range(256):
            self.l1_m1_attack.append(np.where(np.array(self.dataset.share1_attack) == i)[0])
            self.l1_m2_attack.append(np.where(np.array(self.dataset.share2_attack) == i)[0])
            self.l1_y_attack.append(np.where(np.array(self.dataset.attack_labels) == i)[0])

        self.snr_epochs_r = {}
        self.snr_epochs_masked_sbox = {}
        self.snr_epochs_sboxout = {}

        self.ta_epochs_r = {}
        self.ta_epochs_masked_sbox = {}
        self.ta_epochs_sboxout = {}

        self.history_shares = {}
        self.mlp_epochs_r = {}
        self.mlp_epochs_masked_sbox = {}
        self.mlp_epochs_sboxout = {}

        self.hp_values = hp_values

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.epoch_step == 0:

            layer_name = None
            outputs = [[layer_i.output, layer_i.name] for layer_i in self.model.layers if layer_i.name == layer_name or layer_name is None]
            self.layer_names = []
            for layer_i in self.model.layers:
                if layer_i.name not in ["input_layer", "batch_normalization", "flatten", "output"]:
                    self.layer_names.append(layer_i.name)

            for output_index, output in enumerate(outputs):
                if output_index > 0:

                    if "input_layer" not in output[1] and "batch_normalization" not in output[1] and "flatten" not in output[
                        1] and "output" not in output[1]:
                        intermediate_model = Model(inputs=self.model.input, outputs=output[0])
                        layer_activations_profiling = intermediate_model.predict(self.dataset.x_profiling[:self.nt])
                        layer_activations_attack = intermediate_model.predict(self.dataset.x_attack)
                        layer_name = output[1]

                        if "mlp" in self.methods:
                            # if len(np.shape(layer_activations_profiling)) > 2:
                            #     self.explain_filters_mlp(layer_activations_profiling, layer_activations_attack, epoch, layer_name)
                            # else:
                            self.explain_layers_mlp(layer_activations_profiling, layer_activations_attack, epoch, layer_name)
                            if "conv" in layer_name:
                                self.explain_filters_mlp(layer_activations_profiling, layer_activations_attack, epoch, layer_name)
                        if "snr" in self.methods:
                            self.explain_layers_snr(layer_activations_attack, epoch, layer_name)
                        if "ta" in self.methods:
                            self.explain_layers_ta(layer_activations_profiling, layer_activations_attack, epoch, layer_name)

                        del layer_activations_profiling
                        del layer_activations_attack
                        gc.collect()

    def explain_layers_snr(self, layer_activations_attack, epoch, layer_name):

        print(f"Layer: {layer_name}, epoch: {epoch}")

        if epoch == self.epoch_step - 1:
            self.snr_epochs_r[layer_name] = {}
            self.snr_epochs_masked_sbox[layer_name] = {}
            self.snr_epochs_sboxout[layer_name] = {}

        if len(np.shape(layer_activations_attack)) > 2:
            layer_activations_attack = layer_activations_attack.transpose(0, 2, 1)
            layer_activations_attack = layer_activations_attack.reshape(layer_activations_attack.shape[0],
                                                                        layer_activations_attack.shape[1] *
                                                                        layer_activations_attack.shape[2])

        snr_m1 = CalculateSNRFast(layer_activations_attack, self.l1_m1_attack)
        snr_m2 = CalculateSNRFast(layer_activations_attack, self.l1_m2_attack)
        snr_y = CalculateSNRFast(layer_activations_attack, self.l1_y_attack)

        self.snr_epochs_r[layer_name][epoch] = np.max(snr_m1)
        self.snr_epochs_masked_sbox[layer_name][epoch] = np.max(snr_m2)
        self.snr_epochs_sboxout[layer_name][epoch] = np.max(snr_y)

        print(
            f"SNR: {self.snr_epochs_r[layer_name][epoch]}, {self.snr_epochs_masked_sbox[layer_name][epoch]}, {self.snr_epochs_sboxout[layer_name][epoch]}")

    def explain_layers_ta(self, layer_activations_profiling, layer_activations_attack, epoch, layer_name):

        print(f"Layer: {layer_name}, epoch: {epoch}")

        if epoch == self.epoch_step - 1:
            self.ta_epochs_r[layer_name] = {}
            self.ta_epochs_masked_sbox[layer_name] = {}
            self.ta_epochs_sboxout[layer_name] = {}

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

        snr_m1 = CalculateSNRFast(layer_activations_profiling, self.l1_m1_prof)
        snr_m2 = CalculateSNRFast(layer_activations_profiling, self.l1_m2_prof)
        snr_y = CalculateSNRFast(layer_activations_profiling, self.l1_y_prof)

        poi_m1 = np.sort(np.argsort(snr_m1)[::-1][:5])
        poi_m2 = np.sort(np.argsort(snr_m2)[::-1][:5])
        poi_y = np.sort(np.argsort(snr_y)[::-1][:5])

        mean_v, cov_v, classes = template_training(layer_activations_profiling[:, poi_m1], self.dataset.share1_profiling[:self.nt])
        TA_prediction_m1 = template_attacking_proba_fast(mean_v, cov_v, layer_activations_attack[:, poi_m1], classes,
                                                         self.dataset.share1_attack)

        mean_v, cov_v, classes = template_training(layer_activations_profiling[:, poi_m2], self.dataset.share2_profiling[:self.nt])
        TA_prediction_m2 = template_attacking_proba_fast(mean_v, cov_v, layer_activations_attack[:, poi_m2], classes,
                                                         self.dataset.share2_attack)

        mean_v, cov_v, classes = template_training(layer_activations_profiling[:, poi_y], self.dataset.profiling_labels[:self.nt])
        TA_prediction_y = template_attacking_proba_fast(mean_v, cov_v, layer_activations_attack[:, poi_y], classes,
                                                        self.dataset.attack_labels)

        self.ta_epochs_r[layer_name][epoch] = information(TA_prediction_m1, self.dataset.share1_attack, self.dataset.classes)
        self.ta_epochs_masked_sbox[layer_name][epoch] = information(TA_prediction_m2, self.dataset.share2_attack, self.dataset.classes)
        self.ta_epochs_sboxout[layer_name][epoch] = information(TA_prediction_y, self.dataset.attack_labels, self.dataset.classes)

        print(
            f"TA: {self.ta_epochs_r[layer_name][epoch]}, {self.ta_epochs_masked_sbox[layer_name][epoch]}, {self.ta_epochs_sboxout[layer_name][epoch]}")

    def explain_layers_mlp(self, layer_activations_profiling, layer_activations_attack, epoch, layer_name):

        print(f"Layer: {layer_name}, epoch: {epoch}")

        if epoch == self.epoch_step - 1:
            self.history_shares[layer_name] = {}
            self.mlp_epochs_r[layer_name] = {}
            self.mlp_epochs_masked_sbox[layer_name] = {}
            self.mlp_epochs_sboxout[layer_name] = {}

        p_outputs = []
        a_outputs = []

        p_outputs.append(to_categorical(self.dataset.share1_profiling[:self.nt], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.share1_attack)
        p_outputs.append(to_categorical(self.dataset.share2_profiling[:self.nt], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.share2_attack)
        p_outputs.append(to_categorical(self.dataset.profiling_labels[:self.nt], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.attack_labels)

        flatten = True if len(np.shape(layer_activations_profiling)) > 2 else False
        shape = None
        if flatten:
            shape = (layer_activations_profiling.shape[1], layer_activations_profiling.shape[2])

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
        callback_pi_encoded = GetPerceivedInformation(layer_activations_attack, a_outputs, self.dataset.classes, epoch_masks, 3)

        model_encoded = mlp_encoded_multiple(self.dataset.classes, layer_activations_profiling.shape[1], flatten=False, shape=None,
                                             hp_values=self.hp_values)
        history_shares = model_encoded.fit(
            x=layer_activations_profiling,
            y=p_outputs,
            batch_size=2000,
            verbose=2,
            epochs=epoch_masks,
            shuffle=True,
            callbacks=[callback_pi_encoded])

        self.history_shares[layer_name][epoch] = history_shares.history
        self.mlp_epochs_r[layer_name][epoch] = np.max(callback_pi_encoded.get_pi_multiple()[:, 0])
        self.mlp_epochs_masked_sbox[layer_name][epoch] = np.max(callback_pi_encoded.get_pi_multiple()[:, 1])
        self.mlp_epochs_sboxout[layer_name][epoch] = np.max(callback_pi_encoded.get_pi_multiple()[:, 2])

        print(
            f"MLP: {self.mlp_epochs_r[layer_name][epoch]}, {self.mlp_epochs_masked_sbox[layer_name][epoch]}, {self.mlp_epochs_sboxout[layer_name][epoch]}")

        del model_encoded

    def explain_filters_mlp(self, layer_activations_profiling, layer_activations_attack, epoch, layer_name):

        print(f"Layer: {layer_name}, epoch: {epoch}")

        p_outputs = []
        a_outputs = []

        p_outputs.append(to_categorical(self.dataset.share1_profiling[:self.nt], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.share1_attack)
        p_outputs.append(to_categorical(self.dataset.share2_profiling[:self.nt], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.share2_attack)
        p_outputs.append(to_categorical(self.dataset.profiling_labels[:self.nt], num_classes=self.dataset.classes))
        a_outputs.append(self.dataset.attack_labels)

        layer_activations_profiling = layer_activations_profiling.transpose(0, 2, 1)
        layer_activations_attack = layer_activations_attack.transpose(0, 2, 1)

        nb_filters = layer_activations_profiling.shape[1]

        for fi in range(nb_filters):

            if epoch == self.epoch_step - 1:
                self.history_shares[layer_name][f"filter_{fi}"] = {}
                self.mlp_epochs_r[layer_name][f"filter_{fi}"] = {}
                self.mlp_epochs_masked_sbox[layer_name][f"filter_{fi}"] = {}
                self.mlp_epochs_sboxout[layer_name][f"filter_{fi}"] = {}

            filter_activations_profiling = layer_activations_profiling[:, fi]
            filter_activations_attack = layer_activations_attack[:, fi]

            epoch_masks = 10
            callback_pi_encoded = GetPerceivedInformation(filter_activations_attack, a_outputs, self.dataset.classes, epoch_masks, 3)

            model_encoded = mlp_encoded_multiple(self.dataset.classes, filter_activations_profiling.shape[1], flatten=False, shape=None,
                                                 hp_values=self.hp_values)
            history_shares = model_encoded.fit(
                x=filter_activations_profiling,
                y=p_outputs,
                batch_size=2000,
                verbose=2,
                epochs=epoch_masks,
                shuffle=True,
                callbacks=[callback_pi_encoded])

            self.history_shares[layer_name][f"filter_{fi}"][epoch] = history_shares.history
            self.mlp_epochs_r[layer_name][f"filter_{fi}"][epoch] = np.max(callback_pi_encoded.get_pi_multiple()[:, 0])
            self.mlp_epochs_masked_sbox[layer_name][f"filter_{fi}"][epoch] = np.max(callback_pi_encoded.get_pi_multiple()[:, 1])
            self.mlp_epochs_sboxout[layer_name][f"filter_{fi}"][epoch] = np.max(callback_pi_encoded.get_pi_multiple()[:, 2])

            print(
                f"MLP: {self.mlp_epochs_r[layer_name][f'filter_{fi}'][epoch]}, {self.mlp_epochs_masked_sbox[layer_name][f'filter_{fi}'][epoch]}, {self.mlp_epochs_sboxout[layer_name][f'filter_{fi}'][epoch]}")

            del model_encoded
