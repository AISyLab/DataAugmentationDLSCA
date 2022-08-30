from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from src.metrics.perceived_information import *
from src.neural_networks.models import *
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import gc


class GetActivationsTa(Callback):
    def __init__(self, dataset, n_prof):
        super().__init__()
        self.n_prof = n_prof
        self.dataset = dataset
        self.layer_names = None
        self.pi_epochs_r = {}
        self.pi_epochs_masked_sbox = {}
        self.pi_epochs_sboxout = {}
        self.history_shares = {}

    def on_epoch_end(self, epoch, logs=None):
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

    def template_training(self, X, Y, pool=False):
        num_clusters = max(Y) + 1
        classes = np.unique(Y)
        # assign traces to clusters based on lables
        HW_catagory_for_traces = [[] for _ in range(num_clusters)]
        for i in range(len(X)):
            HW = Y[i]
            HW_catagory_for_traces[HW].append(X[i])
        HW_catagory_for_traces = [np.array(HW_catagory_for_traces[HW]) for HW in range(num_clusters)]

        # calculate Covariance Matrices
        # step 1: calculate mean matrix of POIs
        meanMatrix = np.zeros((num_clusters, len(X[0])))
        for i in range(num_clusters):
            meanMatrix[i] = np.mean(HW_catagory_for_traces[i], axis=0)
        # step 2: calculate covariance matrix
        covMatrix = np.zeros((num_clusters, len(X[0]), len(X[0])))
        for HW in range(num_clusters):
            for i in range(len(X[0])):
                for j in range(len(X[0])):
                    x = HW_catagory_for_traces[HW][:, i]
                    y = HW_catagory_for_traces[HW][:, j]
                    covMatrix[HW, i, j] = np.cov(x, y)[0][1]
        if pool:
            covMatrix[:] = np.mean(covMatrix, axis=0)
        return meanMatrix, covMatrix, classes

    # Calculate probability of the most possible cluster for each traces
    def template_attacking_proba(self, meanMatrix, covMatrix, X_test, classes, labels):
        labels = np.array(labels, dtype=np.uint8)
        p_k = np.ones(classes, dtype=np.float64)
        for k in range(classes):
            p_k[k] = np.count_nonzero(labels == k)
        p_k /= len(labels)

        number_traces = X_test.shape[0]
        proba = np.zeros((number_traces, classes.shape[0]))
        rv_array = []
        m = 1e-6
        for idx in range(len(classes)):
            rv_array.append(multivariate_normal(meanMatrix[idx], covMatrix[idx], allow_singular=True))

        for i in range(number_traces):
            if (i % 2000 == 0):
                print(str(i) + '/' + str(number_traces))
            proba[i] = [o.pdf(X_test[i]) for o in rv_array]
            proba[i] = np.multiply(proba[i], p_k) / np.sum(np.multiply(proba[i], p_k))

        return proba

    def compute_pi_layers(self, layer_activations_profiling, layer_activations_attack, epoch, layer_name):

        print(f"Layer: {layer_name}, epoch: {epoch}")

        layer_activations_profiling = layer_activations_profiling.reshape(layer_activations_profiling.shape[0],
                                                                          layer_activations_profiling.shape[1] *
                                                                          layer_activations_profiling.shape[2])
        layer_activations_attack = layer_activations_attack.reshape(layer_activations_attack.shape[0],
                                                                    layer_activations_attack.shape[1] *
                                                                    layer_activations_attack.shape[2])

        if epoch == 0:
            self.history_shares[layer_name] = {}
        self.pi_epochs_r[layer_name] = {}
        self.pi_epochs_masked_sbox[layer_name] = {}
        self.pi_epochs_sboxout[layer_name] = {}

        print("LDA for Y_m1")
        lda = LDA(n_components=5)
        layer_activations_profiling_m1 = lda.fit_transform(layer_activations_profiling, self.dataset.share1_profiling[:self.n_prof])
        layer_activations_attack_m1 = lda.transform(layer_activations_attack)

        print("LDA for Y_m2")
        lda = LDA(n_components=5)
        layer_activations_profiling_m2 = lda.fit_transform(layer_activations_profiling, self.dataset.share2_profiling[:self.n_prof])
        layer_activations_attack_m2 = lda.transform(layer_activations_attack)

        print("LDA for Y")
        lda = LDA(n_components=5)
        layer_activations_profiling_y = lda.fit_transform(layer_activations_profiling, self.dataset.profiling_labels[:self.n_prof])
        layer_activations_attack_y = lda.transform(layer_activations_attack)

        print("TA for Y_m1")
        mean_v, cov_v, classes = self.template_training(layer_activations_profiling_m1, self.dataset.share1_profiling[:self.n_prof])
        TA_prediction_m1 = self.template_attacking_proba(mean_v, cov_v, layer_activations_attack_m1, classes, self.dataset.share1_attack)

        print("TA for Y_m2")
        mean_v, cov_v, classes = self.template_training(layer_activations_profiling_m2, self.dataset.share2_profiling[:self.n_prof])
        TA_prediction_m2 = self.template_attacking_proba(mean_v, cov_v, layer_activations_attack_m2, classes, self.dataset.share2_attack)

        print("TA for Y")
        mean_v, cov_v, classes = self.template_training(layer_activations_profiling_y, self.dataset.profiling_labels[:self.n_prof])
        TA_prediction_y = self.template_attacking_proba(mean_v, cov_v, layer_activations_attack_y, classes, self.dataset.attack_labels)

        self.pi_epochs_r[layer_name][epoch] = information(TA_prediction_m1, self.dataset.share1_attack, classes)
        self.pi_epochs_masked_sbox[layer_name][epoch] = information(TA_prediction_m2, self.dataset.share2_attack, classes)
        self.pi_epochs_sboxout[layer_name][epoch] = information(TA_prediction_y, self.dataset.attack_labels, classes)

        del layer_activations_profiling
        del layer_activations_attack
        gc.collect()
