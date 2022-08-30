import numpy as np
from scipy.stats import entropy
from tensorflow.keras.utils import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import *
import matplotlib.pyplot as plt
from scripts_mask_explainability_paper.src.datasets.load_ascadr import *


def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


def calc_ixt(inputdata, layerdata, num_of_bins):
    p_xs, unique_inverse_x = get_unique_probs(inputdata)

    bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
    digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins) - 1].reshape(len(layerdata), -1)
    p_ts, _ = get_unique_probs(digitized)

    H_LAYER = -np.sum(p_ts * np.log(p_ts))
    H_LAYER_GIVEN_INPUT = 0.
    for xval in unique_inverse_x:
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
        H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))

    itx = H_LAYER - H_LAYER_GIVEN_INPUT

    return itx


def calc_ity(labels, layerdata, num_of_bins):
    p_xs, unique_inverse_x = get_unique_probs(labels)

    bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
    digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins) - 1].reshape(len(layerdata), -1)
    p_ts, _ = get_unique_probs(digitized)

    entropy_activations = -np.sum(p_ts * np.log(p_ts))
    entropy_activations_given_input = 0.
    for x in np.arange(len(p_xs)):
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == x, :])
        entropy_activations_given_input += - p_xs[x] * np.sum(p_t_given_x * np.log(p_t_given_x))
    ity = entropy_activations - entropy_activations_given_input

    return ity


def mutual_information(activations, labels, num_classes, num_of_bins):
    """
    implements I(K;L)

    p_k = the distribution of the sensitive variable K
    data = the samples we 'measured'. It its the n^k_p samples from p(l|k)
    model = the estimated model \hat{p}(l|k).

    returns an estimated of mutual information
    """

    labels = np.array(labels, dtype=np.uint8)
    p_k = np.ones(num_classes, dtype=np.float64)
    for k in range(num_classes):
        p_k[k] = np.count_nonzero(labels == k)
    p_k /= len(labels)

    acc = entropy(p_k, base=2)  # we initialize the value with H(K)

    activations = np.round(activations, 2)
    bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
    binned_activations = bins[np.digitize(np.squeeze(activations.reshape(1, -1)), bins) - 1].reshape(len(activations), -1)
    unique_probabilities_activations, _ = get_unique_probs(binned_activations)

    for k in range(num_classes):
        trace_index_with_label_k = np.where(labels == k)[0]
        y_pred_k = unique_probabilities_activations[trace_index_with_label_k]

        y_pred_k = np.array(y_pred_k)
        if len(y_pred_k) > 0:
            p_k_l = np.sum(np.log2(y_pred_k)) / len(y_pred_k)
            acc += p_k[k] * p_k_l

    # print(f"PI: {acc}")

    return acc


def KL(a, b):
    """Calculate the Kullback Leibler divergence between a and b """
    D_KL = np.nansum(np.multiply(a, np.log(np.divide(a, b + np.spacing(1)))), axis=1)
    return D_KL


def calc_entropy_for_specipic_t(current_ts, px_i):
    """Calc entropy for specipic t"""
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
    return H2X


def calc_condtion_entropy(px, t_data, unique_inverse_x):
    # Condition entropy of t given x
    H2X_array = np.array([calc_entropy_for_specipic_t(t_data[unique_inverse_x == i, :], px[i]) for i in range(px.shape[0])])
    H2X = np.sum(H2X_array)
    return H2X


def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y, unique_array):
    """Calculate the MI based on binning of the data"""
    H2 = -np.sum(ps2 * np.log2(ps2))
    H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y)
    IY = H2 - H2Y
    IX = H2 - H2X
    return IX, IY


def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    pys = np.sum(label, axis=0) / float(label.shape[0])
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    pxs = unique_counts / float(np.sum(unique_counts))
    p_y_given_x = []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs, :], axis=0)
        p_y_given_x.append(py_x_current)
    p_y_given_x = np.array(p_y_given_x).T
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs


def calc_probs(t_index, unique_inverse, label, b, b1, len_unique_a):
    """Calculate the p(x|T) and p(y|T)"""
    indexs = unique_inverse == t_index
    p_y_ts = np.sum(label[indexs], axis=0) / label[indexs].shape[0]
    unique_array_internal, unique_counts_internal = \
        np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
    indexes_x = np.where(np.in1d(b1, b[indexs]))
    p_x_ts = np.zeros(len_unique_a)
    p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))
    return p_x_ts, p_y_ts


def calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y, calc_DKL=False):
    bins = bins.astype(np.float32)
    num_of_bins = bins.shape[0]
    # bins = stats.mstats.mquantiles(np.squeeze(data.reshape(1, -1)), np.linspace(0,1, num=num_of_bins))
    # hist, bin_edges = np.histogram(np.squeeze(data.reshape(1, -1)), normed=True)
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
    if calc_DKL:
        pxy_given_T = np.array(
            [calc_probs(i, unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(unique_array))]
        )
        p_XgT = np.vstack(pxy_given_T[:, 0])
        p_YgT = pxy_given_T[:, 1]
        p_YgT = np.vstack(p_YgT).T
        DKL_YgX_YgT = np.sum([KL(c_p_YgX, p_YgT.T) for c_p_YgX in p_YgX.T], axis=0)
        H_Xgt = np.nansum(p_XgT * np.log2(p_XgT), axis=1)
    local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
                                                     unique_array)
    return local_IXT, local_ITY


def calc_information_for_layer_with_other(data, bins, unique_inverse_x, unique_inverse_y, label,
                                          b, b1, len_unique_a, pxs, p_YgX, pys1,
                                          percent_of_sampling=50):
    local_IXT, local_ITY = calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
                                                     len_unique_a, p_YgX, unique_inverse_x,
                                                     unique_inverse_y)
    params = {}
    params['local_IXT'] = local_IXT
    params['local_ITY'] = local_ITY
    return params


class ZivInformationPlane():
    # The code by Ravid Schwartz-Ziv is very hard to understand
    # (the reader is encouraged to try it themselves)
    # Solution: wrap it into this class and don't touch it with a 10-meter pole

    def __init__(self, X, Y, bins=np.linspace(-1, 1, 30)):
        """
        Inititalize information plane (set X and Y and get ready to calculate I(T;X), I(T;Y))
        X and Y have to be discrete
        """

        plane_params = dict(zip(['pys', 'pys1', 'p_YgX', 'b1', 'b',
                                 'unique_a', 'unique_inverse_x', 'unique_inverse_y', 'pxs'],
                                extract_probs(np.array(Y).astype(np.float), X)))

        plane_params['bins'] = bins
        plane_params['label'] = Y
        plane_params['len_unique_a'] = len(plane_params['unique_a'])
        del plane_params['unique_a']
        del plane_params['pys']

        self.X = X
        self.Y = Y
        self.plane_params = plane_params

    def mutual_information(self, layer_output):
        """
        Given the outputs T of one layer of an NN, calculate MI(X;T) and MI(T;Y)

        params:
            layer_output - a 3d numpy array, where 1st dimension is training objects, second - neurons

        returns:
            IXT, ITY - mutual information
        """

        information = calc_information_for_layer_with_other(layer_output, **self.plane_params)
        return information['local_IXT'], information['local_ITY']


class MI(Callback):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        layer_name = None
        outputs = [[layer_i.output, layer_i.name] for layer_i in self.model.layers if layer_i.name == layer_name or layer_name is None]
        self.layer_names = []
        for layer_i in self.model.layers:
            if layer_i.name not in ["input_layer", "flatten", "output"]:
                self.layer_names.append(layer_i.name)

        colors = ["blue", "orange", "green", "red", "grey", "magenta"]
        layer_count = 0
        for output_index, output in enumerate(outputs):
            if output_index > 0:
                if output[1] not in ["input_layer", "flatten", "output"]:
                    intermediate_model = Model(inputs=self.model.input, outputs=output[0])
                    layer_activations = intermediate_model.predict(self.x)
                    infoplane = ZivInformationPlane(self.x, to_categorical(self.y, 256), bins=np.linspace(-1, 1, 50))
                    ixt, ity = infoplane.mutual_information(layer_activations)
                    print(ixt, ity)
                    plt.scatter(ixt, ity, color=colors[layer_count])
                    layer_count += 1

    def on_train_end(self, logs=None):
        plt.show()


if __name__ == "__main__":
    X = np.random.randint(0, 256, (5000, 100))
    y = np.random.randint(0, 256, 5000)
    X = np.array(X).astype(np.float)

    dataset_parameters = {
        "n_profiling": 200000,
        "n_attack": 5000,
        "n_attack_ge": 3000,
        "target_byte": 2,
        "r_byte": 2,
        "rin_byte": 16,
        "rout_byte": 17
    }
    class_name = ReadASCADr

    dataset = class_name(
        dataset_parameters["n_profiling"],
        dataset_parameters["n_attack"],
        file_path="D:/traces/ascad-variable.h5",
        target_byte=dataset_parameters["target_byte"],
        leakage_model="ID",
        first_sample=0,
        number_of_samples=1400
    )

    dataset.rescale(reshape_to_cnn=False)

    def mlp(classes, number_of_samples):
        input_shape = (number_of_samples)
        input_layer = Input(shape=input_shape, name="input_layer")

        x = Dense(200, kernel_initializer="random_uniform", activation="elu", name='fc_1')(input_layer)
        x = Dense(200, kernel_initializer="random_uniform", activation="elu", name='fc_2')(x)
        x = Dense(200, kernel_initializer="random_uniform", activation="elu", name='fc_3')(x)
        x = Dense(200, kernel_initializer="random_uniform", activation="elu", name='fc_4')(x)
        x = Dense(200, kernel_initializer="random_uniform", activation="elu", name='fc_5')(x)
        x = Dense(200, kernel_initializer="random_uniform", activation="elu", name='fc_6')(x)

        output_layer = Dense(classes, activation='softmax', name=f'output')(x)

        m_model = Model(input_layer, output_layer, name='mlp_softmax')
        optimizer = Adam(lr=0.001)
        m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        m_model.summary()
        return m_model


    mi = MI(dataset.x_profiling[:10000], dataset.profiling_labels[:10000])

    model = mlp(256, 1400)
    model.fit(x=dataset.x_profiling,
              y=to_categorical(dataset.profiling_labels, num_classes=256),
              batch_size=400,
              verbose=2,
              epochs=200,
              shuffle=True,
              validation_data=(dataset.x_attack, dataset.y_attack),
              callbacks=[mi])
