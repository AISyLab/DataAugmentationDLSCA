import numpy as np
from scipy.stats import entropy


def information(model, labels, num_classes):
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

    y_pred = np.array(model + 1e-36)

    for k in range(num_classes):
        trace_index_with_label_k = np.where(labels == k)[0]
        y_pred_k = y_pred[trace_index_with_label_k, k]

        y_pred_k = np.array(y_pred_k)
        if len(y_pred_k) > 0:
            p_k_l = np.sum(np.log2(y_pred_k)) / len(y_pred_k)
            acc += p_k[k] * p_k_l

    print(f"PI: {acc}")

    return acc


def information_multi_output(multi_model, labels, num_classes):
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
    acc_multiple = np.zeros(len(multi_model))

    for y_pred_index, model in enumerate(multi_model):

        acc_multiple[y_pred_index] = acc

        y_pred = np.array(model + 1e-36)

        for k in range(num_classes):
            trace_index_with_label_k = np.where(labels == k)[0]
            y_pred_k = y_pred[trace_index_with_label_k, k]

            y_pred_k = np.array(y_pred_k)
            if len(y_pred_k) > 0:
                p_k_l = np.sum(np.log2(y_pred_k)) / len(y_pred_k)
                acc_multiple[y_pred_index] += p_k[k] * p_k_l

    # print(f"PI: {acc_multiple}")

    return acc_multiple
