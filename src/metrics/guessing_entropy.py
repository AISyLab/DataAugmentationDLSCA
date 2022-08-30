import numpy as np


def guessing_entropy(predictions, labels_guess, good_key, key_rank_attack_traces):
    nt = len(predictions)

    key_rank_executions = 20
    key_rank_report_interval = 1

    key_ranking_sum = np.zeros(key_rank_attack_traces)

    predictions = np.log(predictions + 1e-36)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = predictions[index][
            np.asarray([int(leakage[index]) for leakage in labels_guess[:]])
        ]

    for run in range(key_rank_executions):
        r = np.random.choice(range(nt), key_rank_attack_traces, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(key_rank_attack_traces):

            key_probabilities += probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % key_rank_report_interval == 0:
                key_ranking_good_key = list(key_probabilities_sorted).index(good_key) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                kr_count += 1

    guessing_entropy = key_ranking_sum / key_rank_executions

    result_number_of_traces_val = key_rank_attack_traces
    if guessing_entropy[key_rank_attack_traces - 1] < 2:
        for index in range(key_rank_attack_traces - 1, -1, -1):
            if guessing_entropy[index] > 2:
                result_number_of_traces_val = (index + 1) * key_rank_report_interval
                break

    print("GE = {}".format(guessing_entropy[key_rank_attack_traces - 1]))
    print("Number of traces to reach GE = 1: {}".format(result_number_of_traces_val))

    return guessing_entropy, result_number_of_traces_val


def fast_guessing_entropy(predictions, labels_guess, good_key, key_rank_attack_traces):
    nt = len(predictions)

    key_rank_executions = 100

    predictions = np.log(predictions + 1e-36)

    key_ranking_sum = 0

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = predictions[index][
            np.asarray([int(leakage[index]) for leakage in labels_guess[:]])
        ]

    for run in range(key_rank_executions):
        r = np.random.choice(range(nt), key_rank_attack_traces, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.sum(probabilities_kg_all_traces_shuffled[:key_rank_attack_traces], axis=0)
        key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
        key_ranking_sum += list(key_probabilities_sorted).index(good_key) + 1

    guessing_entropy = key_ranking_sum / key_rank_executions

    print("GE = {}".format(guessing_entropy))

    return guessing_entropy


def fast_guessing_entropy_multiple(predictions, labels_guess, good_key, key_rank_attack_traces):
    key_rank_executions = 100

    guessing_entropy = np.zeros(len(predictions))

    for prediction_index, prediction in enumerate(predictions):
        nt = len(prediction)
        prediction = np.log(prediction + 1e-36)

        key_ranking_sum = 0

        probabilities_kg_all_traces = np.zeros((nt, 256))
        for index in range(nt):
            probabilities_kg_all_traces[index] = prediction[index][
                np.asarray([int(leakage[index]) for leakage in labels_guess[:]])
            ]

        for run in range(key_rank_executions):
            r = np.random.choice(range(nt), key_rank_attack_traces, replace=False)
            probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
            key_probabilities = np.sum(probabilities_kg_all_traces_shuffled[:key_rank_attack_traces], axis=0)
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
            key_ranking_sum += list(key_probabilities_sorted).index(good_key) + 1

        guessing_entropy[prediction_index] = key_ranking_sum / key_rank_executions

    print("GE = {}".format(guessing_entropy))

    return guessing_entropy
