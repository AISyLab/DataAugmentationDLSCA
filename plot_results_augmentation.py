import matplotlib.pyplot as plt
import numpy as np
import os

folder_results = "D:/postdoc/paper_data_augmentation/data_augmentation/data_augmentation_per_epoch/augmented_and_original_traces"
dataset_name = "ascad-variable"
model_type = "CNN"
leakage_model = "ID"

n_prof = 200000

file_id_dict = {
    25: 521,
    50: 194,
    75: 112,
    100: 122,
    125: 330,
    150: 268,
    175: 66,
    200: 74
}

desync_level_augmentation_dict = {
    25: [12, 25, 50],
    50: [12, 25, 50, 75],
    75: [12, 25, 50, 75, 100],
    100: [12, 25, 50, 75, 100, 125],
    125: [12, 25, 50, 75, 100, 125, 150],
    150: [12, 25, 50, 75, 100, 125, 150, 175],
    175: [12, 25, 50, 75, 100, 125, 150, 175, 200],
    200: [12, 25, 50, 75, 100, 125, 150, 175, 200]
}

cell_color_dict = {
    0: "\cellcolor{red!50}",
    1: "\cellcolor{red!48}",
    2: "\cellcolor{red!46}",
    3: "\cellcolor{red!44}",
    4: "\cellcolor{red!42}",
    5: "\cellcolor{red!40}",
    6: "\cellcolor{red!38}",
    7: "\cellcolor{red!36}",
    8: "\cellcolor{red!34}",
    9: "\cellcolor{red!32}",
    10: "\cellcolor{red!30}",
    11: "\cellcolor{red!28}",
    12: "\cellcolor{red!26}",
    13: "\cellcolor{red!24}",
    14: "\cellcolor{red!22}",
    15: "\cellcolor{red!20}",
    16: "\cellcolor{red!18}",
    17: "\cellcolor{red!16}",
    18: "\cellcolor{red!14}",
    19: "\cellcolor{red!12}",
    20: "\cellcolor{red!10}",
    21: "\cellcolor{red!8}",
    22: "\cellcolor{red!6}",
    23: "\cellcolor{red!4}",
    24: "\cellcolor{red!2}",
    25: "\cellcolor{teal!2}",
    26: "\cellcolor{teal!4}",
    27: "\cellcolor{teal!6}",
    28: "\cellcolor{teal!8}",
    29: "\cellcolor{teal!10}",
    30: "\cellcolor{teal!12}",
    31: "\cellcolor{teal!14}",
    32: "\cellcolor{teal!16}",
    33: "\cellcolor{teal!18}",
    34: "\cellcolor{teal!20}",
    35: "\cellcolor{teal!22}",
    36: "\cellcolor{teal!24}",
    37: "\cellcolor{teal!26}",
    38: "\cellcolor{teal!28}",
    39: "\cellcolor{teal!30}",
    40: "\cellcolor{teal!32}",
    41: "\cellcolor{teal!34}",
    42: "\cellcolor{teal!36}",
    43: "\cellcolor{teal!38}",
    44: "\cellcolor{teal!40}",
    45: "\cellcolor{teal!42}",
    46: "\cellcolor{teal!44}",
    47: "\cellcolor{teal!46}",
    48: "\cellcolor{teal!48}",
    49: "\cellcolor{teal!50}",
    50: "\cellcolor{teal!50}",
}

colors = ["blue", "red", "green", "orange", "purple", "yellow", "grey", "black"]

for desync_level in [25, 50, 75, 100, 125, 150, 200]:

    figure = plt.gcf()
    figure.set_size_inches(6, 3)

    nts_all = []

    for color, desync_level_augmentation in enumerate(desync_level_augmentation_dict[desync_level]):

        nts = []
        ges = []

        for n_augmented in range(20000, 210000, 20000):

            dataset = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_"
            countermeasure = f"desync_level_{desync_level}_desync_level_augmentation_{desync_level_augmentation}_"
            augmentation = f"n_prof_{n_prof}_n_augmented_{n_augmented}_desync_file_id_{file_id_dict[desync_level]}"

            file_count = 0
            ge = 0
            nt = 0

            for file_index in range(1, 4):
                filepath = f"{dataset}{countermeasure}{augmentation}_{file_index}.npz"

                if os.path.exists(filepath):
                    npz_file = np.load(filepath, allow_pickle=True)

                    ge += npz_file["GE"]
                    nt += npz_file["NT"]

                    file_count += 1

            ge /= file_count
            nt /= file_count

            ges.append(ge[len(ge) - 1])
            nts.append(nt)

            plt.subplot(1, 2, 1)
            plt.scatter(n_augmented, ge[len(ge) - 1], color=colors[color])
            plt.subplot(1, 2, 2)
            plt.scatter(n_augmented, nt, color=colors[color])

        nts_all.append(nts)

        plt.subplot(1, 2, 1)
        plt.plot(list(range(20000, 210000, 20000)), ges, color=colors[color], label=f"da - {desync_level_augmentation}")
        plt.subplot(1, 2, 2)
        plt.plot(list(range(20000, 210000, 20000)), nts, color=colors[color], label=f"da - {desync_level_augmentation}")

    nts_all = np.array(nts_all)

    min_nt = np.min(nts_all)

    for color, desync_level_augmentation in enumerate(desync_level_augmentation_dict[desync_level]):
        nts_string = ""
        for nt in nts_all[color]:
            if nt == 3000:
                nts_string += f"& - "
            else:
                cell_value = int(min_nt * 50 / nt)
                # cell_color = "\cellcolor{teal!" + f"{cell_value}" + "}"
                cell_color = f"{cell_color_dict[cell_value]}"
                nts_string += f"& {cell_color}{int(nt)} "
        if color == 0:
            print(
                "\multirow{" + f"{len(desync_level_augmentation_dict[desync_level])}" + "}{*}{" + f"{desync_level}" + "}" + f" & {desync_level_augmentation} {nts_string}" + "\\\\")
        else:
            if color == len(desync_level_augmentation_dict[desync_level]) - 1:
                print(f"& {desync_level_augmentation} {nts_string}" + "\\\\ \hline")
            else:
                print(f"& {desync_level_augmentation} {nts_string}" + "\\\\")

    plt.subplot(1, 2, 1)
    plt.grid(True, which="both", ls="-")
    plt.xlabel("n_augmented", fontsize=12)
    plt.ylabel("Guessing Entropy", fontsize=12)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.grid(True, which="both", ls="-")
    plt.xlabel("n_augmented", fontsize=12)
    plt.ylabel("$N_{GE = 0}$", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"D:/postdoc/paper_data_augmentation/{dataset_name}_{model_type}_{leakage_model}_desync_level_{desync_level}.png")
    plt.close()
    # plt.show()
