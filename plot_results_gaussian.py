import matplotlib.pyplot as plt
import numpy as np
import os

folder_results = "D:/postdoc/paper_data_augmentation/random_search"
dataset_name = "dpa_v42"
model_type = "CNN"
leakage_model = "HW"
std = 8.0
std_augmentation = 0

figure = plt.gcf()
figure.set_size_inches(5, 4)

for file_id in range(1000):
    filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}_std_{std}_std_augmentation_{std_augmentation}_gaussian_noise_{file_id}.npz"
    if os.path.exists(filepath):
        npz_file = np.load(filepath, allow_pickle=True)

        guessing_entropy = npz_file["GE"]
        if guessing_entropy[len(guessing_entropy) - 1] < 10:
            plt.plot(guessing_entropy, linewidth=1)
            print(file_id, npz_file["NT"])

plt.grid(True, which="both", ls="-")
plt.xlabel("Attack Traces", fontsize=12)
plt.ylabel("Guessing Entropy", fontsize=12)
plt.tight_layout()
# plt.savefig(plot_filename)
# plt.close()
plt.show()
