import matplotlib.pyplot as plt
import numpy as np

folder_results = "D:/postdoc/paper_data_augmentation/random_search"
dataset_name = "ascad-variable"
model_type = "CNN"
leakage_model = "ID"
hiding = "_desync_gaussian_noise"
file_id = 1

filepath = f"{folder_results}/{dataset_name}_{model_type}_{leakage_model}{hiding}_{file_id}.npz"
npz_file = np.load(filepath, allow_pickle=True)

dataset_parameters = npz_file["dataset"]
guessing_entropy = npz_file["GE"]

figure = plt.gcf()
figure.set_size_inches(5, 4)
plt.plot(guessing_entropy, linewidth=1)
plt.grid(True, which="both", ls="-")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Guessing Entropy", fontsize=12)
plt.tight_layout()
# plt.savefig(plot_filename)
# plt.close()
plt.show()
