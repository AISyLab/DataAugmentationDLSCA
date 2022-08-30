from tensorflow.keras.callbacks import *


class SaveModel(Callback):
    def __init__(self, path, dataset_name, model_type, leakage_model, npoi, attack_type, search_id):
        super().__init__()
        self.path = path
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.leakage_model = leakage_model
        self.npoi = npoi
        self.attack_type = attack_type
        self.search_id = search_id

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(
            f"{self.path}/model_epoch_{epoch}_{self.dataset_name}_{self.model_type}_{self.leakage_model}_{self.attack_type}_{self.npoi}_poi_{self.search_id}.h5")
