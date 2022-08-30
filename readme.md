# Data augmentation for deep learning-based profiling side-channel analysis

### Installation

```
git clone https://github.com/AISyLab/DataAugmentationDLSCA.git
cd DataAugmentationDLSCA
pip install -r requirements.txt
```

### Random Search

Hyperparameters search ranges are set in ```src/hyperparameters/random_search_ranges.py```

```
python random_search.py [dataset_name] [model_type] [leakage_model] [desync] [gaussian_noise] [time_warping] 
```

Arguments:
- ```[dataset_name]```: ```ascad-varible``` or ```dpa_v42```
- ```[model_type]```: ```CNN``` or ```MLP```
- ```[leakage_model]```: ```ID``` or ```HW```
- ```[desync]```, ```[gaussian_noise]``` and ```[time_warping]``` should be ```0``` for disable and ```1``` for enable.

Example (only desynchronization):

```
python random_search.py ascad-variable cnn ID 1 0 0 
```

Example (only Gaussian noise):

```
python random_search.py ascad-variable cnn ID 0 1 0 
```

Example (only time warping):

```
python random_search.py ascad-variable cnn ID 0 0 1 
```

Example (desynchronization and time warping):

```
python random_search.py ascad-variable cnn ID 1 0 1 
```
