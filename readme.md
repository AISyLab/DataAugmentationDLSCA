# Data augmentation for deep learning-based profiling side-channel analysis

### Installation

```
git clone https://github.com/AISyLab/DataAugmentationDLSCA.git
cd DataAugmentationDLSCA
pip install -r requirements.txt
```

### Setting folder paths

User needs to define the paths in:
- ```script_search.py``` lines 16 (root folder, where ```script_search.py``` is located), 26 (traces folder) and 27 (results folder).

Simply copy the ```ascad-variable.h5``` (traceset with 1400 samples per trace) into the trace folder.

### Running random search with and without data augmentation

In ```script_search.py```, line 29, variable ```data_augmentation``` defines if data augmentation is implemented or not during training.

### Analysis parameters for data augmentation

- Number of augmented traces: in ```script_search.py``` file, there is a structure called ```dataset_parameters``` where the user sets the amount of augmented profiling traces through "n_profiling_augmented" option. 
- Desynchronization level: also set to max of 50 samples. Desynchronization levels follows a Gaussian distribution.
- Gaussian noise standard deviation: in ```src/preprocess/generate_hiding_countermeasures.py```, function ```make_gaussian_noise```, the level of noise can be set by changing the ```std``` variable. Default value is 5.
- Gaussian noise standard deviation in augmentation: in ```src/training/data_augmentation.py```, line 75, ```std``` variable defines the level of noise for data augmentation. Default is 1.

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

```[desync]```, ```[gaussian_noise]``` and ```[time_warping]``` defines which type of hiding countermeasures we add to original dataset.

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


### Running with sbatch

```
sbatch run_search.sbatch ascad-variable cnn ID 1 0 0 
```

