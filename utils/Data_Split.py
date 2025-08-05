############ seperating the dataset to training and validation
# The training set is 80% of the data, and the validation set is 20
from pathlib import Path
# from data_loading import BasicDataset, CarvanaDataset, PerfusionDataset
import random
# from torch.utils.data import random_split
# import torch
import json
from os.path import splitext, isfile, join
from os import listdir

dir_img = Path('./data_perfusion/images/')
dir_mask = Path('./data_perfusion/labels/')
random_seed = 2025
val_percent = 0.2

# dataset = PerfusionDataset(dir_img, dir_mask)

dataset_names = [splitext(file)[0] for file in listdir(dir_img) if isfile(join(dir_img, file)) and not file.startswith('.')]

random.seed(2025)  # for reproducibility

total = len(dataset_names)
val_size = int(val_percent * total)
val_names = random.sample(dataset_names, val_size)
train_names = list(set(dataset_names) - set(val_names))

n_val = len(val_names)
n_train = len(train_names)
print("total, n_train, n_val: ",total, n_train, n_val)

# Create the dictionary
dataset_info = {
    "train_filenames": train_names,
    "val_filenames": val_names,
}

# Save to JSON
with open("dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=4)
