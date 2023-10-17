import json

import numpy as np
from gh_dataset import GHDataset
from sklearn.model_selection import train_test_split

dataset = GHDataset()

idxs = np.arange(len(dataset))
train_gru, idxs = train_test_split(idxs, test_size=0.66, random_state=42)
train_svc, test = train_test_split(idxs, test_size=0.5, random_state=42)


splits = {
    "train_gru": [dataset.files[i].as_posix() for i in train_gru],
    "train_svc": [dataset.files[i].as_posix() for i in train_svc],
    "test": [dataset.files[i].as_posix() for i in test],
}

with open("./data/splits.json", "w") as file:
    json.dump(splits, file)