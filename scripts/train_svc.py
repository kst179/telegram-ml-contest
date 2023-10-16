import pickle

from scipy.sparse import load_npz
from sklearn.model_selection import ParameterGrid
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import numpy as np
import pickle

with open("../artifacts/svc_labels.pkl", "rb") as file:
    train_labels, val_labels = pickle.load(file)

train_matrix = load_npz("../artifacts/svc_train_norm.npz")
val_matrix = load_npz("../artifacts/svc_val_norm.npz")

grid = ParameterGrid(dict(
    C=[10, 1, 0.1],
    penalty=["l2", "l1"],
    loss=["squared_hinge", "hinge"],
))

for params in tqdm(grid):
    C = params["C"]
    penalty = params["penalty"]
    loss = params["loss"]

    try:
        svc = LinearSVC(C=C, penalty=penalty, loss=loss)
        svc.fit(train_matrix, train_labels)

        predictions = svc.predict(val_matrix)
        accuracy = accuracy_score(val_labels, predictions)

        with open(f"../artifacts/svc_C={C}_penalty={penalty}_loss={loss}_acc={accuracy}.pkl", "wb") as file:
            pickle.dump(svc, file)
    except ValueError as ex: # skip unsupported parameters combinations
        print(ex)