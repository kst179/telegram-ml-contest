import numpy as np
from scipy.sparse import csr_array
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import save_npz
from tqdm import tqdm

from cgru import CGRU
from gh_dataset import GHDataset


def normalize_sparse_mat(x):
    row_sum = x.sum(axis=1)
    x.data = x.data / np.repeat(row_sum, np.diff(x.indptr))
    return x


gru = CGRU(
    weights_path="../solution/resources/gru_weights.bin",
    libpath="../solution/build/libgru.so",
)

train_data = GHDataset(split="train_svc", tokenize=True, subsample_lines=True)
val_data = GHDataset(split="test", tokenize=True, subsample_lines=True)

for dataset, name in (
    (train_data, "../artifacts/svc_train.npz"),
    (val_data, "../artifacts/svc_val.npz"),
):
    num_rows = len(dataset)
    num_cols = 2**15

    data = []
    indices = []
    indptr = [0]

    states = np.empty((len(dataset), 96))
    # labels = []

    for i, (tokens, label) in enumerate(tqdm(dataset)):
        tokens = tokens[:8192]

        last_state = gru.get_last_state(tokens)
        states[i, ...] = last_state.copy()
        # labels.append(label.value - 1)

        ids, counts = np.unique(tokens, return_counts=True)
        data.extend(counts)
        indices.extend(ids)
        indptr.append(indptr[-1] + len(ids))

    # labels = np.array(labels)

    data_matrix = csr_array((data, indices, indptr), shape=(num_rows, num_cols))
    data_matrix = normalize_sparse_mat(data_matrix)
    data_matrix = sparse_hstack((data_matrix, states))

    save_npz(name, data_matrix)
    del data_matrix  # free some memory cause even sparse matrices are quite big
