import numpy as np
from cgru import CGRU
from gh_dataset import GHDataset
from tqdm import tqdm

gru = CGRU(
    weights_path="../solution/resources/gru_weights.bin",
    libpath="../solution/build/libgru.so",
)

test_dataset = GHDataset(split="test", tokenize=True, subsample_lines=True)

predictions = []
labels = []

order = np.arange(len(test_dataset))
np.random.shuffle(order)

for idx in tqdm(order[:50000]):
    text, label = test_dataset[idx]
    text = text[:16384]
    prediction = gru(text)
    label = label.value - 1

    predictions.append(prediction)
    labels.append(label)

predictions = np.array(predictions)
labels = np.array(labels)

print((predictions == labels).mean())

np.save("artifacts/gru_predictions_labels.npy", (predictions, labels))
