from gh_dataset import GHDataset
from collections import Counter
from languages_list import Languages
import numpy as np
from matplotlib import pyplot as plt

dataset = GHDataset()

counts = Counter(dataset.labels)

labels = [Languages.to_string(lang) for lang in Languages]
occurences = np.array([counts[Languages.from_string(label)] for label in labels])

order = np.argsort(occurences)

plt.figure(figsize=[20, 40], dpi=200)
plt.barh(np.arange(len(labels)), occurences[order])
plt.yticks(np.arange(len(labels)), np.array(labels)[order], fontsize=16)
plt.xticks(fontsize=16)
plt.tick_params(labeltop=True, labelsize=16)
plt.grid(axis="x")
plt.ylim(-1, 101)
plt.tight_layout()
plt.savefig("../images/data_distribution.png")