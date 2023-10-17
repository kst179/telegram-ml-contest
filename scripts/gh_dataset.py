import json
from pathlib import Path

import numpy as np
import torch

from ctokenizer import CTokenizer
from languages_list import Languages


class GHDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir=Path("../data/files"),
        split_file=Path("../data/splits.json"),
        split=None,
        tokenize=False,
        subsample_lines=False,
        max_num_lines=1024,
    ):
        self.dir = dir
        self.subsample_lines = subsample_lines
        self.max_num_lines = max_num_lines
        self.split = split

        if split is not None:
            with open(split_file, "r") as file:
                splits = json.load(file)
            self.files = sorted(Path(file) for file in splits[split])
        else:
            self.files = sorted(dir.glob("*/*"))

        self.labels = [
            Languages.from_string(filepath.parent.name) for filepath in self.files
        ]

        self.tokenize = tokenize
        if tokenize:
            self.tokenizer = CTokenizer()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        content = self.files[item].read_text()
        label = self.labels[item]

        if self.subsample_lines:
            lines = content.split("\n")

            if len(lines) > 5:
                for _ in range(5):
                    size = np.random.randint(1, min(len(lines), self.max_num_lines))
                    first = np.random.randint(0, len(lines) - size)

                    content = "\n".join(lines[first : first + size]).strip()
                    if content:
                        break

            if not content:
                content = "\n".join(lines)

        if self.tokenize:
            content = self.tokenizer.encode(content)

        return content, label
