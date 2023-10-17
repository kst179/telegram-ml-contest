from gh_dataset import GHDataset
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

output_dir = Path("../artifacts")
output_dir.mkdir(exist_ok=True)

dataset = GHDataset()
paths = [file.as_posix() for file in dataset.files]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=paths,
    vocab_size=2**15,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
    show_progress=True,
)

tokenizer.save_model(output_dir.as_posix(), "tokenizer")
