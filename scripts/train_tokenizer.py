from tokenizers import ByteLevelBPETokenizer

from gh_dataset import GHDataset

dataset = GHDataset()
paths = [file.as_posix() for file in dataset.files]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=2**15, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
], show_progress=True)

tokenizer.save_model("../artifacts", "tokenizer")