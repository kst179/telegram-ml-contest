import numpy as np
from tokenizers.implementations import ByteLevelBPETokenizer


tokenizer = ByteLevelBPETokenizer(
    "../artifacts/tokenizer-vocab.json",
    "../artifacts/tokenizer-merges.txt",
)

tokens = [
    tokenizer.decode([i]) 
    for i in range(tokenizer.get_vocab_size())
]

num_tokens = len(tokens)
offsets = [0] + [len(token.encode("utf8")) + 1 for token in tokens]
offsets = np.cumsum(offsets).tolist()
total_len = offsets[-1]
offsets = offsets[:-1]

with open("../solution/resources/tokenizer_vocab2.bin", "wb") as file:
    file.write(num_tokens.to_bytes(length=4, byteorder="little"))
    file.write(total_len.to_bytes(length=4, byteorder="little"))
    
    for offset in offsets:
        file.write(offset.to_bytes(length=8, byteorder="little"))

    for token in tokens:
        file.write(token.encode("utf8"))
        file.write(b"\0")
