import ctypes


class CTokenizer:
    def __init__(self):
        self.lib = ctypes.CDLL("../solution/build/libtokenizer.so")
        self.lib.createTokenizer.argtypes = [ctypes.c_char_p]
        self.lib.createTokenizer.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.tokenize.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ]

        self.tokenizer_p = self.lib.createTokenizer(
            "solution/resources/tokenizer_vocab.bin".encode()
        )

    def __call__(self, string):
        return self.encode(string)

    def encode(self, string):
        num_tokens = ctypes.c_int()
        tokens = ctypes.POINTER(ctypes.c_int)()

        self.lib.tokenize(
            self.tokenizer_p,
            string.encode(),
            ctypes.byref(num_tokens),
            ctypes.byref(tokens),
        )
        tokens_list = [tokens[i] for i in range(num_tokens.value)]
        self.lib.free(tokens)

        return tokens_list
