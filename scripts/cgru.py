import ctypes

# from ctokenizer import CTokenizer
import numpy as np


class Matrix(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("stride", ctypes.c_int),
    ]


class CGRU:
    def __init__(
        self,
        libpath="../solution/build/libgru.so",
        weights_path="../solution/resources/gru_weights.bin",
    ):
        self.lib = ctypes.CDLL(libpath)
        self.lib.createGRU.argtypes = [ctypes.c_char_p]
        self.lib.createGRU.restype = ctypes.POINTER(ctypes.c_void_p)

        self.lib.predictGRU.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # GRU obj
            ctypes.POINTER(ctypes.c_int),  # tokens
            ctypes.c_int,  # num_tokens
        ]
        self.lib.predictGRU.restype = ctypes.c_int

        self.lib.getLastStateGRU.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # GRU obj
            ctypes.POINTER(ctypes.c_int),  # tokens
            ctypes.c_int,  # num_tokens
        ]
        self.lib.getLastStateGRU.restype = Matrix

        self.gru = self.lib.createGRU(weights_path.encode())

    def get_last_state(self, tokens):
        tokens_array = (ctypes.c_int * len(tokens))()
        for i, token in enumerate(tokens):
            tokens_array[i] = token

        last_state = self.lib.getLastStateGRU(
            self.gru, tokens_array, ctypes.c_int(len(tokens))
        )
        last_state = np.ctypeslib.as_array(last_state.data, (last_state.n,))

        return last_state

    def __call__(self, tokens):
        tokens_array = (ctypes.c_int * len(tokens))()
        for i, token in enumerate(tokens):
            tokens_array[i] = token

        prediction = self.lib.predictGRU(
            self.gru, tokens_array, ctypes.c_int(len(tokens))
        )

        return prediction
