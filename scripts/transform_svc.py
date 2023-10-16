import pickle
import numpy as np

num_classes = 104
num_tokens = 2**15
gru_emb_dim = 96

with open("../artifacts/svc_C=10_penalty=l2_loss=squared_hinge_acc=0.9441013725819324.pkl", "rb") as file:
   svc = pickle.load(file)

bias = np.ones(104) * -1000.0
bias[:100] = svc.intercept_

token_weights = np.zeros((num_tokens, 104))
gru_features_weights = np.zeros((104, gru_emb_dim))

token_weights[:, svc.classes_] = svc.coef_[:, :num_tokens].T
gru_features_weights[svc.classes_] = svc.coef_[:, num_tokens:]

with open("../solution/resources/svc_weights.bin", "wb") as file:
    file.write(num_classes.to_bytes(length=4, byteorder="little"))
    file.write(num_tokens.to_bytes(length=4, byteorder="little"))
    file.write(gru_emb_dim.to_bytes(length=4, byteorder="little"))

    file.write(token_weights.astype(np.float32).tobytes())
    file.write(gru_features_weights.astype(np.float32).tobytes())
    file.write(bias.astype(np.float32).tobytes())

