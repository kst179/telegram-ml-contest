import torch
from torch import nn
from ctokenizer import CTokenizer

tokenizer = CTokenizer()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=2**15, embedding_dim=96)
        self.gru = nn.GRU(input_size=96, hidden_size=96, batch_first=True)
        self.classifier = nn.Linear(96, 100)

    def forward(self, ids, last_elements, return_last_state=False):
        """ ids: [batch_size, seq_len]
        """
        batch_size = ids.shape[0]

        # [batch_size, seq_len, emb_dim]
        emb = self.embedding(ids)

        # [batch_size, seq_len, emb_dim]
        features, _ = self.gru(emb)

        last_feature = features[range(batch_size), last_elements]

        # [batch_size, hid_dim] -> [batch_size, 100]
        logits = self.classifier(last_feature)

        if return_last_state:
            return logits, last_feature

        return logits

state_dict = torch.load("../artifacts/gru_weights/model_88.pth", map_location="cpu")
model = Network()
model.load_state_dict(state_dict)


embeddings = state_dict["embedding.weight"]

weights_i = state_dict["gru.weight_ih_l0"]
weights_h = state_dict["gru.weight_hh_l0"]

bias_i = state_dict["gru.bias_ih_l0"]
bias_h = state_dict["gru.bias_hh_l0"]

classifier_weight = state_dict["classifier.weight"]
classifier_bias = state_dict["classifier.bias"]

classifier_weight = torch.nn.functional.pad(classifier_weight, (0, 0, 0, 4))
classifier_bias = torch.nn.functional.pad(classifier_bias, (0, 4), value=-torch.inf)

embeddings_ = embeddings @ weights_i.T + bias_i

num_embeddings, hidden_dim = embeddings.shape
num_classes, = classifier_bias.shape

with open("../solution/resources/gru_weights.bin", "wb") as file:
    file.write(hidden_dim.to_bytes(length=4, byteorder="little"))
    file.write(num_embeddings.to_bytes(length=4, byteorder="little"))
    file.write(num_classes.to_bytes(length=4, byteorder="little"))
    
    file.write(embeddings_.numpy().tobytes())
    file.write(weights_h.numpy().tobytes())
    file.write(bias_h.numpy().tobytes())

    file.write(classifier_weight.numpy().tobytes())
    file.write(classifier_bias.numpy().tobytes())