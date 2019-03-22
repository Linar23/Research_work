import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, emb_size):
        super().__init__(vocab_size, emb_size)