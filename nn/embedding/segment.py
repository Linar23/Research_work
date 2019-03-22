import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, emb_size):
        super().__init__(3, emb_size)
