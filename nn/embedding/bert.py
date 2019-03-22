import torch.nn as nn
from .token import TokenEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        """
        :param vocab_size: Int size of vocabulary
        :param emb_size: Int size of embedding
        """
        super(BERTEmbedding, self).__init__()

        self.v_s = vocab_size
        self.e_s = emb_size

        self.token = TokenEmbedding(self.v_s, self.e_s)
        self.segment = SegmentEmbedding(self.e_s)

    def forward(self, seq, segment_label):
        """
        :param input: An long tensor with shape of [b_s, seq_len]
        :return: An float tensor with shape of [b_s, seq_len, emb_size]
        """

        return self.token(seq) + self.segment(segment_label)
