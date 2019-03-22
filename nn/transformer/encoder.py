import torch.nn as nn

from nn.attention import MultiHeadAttention
from nn.utils import PositionWise


class Encoder(nn.Module):
    def __init__(self, embeddings, d_model, n_heads, vocab_s):
        """
        :param embeddings: An float embeddings tensor with shape [b_s, seq_len, d_model]
        :param d_model: Int size of input
        :param n_heads: Int number of heads
        :param vocab_s: Int size of vocabulary
        """
        super(Encoder, self).__init__()

        self.embeddings = embeddings
        self.vocab_s = vocab_s

        self.attention = MultiHeadAttention(n_heads, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.position_wise = PositionWise(d_model, d_model * 4)


    def forward(self, x, segment_label):
        """
        :param input: An long tensor with shape of [b_s, seq_len]

        :return: An float tensor with shape of [b_s, seq_len, vocab_size]
        """
        input = self.embeddings(x, segment_label)

        residual = input
        result = self.attention(q=input, k=input, v=input)
        result = self.layer_norm(result + residual)

        return self.position_wise(result)