from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        """
        :param d_k: int scaling factor
        """
        super(ScaledDotProductAttention, self).__init__()

        self.scaling = 1 / (sqrt(d_k))

    def forward(self, q, k, v):
        """
        :param q:  An float tensor with shape of [b_s, seq_len, d_model / n_head]
        :param k: An float tensor with shape of [b_s, seq_len, d_model / n_head]
        :param v: An float tensor with shape of [b_s, seq_len, d_model / n_head]

        :return: An float tensor with shape of [b_s, seq_len, d_model / n_head]
        """
        attention = torch.bmm(q, k.transpose(1, 2)) * self.scaling

        attention = F.softmax(attention, dim=2)

        output = torch.bmm(attention, v)

        return output


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        """
        :param d_model: Int
        :param d_k: Int = d_model / n_head
        :param d_v: Int = d_model / n_head
        """
        super(SingleHeadAttention, self).__init__()

        self.q_linear = nn.Linear(d_model, d_k)
        self.k_linear = nn.Linear(d_model, d_k)
        self.v_linear = nn.Linear(d_model, d_v)

        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, q, k, v):
        """
        :param q: An float tensor with shape of [b_s, seq_len, d_model]
        :param k: An float tensor with shape of [b_s, seq_len, d_model]
        :param v: An float tensor with shape of [b_s, seq_len, d_model]

        :return: An float tensor with shape of [b_s, seq_len, d_model / n_heads]
        """
        proj_q = self.q_linear(q)
        proj_k = self.k_linear(k)
        proj_v = self.v_linear(v)

        output = self.attention(proj_q, proj_k, proj_v)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        """
        :param n_head: Int number of heads
        :param d_model: Int
        """
        super(MultiHeadAttention, self).__init__()

        d_v = int(d_model / n_head)
        d_k = int(d_model / n_head)

        self.attention = nn.ModuleList([SingleHeadAttention(d_model, d_k, d_v) for _ in range(n_head)])

        self.Linear = nn.Linear(n_head * d_v, d_model)

    def forward(self, q, k, v):
        """
        :param q: An float tensor with shape of [b_s, seq_len, d_model]
        :param k: An float tensor with shape of [b_s, seq_len, d_model]
        :param v: An float tensor with shape of [b_s, seq_len, d_model]

        :return: An float tensor with shape of [b_s, seq_len, d_model]
        """
        results = []

        for i, single_attention in enumerate(self.attention):
            attention_out = single_attention(q, k, v)
            results.append(attention_out)

        concat = torch.cat(results, dim=2)

        linear_output = self.Linear(concat)

        return linear_output
