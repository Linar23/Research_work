import torch.nn as nn


class PositionWise(nn.Module):
    def __init__(self, size, inner_size):
        """
        :param size: Int input size
        :param inner_size: Int inner size of position wise
        """
        super(PositionWise, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(size, inner_size),
            nn.ReLU(),
            nn.Linear(inner_size, size)
        )

        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [b_s, seq_len, embedding_s]
        :return: An float tensor with shape of [b_s, seq_len, embedding_s]
        """
        residual = input

        result = self.fc(input)

        return self.layer_norm(result + residual)