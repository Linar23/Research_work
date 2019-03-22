import torch.nn as nn
from torch.utils.data import DataLoader
from data import BERTDataset, Vocab
from nn.embedding.bert import BERTEmbedding
from nn.transformer.encoder import Encoder
from torch.optim import Adam


class Model(nn.Module):
    def __init__(self, n_heads, vocab_size, embedding_size):
        """
        :param n_heads: Int number of heads
        :param vocab_size: Int size of vocabulary
        :param embedding_size: Int embedding size
        """
        super(Model, self).__init__()

        self.embed = BERTEmbedding(vocab_size, embedding_size)

        self.d_model = self.embed.e_s
        self.v_s = self.embed.v_s

        self.encoder = Encoder(self.embed, self.d_model, n_heads, self.v_s)
        self.next_sentence = NextSentencePrediction(self.d_model)
        self.mask_lm = MaskedLanguageModel(self.d_model, self.v_s)

    def forward(self, x, segment_label):
        """
        :param input:  An float tensor with shape of [b_s, seq_len]
        :param target:  An float tensor with shape of [b_s, seq_len]
        """
        prediction = self.encoder(x, segment_label)

        return self.next_sentence(prediction), self.mask_lm(prediction)


class NextSentencePrediction(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


train_dataset_path = "data/train.txt"

with open(train_dataset_path, "r") as f:
    vocab = Vocab(f)

seq_len = 15
emb_size = 100
epochs = 100

train_dataset = BERTDataset(train_dataset_path, vocab, seq_len)
train_data_loader = DataLoader(train_dataset, batch_size=2)

model = Model(2, len(vocab), emb_size)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optim = Adam(model.parameters(), lr=0.1)

for l in range(epochs):
    for i, data in enumerate(train_data_loader):
        next_sent_output, mask_lm_output = model.forward(data["bert_input"], data["segment_label"])

        next_loss = criterion(next_sent_output, data["is_next"])
        mask_loss = criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

        loss = next_loss + mask_loss
        print(loss)
        loss.backward()
        optim.step()
