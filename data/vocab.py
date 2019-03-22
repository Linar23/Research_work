from collections import Counter


class Vocab:
    def __init__(self, text):
        self.specials = ["<pad>", "<unk>", "<sep>", "<sos>", "<mask>"]

        self.pad_index = 0
        self.unk_index = 1
        self.sep_index = 2
        self.cls_index = 3
        self.mask_index = 4

        self.index_to_token = list(self.specials)

        counter = Counter()

        for line in text:
            words = line.replace("\n", "").replace("\\t", "").split()

            for word in words:
                counter[word] += 1

        words_and_frequencies = sorted(counter.items())

        for word, freq in words_and_frequencies:
            self.index_to_token.append(word)

        self.token_to_index = {token: i for i, token in enumerate(self.index_to_token)}

    def __len__(self):
        return len(self.index_to_token)
