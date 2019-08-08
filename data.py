import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.src'))
        self.valid = self.tokenize(os.path.join(path, 'valid.src'))
        self.test = self.tokenize(os.path.join(path, 'test.src'))

    def tokenize(self, path, keep_sentence_boundaries=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary

        Max_Length = 0
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                line = line.replace("\n", "")
                if line is not None:
                    print(line)
                    words = line.split() + ['<eos>']
                    if len(words) > Max_Length:
                        Max_Length = len(words)
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        if not keep_sentence_boundaries:
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    line = line.replace("\n", "")
                    if line is not None:
                        print(line)
                        words = line.split() + ['<eos>']
                        for word in words:
                            ids[token] = self.dictionary.word2idx[word]
                            token += 1
        else:
            encoded_sentences = []
            with open(path, 'r') as f:
                for line in f:
                    encsentence = []
                    line = line.replace("\n", "")
                    if line is not None:
                        print(line)
                        words = line.split() + ['<eos>']
                        for word in words:
                            encsentence.append(self.dictionary.word2idx[word])

                        if (Max_Length - len(encsentence)) > 0:
                            encsentence = torch.LongTensor(encsentence)
                            encsentence = torch.nn.functional.pad(encsentence, pad=(0, Max_Length - len(encsentence)))
                        else:
                            encsentence = torch.LongTensor(encsentence)

                        encoded_sentences.append(encsentence)
            ids = torch.stack(encoded_sentences)
        return ids