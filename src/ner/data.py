import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from collections import defaultdict
from typing import List, Tuple, Dict


def read_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Read a token-per-line NER dataset. Returns (sentences, tags).

    File format: each non-empty line contains "word TAG". Sentences separated by blank lines.
    """
    sentences: List[List[str]] = []
    tags: List[List[str]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        sentence: List[str] = []
        tag_seq: List[str] = []

        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_seq)
                    sentence = []
                    tag_seq = []
                continue

            parts = line.split()
            if len(parts) < 2:
                # ignore malformed lines
                continue
            word, tag = parts[0], parts[-1]
            sentence.append(word)
            tag_seq.append(tag)

        if sentence:
            sentences.append(sentence)
            tags.append(tag_seq)

    return sentences, tags


def build_vocab(sequences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    word_freq = defaultdict(int)
    for seq in sequences:
        for token in seq:
            word_freq[token] += 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in word_freq.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def build_tag_vocab(tag_sequences: List[List[str]]) -> Dict[str, int]:
    tag_vocab: Dict[str, int] = {}
    for seq in tag_sequences:
        for tag in seq:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)
    return tag_vocab


class NERDataset(Dataset):
    def __init__(self, sentences: List[List[str]], tags: List[List[str]], word2idx: Dict[str, int], tag2idx: Dict[str, int], max_len: int = 100):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        sentence = self.sentences[idx]
        tag_seq = self.tags[idx]
        word_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in sentence]
        tag_ids = [self.tag2idx[t] for t in tag_seq]
        length = len(word_ids)
        if length < self.max_len:
            word_ids += [self.word2idx["<PAD>"]] * (self.max_len - length)
            tag_ids += [0] * (self.max_len - length)
        else:
            word_ids = word_ids[:self.max_len]
            tag_ids = tag_ids[:self.max_len]
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long), torch.tensor(length, dtype=torch.long)


def get_kfold_loaders(sentences: List[List[str]], tags: List[List[str]], word2idx: Dict[str, int], tag2idx: Dict[str, int], k: int = 5, batch_size: int = 32, max_len: int = 100):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_data = []
    for train_index, val_index in kf.split(sentences):
        train_sents = [sentences[i] for i in train_index]
        train_tags = [tags[i] for i in train_index]
        val_sents = [sentences[i] for i in val_index]
        val_tags = [tags[i] for i in val_index]
        train_dataset = NERDataset(train_sents, train_tags, word2idx, tag2idx, max_len)
        val_dataset = NERDataset(val_sents, val_tags, word2idx, tag2idx, max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        fold_data.append((train_loader, val_loader))
    return fold_data
