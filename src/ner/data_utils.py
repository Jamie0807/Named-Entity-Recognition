from collections import defaultdict
from typing import List, Tuple, Dict


def read_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sentences = []
    tags = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        tag_seq = []
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
    vocab = {'<PAD>': 0, '<UNK>': 1}
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
