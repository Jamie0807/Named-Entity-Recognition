"""
数据加载和预处理模块
负责读取NER数据集、构建词表、创建Dataset和DataLoader
"""
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from collections import defaultdict
from typing import List, Tuple, Dict


def read_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    读取按行标注的NER数据集
    
    数据格式：
        每行包含一个词和对应的标签，用空格分隔
        句子之间用空行分隔
        
    示例：
        John    B-PER
        Doe     I-PER
        works   O
        at      O
        Google  B-ORG
        
        (空行)
        
        Jane    B-PER
        ...
    
    参数:
        file_path: 数据集文件路径
    
    返回:
        sentences: 句子列表，每个句子是词的列表
        tags: 标签列表，每个标签序列对应一个句子
    """
    sentences: List[List[str]] = []  # 存储所有句子
    tags: List[List[str]] = []       # 存储所有标签序列

    with open(file_path, "r", encoding="utf-8") as f:
        sentence: List[str] = []     # 当前句子的词列表
        tag_seq: List[str] = []      # 当前句子的标签列表

        for line in f:
            line = line.strip()      # 去除首尾空白字符
            
            # 空行表示句子结束
            if not line:
                if sentence:         # 如果当前句子非空，保存它
                    sentences.append(sentence)
                    tags.append(tag_seq)
                    sentence = []    # 重置，准备读取下一个句子
                    tag_seq = []
                continue

            # 分割词和标签
            parts = line.split()
            if len(parts) < 2:
                # 忽略格式错误的行
                continue
            
            word, tag = parts[0], parts[-1]  # 第一个是词，最后一个是标签
            sentence.append(word)
            tag_seq.append(tag)

        # 处理文件末尾的最后一个句子（如果没有空行结尾）
        if sentence:
            sentences.append(sentence)
            tags.append(tag_seq)

    return sentences, tags


def build_vocab(sequences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """
    构建词表（word to index 映射）
    
    参数:
        sequences: 句子列表，每个句子是词的列表
        min_freq: 最小词频阈值，低于此频率的词会被忽略，默认为1（保留所有词）
    
    返回:
        vocab: 词到索引的映射字典
            特殊标记：
            - <PAD> (索引0): 填充标记
            - <UNK> (索引1): 未知词标记
    """
    # 统计词频
    word_freq = defaultdict(int)
    for seq in sequences:
        for token in seq:
            word_freq[token] += 1

    # 初始化词表，添加特殊标记
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    # 将满足最小词频的词添加到词表
    for word, freq in word_freq.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)  # 按顺序分配索引
    
    return vocab


def build_tag_vocab(tag_sequences: List[List[str]]) -> Dict[str, int]:
    """
    构建标签表（tag to index 映射）
    
    参数:
        tag_sequences: 标签序列列表，每个标签序列对应一个句子
    
    返回:
        tag_vocab: 标签到索引的映射字典
            例如: {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, ...}
    """
    tag_vocab: Dict[str, int] = {}
    
    for seq in tag_sequences:
        for tag in seq:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)  # 按出现顺序分配索引
    
    return tag_vocab


class NERDataset(Dataset):
    """
    命名实体识别数据集类
    继承自PyTorch的Dataset，用于批量数据加载
    
    功能：
    1. 将词和标签转换为索引
    2. 对序列进行填充（padding）或截断（truncation）
    3. 返回固定长度的张量
    """
    def __init__(self, sentences: List[List[str]], tags: List[List[str]], 
                 word2idx: Dict[str, int], tag2idx: Dict[str, int], max_len: int = 100):
        """
        初始化数据集
        
        参数:
            sentences: 句子列表
            tags: 标签列表
            word2idx: 词到索引的映射
            tag2idx: 标签到索引的映射
            max_len: 最大序列长度，超过此长度会截断，不足会填充，默认100
        """
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sentences)

    def __getitem__(self, idx: int):
        """
        获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            word_ids: 词ID序列张量，shape: (max_len,)
            tag_ids: 标签ID序列张量，shape: (max_len,)
            length: 原始序列长度（填充前），用于后续忽略填充位置
        """
        sentence = self.sentences[idx]
        tag_seq = self.tags[idx]
        
        # 将词转换为索引，未知词使用<UNK>的索引
        word_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in sentence]
        
        # 将标签转换为索引
        tag_ids = [self.tag2idx[t] for t in tag_seq]
        
        # 记录原始长度（填充前）
        length = len(word_ids)
        
        # 序列填充或截断
        if length < self.max_len:
            # 短序列：填充到max_len
            word_ids += [self.word2idx["<PAD>"]] * (self.max_len - length)
            tag_ids += [0] * (self.max_len - length)  # 标签也填充为0
        else:
            # 长序列：截断到max_len
            word_ids = word_ids[:self.max_len]
            tag_ids = tag_ids[:self.max_len]
        
        # 转换为PyTorch张量
        return (torch.tensor(word_ids, dtype=torch.long), 
                torch.tensor(tag_ids, dtype=torch.long), 
                torch.tensor(length, dtype=torch.long))


def get_kfold_loaders(sentences: List[List[str]], tags: List[List[str]], 
                     word2idx: Dict[str, int], tag2idx: Dict[str, int], 
                     k: int = 5, batch_size: int = 32, max_len: int = 100):
    """
    生成K折交叉验证的数据加载器
    
    将数据集分成k份，轮流使用其中k-1份作为训练集，1份作为验证集
    这样可以更可靠地评估模型性能，避免单次划分的偶然性
    
    参数:
        sentences: 句子列表
        tags: 标签列表
        word2idx: 词到索引的映射
        tag2idx: 标签到索引的映射
        k: 折数，默认5（即5折交叉验证）
        batch_size: 批大小，默认32
        max_len: 最大序列长度，默认100
    
    返回:
        fold_data: 列表，每个元素是一个(train_loader, val_loader)元组
    """
    # 创建KFold对象
    # shuffle=True: 随机打乱数据
    # random_state=42: 固定随机种子，确保实验可复现
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_data = []
    
    # 迭代每一折
    for train_index, val_index in kf.split(sentences):
        # 根据索引划分训练集和验证集
        train_sents = [sentences[i] for i in train_index]
        train_tags = [tags[i] for i in train_index]
        val_sents = [sentences[i] for i in val_index]
        val_tags = [tags[i] for i in val_index]
        
        # 创建训练集和验证集的Dataset
        train_dataset = NERDataset(train_sents, train_tags, word2idx, tag2idx, max_len)
        val_dataset = NERDataset(val_sents, val_tags, word2idx, tag2idx, max_len)
        
        # 创建DataLoader
        # 训练集shuffle=True：每个epoch打乱数据顺序
        # 验证集shuffle=False：保持固定顺序
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        fold_data.append((train_loader, val_loader))
    
    return fold_data
