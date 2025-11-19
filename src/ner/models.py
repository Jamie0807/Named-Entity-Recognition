"""
NER模型定义模块
包含两种序列标注模型：BiLSTM 和 Transformer
"""
import math
import torch
import torch.nn as nn


class BiLSTMTagger(nn.Module):
    """
    双向LSTM序列标注模型
    用于命名实体识别（NER）任务
    
    模型架构：
    Embedding层(128维) -> BiLSTM(256维) -> Dropout(0.3) -> 全连接层
    """
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        """
        初始化BiLSTM模型
        
        参数:
            vocab_size: 词表大小
            tagset_size: 标签集大小（如 B-PER, I-PER, O 等）
            embedding_dim: 词嵌入维度，默认128
            hidden_dim: LSTM隐藏层维度，默认256（双向LSTM每个方向128维）
        """
        super(BiLSTMTagger, self).__init__()
        # 词嵌入层：将词ID转换为128维向量，padding_idx=0表示填充位置的嵌入为0向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 双向LSTM层：hidden_dim//2 是因为双向LSTM会拼接两个方向的输出
        # 前向LSTM输出128维 + 后向LSTM输出128维 = 256维
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, 
                           bidirectional=True, batch_first=True)
        
        # Dropout正则化：训练时随机丢弃30%的神经元，防止过拟合
        self.dropout = nn.Dropout(0.3)
        
        # 全连接输出层：将LSTM输出映射到标签空间
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入的词ID序列，shape: (batch_size, seq_len)
        
        返回:
            tag_space: 每个位置的标签得分，shape: (batch_size, seq_len, tagset_size)
        """
        # 步骤1: 词嵌入 (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        emb = self.embedding(x)
        
        # 步骤2: BiLSTM编码 -> (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(emb)
        
        # 步骤3: Dropout正则化
        out = self.dropout(lstm_out)
        
        # 步骤4: 全连接层映射到标签空间 -> (batch_size, seq_len, tagset_size)
        tag_space = self.fc(out)
        return tag_space


class PositionalEncoding(nn.Module):
    """
    位置编码模块（自定义实现）
    为Transformer提供序列位置信息
    
    使用正弦和余弦函数编码位置：
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参考论文：Attention is All You Need (Vaswani et al., 2017)
    """
    def __init__(self, d_model, max_len=100):
        """
        初始化位置编码
        
        参数:
            d_model: 模型维度（embedding_dim）
            max_len: 最大序列长度，默认100
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 (max_len, 1)：[0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算分母项：10000^(2i/d_model)
        # 对于不同维度使用不同的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 偶数维度使用sin函数
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 奇数维度使用cos函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加batch维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 将位置编码注册为buffer（不参与梯度更新，但会被保存到模型state_dict中）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将位置编码添加到输入
        
        参数:
            x: 输入的词嵌入，shape: (batch_size, seq_len, d_model)
        
        返回:
            添加位置编码后的输入，shape不变
        """
        # 将位置编码加到输入上（只使用实际序列长度的部分）
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerTagger(nn.Module):
    """
    基于Transformer的序列标注模型
    用于命名实体识别（NER）任务
    
    模型架构：
    Embedding层(128维) -> 位置编码 -> Transformer编码器(8头,2层) -> 全连接层
    """
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, nhead=8, num_layers=2, max_len=100):
        """
        初始化Transformer模型
        
        参数:
            vocab_size: 词表大小
            tagset_size: 标签集大小
            embedding_dim: 词嵌入维度，默认128
            nhead: 多头注意力的头数，默认8
            num_layers: Transformer编码器层数，默认2
            max_len: 最大序列长度，默认100
        """
        super(TransformerTagger, self).__init__()
        
        # 词嵌入层：将词ID转换为128维向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 位置编码模块（自定义实现）
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len)
        
        # Transformer编码器层：包含多头自注意力和前馈网络
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        
        # 堆叠多层Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全连接输出层：将Transformer输出映射到标签空间
        self.fc = nn.Linear(embedding_dim, tagset_size)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入的词ID序列，shape: (batch_size, seq_len)
        
        返回:
            tag_space: 每个位置的标签得分，shape: (batch_size, seq_len, tagset_size)
        """
        # 步骤1: 词嵌入 (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        emb = self.embedding(x)
        
        # 步骤2: 添加位置编码（Transformer无法感知位置，需要显式添加位置信息）
        emb = self.pos_encoder(emb)
        
        # 步骤3: 转换维度顺序 (batch_size, seq_len, embedding_dim) -> (seq_len, batch_size, embedding_dim)
        # PyTorch的Transformer期望输入为 (seq_len, batch_size, embedding_dim)
        emb = emb.permute(1, 0, 2)
        
        # 步骤4: Transformer编码
        out = self.transformer_encoder(emb)
        
        # 步骤5: 转换回原来的维度顺序 (seq_len, batch_size, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        out = out.permute(1, 0, 2)
        
        # 步骤6: 全连接层映射到标签空间 -> (batch_size, seq_len, tagset_size)
        tag_space = self.fc(out)
        return tag_space
