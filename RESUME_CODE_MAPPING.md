# 简历内容与代码对照文档

本文档详细说明简历中的每一项技术点在项目代码中的具体位置，方便面试准备和技术复盘。

---

## 📝 简历描述（完整版）

```
用于命名实体识别的深度学习架构比较研究

• 模型实现：对比 BiLSTM（双向 LSTM + 256维隐藏层）和 Transformer（8头自注意力 + 2层编码器）两种架构
  自定义实现正弦/余弦位置编码模块，理解 Transformer 位置信息编码的数学原理
  应用 Dropout 正则化（0.3）和 Embedding 层（128维）优化模型性能

• 数据工程：处理约 1,700 个标注句子，构建端到端 NLP 预处理流程
  实现分词、词表构建（支持 <PAD>/<UNK> 特殊标记和词频过滤）、序列填充算法
  设计自定义 Dataset 和 DataLoader，支持批量处理（batch_size=32）和 IOB 标注格式

• 训练优化：5-Fold 交叉验证 + Adam 优化器（lr=1e-3）+ 交叉熵损失函数
  训练 30 epoch，实时追踪 4 项指标（训练/验证的损失和准确率）
  自动生成训练曲线可视化，监控模型收敛状态和过拟合风险

• 评估分析：多维度评估体系 - 精确率、召回率、F1-Score（Macro）和混淆矩阵
  针对每个实体类别（PER、LOC、ORG）单独分析性能表现
  生成混淆矩阵热力图，识别模型预测的常见错误模式
  Transformer 模型 F1 约 42%（小规模数据集场景）

• 工程实践：模块化设计（数据/模型/训练/评估分离）+ CLI 接口 + 单元测试
  编写详细技术文档，包括项目结构、技术栈说明、快速开始指南
  提供灵活的参数配置，支持快速模型对比实验和超参数调优

技术栈：PyTorch, NumPy, Scikit-learn, Matplotlib, Seaborn, Pytest
```

---

## 🗺️ 代码位置对照表

### 1️⃣ 模型实现

#### ✅ BiLSTM（双向 LSTM + 256维隐藏层）
**文件**: `src/ner/models.py`  
**代码行**: 第 6-13 行

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, tagset_size)
```

**技术点**：
- ✅ `hidden_dim: int = 256` - 256维隐藏层
- ✅ `bidirectional=True` - 双向LSTM
- ✅ `hidden_dim // 2` - 因为双向LSTM，每个方向128维，合并后256维

---

#### ✅ Transformer（8头自注意力 + 2层编码器）
**文件**: `src/ner/models.py`  
**代码行**: 第 38-46 行

```python
class TransformerTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, nhead=8, num_layers=2, max_len=100):
        super(TransformerTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, tagset_size)
```

**技术点**：
- ✅ `nhead=8` - 8个注意力头
- ✅ `num_layers=2` - 2层Transformer编码器
- ✅ `nn.TransformerEncoderLayer` - PyTorch内置的Transformer层

---

#### ✅ 自定义实现正弦/余弦位置编码模块
**文件**: `src/ner/models.py`  
**代码行**: 第 22-35 行

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
```

**技术点**：
- ✅ **完全自定义实现**（不是PyTorch内置）
- ✅ 数学公式：PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- ✅ 数学公式：PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

**面试话术**：
> "我根据 Transformer 原论文（Attention is All You Need）实现了位置编码。使用正弦和余弦函数可以让模型学习相对位置关系，并且对任意长度的序列都有效。"

---

#### ✅ Dropout 正则化（0.3）
**文件**: `src/ner/models.py`  
**代码行**: 第 12 行

```python
self.dropout = nn.Dropout(0.3)
```

**技术点**：
- ✅ Dropout率 = 0.3（即训练时随机丢弃30%的神经元）
- ✅ 防止过拟合

---

#### ✅ Embedding 层（128维）
**文件**: `src/ner/models.py`  
**代码行**: 第 7 行（BiLSTM）、第 41 行（Transformer）

```python
embedding_dim: int = 128
self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
```

**技术点**：
- ✅ 将词转换为128维向量
- ✅ `padding_idx=0` - 填充位置的嵌入为0向量

---

### 2️⃣ 数据工程

#### ✅ 处理约 1,700 个标注句子
**验证方法**：
```python
# 在项目根目录运行
from src.ner.data import read_data
sentences, tags = read_data('ner_dataset.txt')
print(f"Total sentences: {len(sentences)}")
```

**实际数量**：需要运行上述代码确认

---

#### ✅ 实现分词
**文件**: `src/ner/data.py`  
**代码行**: 第 7-42 行

```python
def read_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """读取按行标注的NER数据集，返回(句子列表, 标签列表)"""
    # ...
    for line in f:
        line = line.strip()
        if not line:  # 空行分隔句子
            # ...
        parts = line.split()  # 分词：按空格分割
        word, tag = parts[0], parts[-1]  # 提取词和标签
        sentence.append(word)
        tag_seq.append(tag)
```

**技术点**：
- ✅ 按行读取，每行一个词和对应标签
- ✅ 空行分隔不同句子
- ✅ 格式：`John B-PER`

---

#### ✅ 词表构建（支持 <PAD>/<UNK> 特殊标记和词频过滤）
**文件**: `src/ner/data.py`  
**代码行**: 第 45-54 行

```python
def build_vocab(sequences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    word_freq = defaultdict(int)
    for seq in sequences:
        for token in seq:
            word_freq[token] += 1
    
    vocab = {"<PAD>": 0, "<UNK>": 1}  # 特殊标记
    for word, freq in word_freq.items():
        if freq >= min_freq and word not in vocab:  # 词频过滤
            vocab[word] = len(vocab)
    return vocab
```

**技术点**：
- ✅ `<PAD>` (索引0) - 填充标记
- ✅ `<UNK>` (索引1) - 未知词标记
- ✅ `min_freq` - 低频词过滤参数

---

#### ✅ 序列填充算法
**文件**: `src/ner/data.py`  
**代码行**: 第 78-86 行

```python
def __getitem__(self, idx: int):
    # ...
    length = len(word_ids)
    if length < self.max_len:
        # 填充（Padding）
        word_ids += [self.word2idx["<PAD>"]] * (self.max_len - length)
        tag_ids += [0] * (self.max_len - length)
    else:
        # 截断（Truncation）
        word_ids = word_ids[:self.max_len]
        tag_ids = tag_ids[:self.max_len]
```

**技术点**：
- ✅ 短序列：用 `<PAD>` 填充到 `max_len`
- ✅ 长序列：截断到 `max_len`
- ✅ `max_len=100` - 最大序列长度

---

#### ✅ 自定义 Dataset 和 DataLoader
**文件**: `src/ner/data.py`  
**代码行**: 第 64-88 行（Dataset）、第 97-100 行（DataLoader）

```python
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word2idx, tag2idx, max_len=100):
        # ...
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        # 返回：(word_ids, tag_ids, length)
        return torch.tensor(word_ids), torch.tensor(tag_ids), torch.tensor(length)

# DataLoader使用
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

**技术点**：
- ✅ 继承 PyTorch 的 `Dataset` 类
- ✅ 实现 `__len__` 和 `__getitem__` 方法
- ✅ `DataLoader` 自动批处理和打乱

---

#### ✅ batch_size=32
**文件**: `src/ner/data.py`  
**代码行**: 第 91 行

```python
def get_kfold_loaders(..., batch_size: int = 32, ...):
```

---

#### ✅ IOB 标注格式支持
**文件**: `src/ner/evaluate.py`  
**代码行**: 第 24-28 行

```python
# 过滤出实体标签（排除'O'标签）
entity_tags = [t for t in target_names if t != 'O']
entity_indices = [tag2idx[t] for t in entity_tags if t in tag2idx]
```

**技术点**：
- ✅ 支持 IOB 格式：B-PER, I-PER, B-LOC, I-LOC, O 等
- ✅ 自动识别和评估实体标签

---

### 3️⃣ 训练优化

#### ✅ 5-Fold 交叉验证
**文件**: `src/ner/data.py`  
**代码行**: 第 91-103 行

```python
def get_kfold_loaders(..., k: int = 5, ...):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_data = []
    for train_index, val_index in kf.split(sentences):
        # 生成训练集和验证集
        # ...
    return fold_data
```

**技术点**：
- ✅ K=5（5折交叉验证）
- ✅ `shuffle=True` - 随机打乱数据
- ✅ `random_state=42` - 固定随机种子，确保可复现

---

#### ✅ Adam 优化器（lr=1e-3）
**文件**: `src/ner/train.py`  
**代码行**: 第 9 行

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**技术点**：
- ✅ Adam（自适应学习率优化器）
- ✅ 学习率 = 0.001

---

#### ✅ 交叉熵损失函数
**文件**: `src/ner/train.py`  
**代码行**: 第 8 行

```python
criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)
```

**技术点**：
- ✅ `CrossEntropyLoss` - 多分类损失函数
- ✅ `ignore_index=tag_pad_idx` - 忽略填充位置的损失计算

---

#### ✅ 训练 30 epoch
**文件**: `cli.py`  
**代码行**: 第 14 行

```python
parser.add_argument('--epochs', type=int, default=3)
```

**使用方法**：
```bash
python cli.py --data ner_dataset.txt --model transformer --epochs 30 --folds 5
```

**注意**：代码默认是3个epoch，需要通过命令行参数指定30

---

#### ✅ 实时追踪 4 项指标
**文件**: `src/ner/train.py`  
**代码行**: 第 10-11、28-33 行

```python
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# 每个epoch后打印
print(f"Epoch {epoch+1}: Train Loss {total_loss:.3f}, Acc {train_accs[-1]:.3f}, Val Acc {val_acc:.3f}")
```

**技术点**：
- ✅ 训练损失（Train Loss）
- ✅ 验证损失（Val Loss）
- ✅ 训练准确率（Train Acc）
- ✅ 验证准确率（Val Acc）

---

#### ✅ 自动生成训练曲线可视化
**文件**: `src/ner/train.py`  
**代码行**: 第 35-48 行

```python
# 损失曲线
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

# 准确率曲线
plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()
```

**技术点**：
- ✅ 使用 Matplotlib 绘图
- ✅ 双曲线对比训练集和验证集
- ✅ 用于监控过拟合

---

### 4️⃣ 评估分析

#### ✅ 精确率、召回率、F1-Score（Macro）
**文件**: `src/ner/evaluate.py`  
**代码行**: 第 4、27-28 行

```python
from sklearn.metrics import classification_report, confusion_matrix

report = classification_report(all_labels, all_preds, labels=entity_indices, 
                               target_names=entity_tags, zero_division=0)
print(report)
```

**输出示例**：
```
              precision    recall  f1-score   support
     B-PER       0.65      0.58      0.61       245
     I-PER       0.72      0.68      0.70       198
     B-LOC       0.58      0.52      0.55       187
     ...
```

**技术点**：
- ✅ Precision（精确率）
- ✅ Recall（召回率）
- ✅ F1-Score（调和平均）
- ✅ Macro平均（每类权重相同）

---

#### ✅ 混淆矩阵
**文件**: `src/ner/evaluate.py`  
**代码行**: 第 30 行

```python
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(idx2tag))))
```

---

#### ✅ 针对每个实体类别单独分析
**文件**: `src/ner/evaluate.py`  
**代码行**: 第 24-28 行

```python
entity_tags = [t for t in target_names if t != 'O']
entity_indices = [tag2idx[t] for t in entity_tags if t in tag2idx]
if entity_indices:
    report = classification_report(all_labels, all_preds, labels=entity_indices, 
                                   target_names=entity_tags, zero_division=0)
```

**技术点**：
- ✅ 过滤掉 'O' 标签（非实体）
- ✅ 只评估实体标签（PER、LOC、ORG等）
- ✅ 每个类别单独计算指标

---

#### ✅ 生成混淆矩阵热力图
**文件**: `src/ner/evaluate.py`  
**代码行**: 第 31-36 行

```python
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, 
            yticklabels=target_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

**技术点**：
- ✅ 使用 Seaborn 的 heatmap
- ✅ `annot=True` - 显示数值
- ✅ 蓝色渐变色图

---

### 5️⃣ 工程实践

#### ✅ 模块化设计
**项目结构**：
```
src/ner/
├── data.py       # 数据处理模块
├── models.py     # 模型定义模块
├── train.py      # 训练逻辑模块
└── evaluate.py   # 评估分析模块
```

**技术点**：
- ✅ 单一职责原则
- ✅ 高内聚低耦合
- ✅ 易于维护和扩展

---

#### ✅ CLI 接口
**文件**: `cli.py`  
**代码行**: 第 10-15 行

```python
parser = argparse.ArgumentParser(description='NER training CLI')
parser.add_argument('--data', required=True, help='Path to dataset file')
parser.add_argument('--model', choices=['bilstm', 'transformer'], default='bilstm')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--folds', type=int, default=5)
```

**使用示例**：
```bash
python cli.py --data ner_dataset.txt --model transformer --epochs 30 --folds 5
```

---

#### ✅ 单元测试
**文件**: `tests/test_data.py`  
**代码行**: 第 4-12 行

```python
def test_read_data_and_vocabs():
    sentences, tags = read_data('tests/sample_dataset.txt')
    assert len(sentences) == 2
    assert len(tags) == 2
    assert sentences[0][0] == 'John'
    w2i = build_vocab(sentences)
    t2i = build_tag_vocab(tags)
    assert '<PAD>' in w2i
    assert '<UNK>' in w2i
```

**运行测试**：
```bash
pytest tests/test_data.py -v
```

---

#### ✅ 编写详细技术文档
**文件**: `README.md`

包含：
- 项目介绍
- 技术栈说明
- 项目结构
- 快速开始指南
- 模型架构详解
- 使用示例

---

#### ✅ 提供灵活的参数配置
**CLI参数**：
- `--data` - 数据集路径
- `--model` - 模型选择（bilstm/transformer）
- `--epochs` - 训练轮数
- `--folds` - 交叉验证折数

**代码内参数**：
- `embedding_dim`, `hidden_dim` - 可在 `models.py` 修改
- `batch_size`, `max_len` - 可在 `data.py` 修改
- `lr` - 可在 `train.py` 修改

---

## 🎯 面试准备建议

### 1. **技术深度问题准备**

**Q: 你说自定义实现了位置编码，具体是怎么做的？**

**A**: "我根据 Transformer 原论文实现了 Sinusoidal Position Encoding。具体公式是：
- 偶数维度：PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- 奇数维度：PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

代码在 `src/ner/models.py` 第22-35行。使用正弦和余弦函数的好处是可以让模型学习相对位置关系，并且对任意长度的序列都有效。"

---

**Q: 为什么 BiLSTM 的 hidden_dim 是 256，但 LSTM 内部是 hidden_dim // 2？**

**A**: "因为是双向LSTM（bidirectional=True）。前向LSTM输出128维，后向LSTM输出128维，拼接后是256维。这样可以同时捕获从左到右和从右到左的序列信息。代码在 `src/ner/models.py` 第11行。"

---

**Q: 5-Fold 交叉验证的作用是什么？**

**A**: "在数据量有限（约1700句）的情况下，单次划分可能有偶然性。5-Fold将数据分成5份，轮流用4份训练、1份验证，得到5次实验结果，更能反映模型的真实泛化能力。代码在 `src/ner/data.py` 第91-103行，使用了 sklearn 的 KFold。"

---

**Q: 交叉熵损失函数中的 ignore_index 是什么作用？**

**A**: "ignore_index=tag_pad_idx 的作用是在计算损失时忽略填充位置。因为我们对序列进行了padding，填充的部分不是真实数据，不应该参与损失计算和梯度更新。代码在 `src/ner/train.py` 第8行。"

---

### 2. **工程能力问题准备**

**Q: 你的代码是如何实现模块化的？**

**A**: "我将项目分为4个独立模块：
- `data.py` - 数据加载和预处理
- `models.py` - 模型定义
- `train.py` - 训练逻辑
- `evaluate.py` - 评估分析

每个模块职责单一，通过 `cli.py` 整合。这样易于维护、测试和扩展。"

---

**Q: 如何确保实验的可复现性？**

**A**: "我采取了以下措施：
1. 固定随机种子：KFold 中 random_state=42
2. 详细记录超参数：embedding_dim=128, hidden_dim=256 等
3. 版本控制：使用 Git 管理代码
4. 环境管理：requirements.txt 锁定依赖版本"

---

### 3. **项目改进问题准备**

**Q: F1 分数只有 42%，如何改进？**

**A**: "我分析了几个改进方向：
1. 使用预训练词向量（GloVe 300维）替代随机初始化
2. 添加 CRF 层建模标签序列依赖（BiLSTM-CRF）
3. 实现 Early Stopping 避免过拟合
4. 学习率衰减和超参数调优
5. 数据增强（同义词替换、实体替换）

这些改进预计可以将 F1 提升到 65-70%。详见 `IMPROVEMENTS.md`。"

---

## 📚 相关文件

- **项目文档**: `README.md`
- **改进建议**: `IMPROVEMENTS.md`
- **技术栈**: `requirements.txt`
- **测试文件**: `tests/test_data.py`

---

## ✅ 验证清单

在面试前，建议运行以下命令确认：

```bash
# 1. 统计数据集大小
python -c "from src.ner.data import read_data; s, t = read_data('ner_dataset.txt'); print(f'Sentences: {len(s)}')"

# 2. 运行单元测试
pytest tests/test_data.py -v

# 3. 训练模型并记录F1分数
python cli.py --data ner_dataset.txt --model transformer --epochs 30 --folds 5

# 4. 检查代码风格
# （可选）使用 black 或 flake8
```

---

**文档创建日期**: 2025-11-19  
**项目**: Named-Entity-Recognition  
**作者**: Jamie0807
