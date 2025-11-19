# 命名实体识别项目

本仓库提供了一个小型Python包，用于训练和评估NER模型（BiLSTM和Transformer）。

## 📋 项目介绍

命名实体识别（Named Entity Recognition, NER）是自然语言处理（NLP）中的一项关键任务，用于从文本中识别和分类命名实体，如人名、地名、组织名等。

本项目实现了两种主流的NER模型架构：
- **BiLSTM模型**：使用双向长短期记忆网络进行序列标注
- **Transformer模型**：基于自注意力机制的现代神经网络架构

### 项目特点
- ✨ 简洁的模块化设计，易于扩展和修改
- 🎯 支持自定义数据集和模型配置
- 📊 包含完整的训练、评估和推理流程
- 🖥️ 提供命令行接口，方便使用
- 📈 自动生成训练曲线和混淆矩阵可视化

---

## 📁 项目结构

```
Named-Entity-Recognition/
├── cli.py                    # 命令行入口，用于运行训练/评估流程
├── ner_dataset.txt           # 样本数据集文件
├── requirements.txt          # Python依赖项
├── README.md                 # 项目文档
├── src/
│   ├── __init__.py
│   └── ner/
│       ├── __init__.py
│       ├── data.py           # 数据加载和预处理模块
│       ├── data_utils.py     # 数据工具函数
│       ├── models.py         # NER模型定义（BiLSTM & Transformer）
│       ├── train.py          # 训练逻辑
│       └── evaluate.py       # 评估和可视化
└── tests/
    ├── test_data.py          # 单元测试
    └── sample_dataset.txt    # 测试数据集
```

---

## 📦 核心文件说明

### src/ner/ 模块

| 文件 | 功能描述 |
|------|---------|
| **data.py** | 数据加载和预处理：读取数据、构建词表、生成DataLoader、k-fold交叉验证 |
| **models.py** | 定义两种NER模型：BiLSTMTagger和TransformerTagger，包含位置编码 |
| **train.py** | 训练循环实现：损失计算、反向传播、精度统计、绘制训练曲线 |
| **evaluate.py** | 模型评估：生成分类报告、混淆矩阵、热力图可视化 |

### 根目录文件

| 文件 | 功能描述 |
|------|---------|
| **cli.py** | 命令行入口，支持选择模型、设置训练参数 |
| **requirements.txt** | 依赖包列表（torch, numpy, scikit-learn等） |

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

数据格式：每行一个token和标签，句子之间用空行分隔
```
John       B-PER
Doe        I-PER
works      O
at         O
Google     B-ORG

Jane       B-PER
Smith      I-PER
lives      O
in         O
New        B-LOC
York       I-LOC
```

将数据文件（如 `ner_dataset.txt`）放在项目根目录。

### 3. 模型训练

#### 使用BiLSTM模型训练（推荐新手）
```bash
python cli.py --data ner_dataset.txt --model bilstm --epochs 10 --folds 5
```

#### 使用Transformer模型训练（更强大）
```bash
python cli.py --data ner_dataset.txt --model transformer --epochs 10 --folds 5
```

#### 命令行参数说明
- `--data`: 数据集文件路径（必需）
- `--model`: 模型选择，`bilstm` 或 `transformer`（默认：bilstm）
- `--epochs`: 每个fold的训练轮数（默认：3）
- `--folds`: k-fold交叉验证的fold数（默认：5）

### 4. 运行测试

```bash
pytest tests/test_data.py -v
```

---

## 🔧 模型架构详解

### BiLSTM模型
- **输入层**：词嵌入（Embedding）
- **隐藏层**：双向LSTM（Bidirectional LSTM）
- **输出层**：全连接层（FC）
- **特点**：轻量级、训练快、适合中小数据集

### Transformer模型
- **输入层**：词嵌入 + 位置编码（Positional Encoding）
- **隐藏层**：Transformer编码器（多头自注意力）
- **输出层**：全连接层
- **特点**：性能强、并行度高、适合大数据集

---

## 📊 训练过程

1. **数据加载**：读取数据并生成k-fold数据加载器
2. **模型初始化**：创建所选模型实例
3. **训练循环**：
   - 前向传播计算预测
   - 交叉熵损失计算
   - 反向传播更新参数
   - 验证集评估
4. **可视化**：绘制损失和精度曲线
5. **评估**：在测试集上计算指标和混淆矩阵

---

## 📈 输出结果

训练完成后会生成：
- 📉 **训练曲线**：损失和精度随epoch变化
- 📊 **混淆矩阵**：预测标签 vs 真实标签
- 📋 **分类报告**：Precision、Recall、F1-Score等指标

---

## 🔍 使用示例

```python
# 直接在Python中使用
from src.ner.data import read_data, build_vocab, build_tag_vocab
from src.ner.models import BiLSTMTagger
import torch

# 读取数据
sentences, tags = read_data('ner_dataset.txt')

# 构建词表
word2idx = build_vocab(sentences)
tag2idx = build_tag_vocab(tags)

# 创建模型
model = BiLSTMTagger(vocab_size=len(word2idx), tagset_size=len(tag2idx))

# 模型训练（参见 train.py）
```

---

## 💡 常见问题

**Q: BiLSTM和Transformer哪个更好？**
A: 取决于数据量和计算资源。小数据用BiLSTM，大数据用Transformer。

**Q: 如何自定义模型参数？**
A: 编辑 `cli.py` 或直接修改 `models.py` 中的模型类。

**Q: 支持GPU训练吗？**
A: 支持，代码自动检测GPU并使用（如有CUDA可用）。

---

## 📝 依赖包

- `torch`: PyTorch深度学习框架
- `numpy`: 数值计算
- `scikit-learn`: 机器学习工具
- `matplotlib`: 绘图
- `seaborn`: 统计可视化
- `pytest`: 单元测试

---

## 📄 许可证

MIT License

---

## 👤 作者

Jamie0807

---

## 📞 反馈

如有问题或建议，欢迎提Issue或Pull Request。
