# 命名实体识别项目

本仓库提供了一个小型Python包，用于训练和评估NER模型（BiLSTM和Transformer）。

> 📖 **面试准备？** 查看 [简历内容与代码对照文档](RESUME_CODE_MAPPING.md)，详细说明简历中每一项技术点在代码中的具体位置！

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

## �️ 技术栈

### 深度学习框架

#### 🔥 PyTorch
- **版本**: Latest stable
- **用途**: 核心深度学习框架
- **具体应用**:
  - `torch.nn` - 神经网络模块
  - `torch.optim` - 优化器（Adam）
  - `torch.utils.data` - 数据加载器
  - 自动微分和梯度计算
  - GPU 加速支持（CUDA）

### 深度学习模型架构

#### 🧠 BiLSTM（双向长短期记忆网络）
- **组件**:
  - `nn.Embedding` - 词嵌入层（128维）
  - `nn.LSTM` - 双向LSTM（256维隐藏层）
  - `nn.Dropout` - Dropout正则化（0.3）
  - `nn.Linear` - 全连接输出层

#### 🤖 Transformer
- **组件**:
  - `nn.Embedding` - 词嵌入层
  - 位置编码（Positional Encoding）
  - `nn.TransformerEncoder` - Transformer编码器
  - 多头注意力机制（8个头）
  - 2层编码器堆叠

### 数据处理与科学计算

#### 📊 NumPy
- 数值计算基础库
- 数组操作和数学运算

#### 🔬 Scikit-learn
- **K-Fold交叉验证** (`KFold`)
- **评估指标**:
  - `classification_report` - 精确率、召回率、F1分数
  - `confusion_matrix` - 混淆矩阵
- 数据划分和预处理

### 数据可视化

#### 📈 Matplotlib
- 绘制训练损失曲线
- 绘制精度曲线
- 图表展示和保存

#### 🎨 Seaborn
- 美化混淆矩阵热力图
- 统计数据可视化
- 基于Matplotlib的高级可视化

### 测试框架

#### ✅ Pytest
- 单元测试框架
- 测试数据读取和处理函数
- 自动化测试运行

### 机器学习技术

#### 🎯 训练技术
- **K-Fold交叉验证** - 5折验证，提高模型评估可靠性
- **批量训练** (Batch Training) - 高效数据处理
- **反向传播** (Backpropagation) - 梯度下降优化
- **Adam优化器** - 自适应学习率优化
- **交叉熵损失** (Cross-Entropy Loss) - 序列标注损失函数
- **Dropout正则化** - 防止过拟合

#### 📊 评估指标
- **Accuracy**（准确率） - 整体预测正确率
- **Precision**（精确率） - 预测为正的样本中真正为正的比例
- **Recall**（召回率） - 真正为正的样本中被正确预测的比例
- **F1-Score**（F1分数） - 精确率和召回率的调和平均
- **Confusion Matrix**（混淆矩阵） - 分类结果可视化

### 自然语言处理技术

#### 📝 NLP核心技术
- **序列标注** (Sequence Labeling) - 为序列中的每个元素分配标签
- **词嵌入** (Word Embedding) - 将词转换为稠密向量表示
- **命名实体识别** (Named Entity Recognition) - 识别文本中的实体
- **IOB标注格式** (Inside-Outside-Begin) - B-PER, I-PER, O 等标签

### 技术栈总结

| 类别 | 技术/工具 | 版本 | 用途 |
|------|-----------|------|------|
| **深度学习框架** | PyTorch | Latest | 模型构建与训练 |
| **科学计算** | NumPy | Latest | 数值计算 |
| **机器学习** | Scikit-learn | Latest | 评估与验证 |
| **数据可视化** | Matplotlib | Latest | 绘制曲线图 |
| **统计可视化** | Seaborn | Latest | 热力图 |
| **测试框架** | Pytest | Latest | 单元测试 |
| **编程语言** | Python | 3.13+ | 主要开发语言 |
| **模型架构** | BiLSTM | - | 序列标注模型 |
| **模型架构** | Transformer | - | 自注意力模型 |
| **NLP任务** | NER | - | 命名实体识别 |

### 技术亮点 ✨

1. **双模型支持** - BiLSTM 和 Transformer 可选，适应不同场景
2. **K-Fold交叉验证** - 提高模型评估的可靠性和泛化能力
3. **自动化可视化** - 训练过程和结果自动生成图表
4. **模块化设计** - 数据、模型、训练、评估清晰分离
5. **GPU自动检测** - 智能使用CUDA加速训练
6. **类型安全** - 使用Python类型注解，提高代码质量

### 适用场景 💡

- 📚 学术研究和论文实验
- 🎓 NLP课程作业和项目
- 🏢 中小规模NER任务
- 🔰 深度学习入门学习
- 🧪 模型对比实验

---

## �📝 依赖包

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

## � 相关文档

- 📖 **[简历内容与代码对照文档](RESUME_CODE_MAPPING.md)** - 面试准备必看！详细说明简历中每一项技术点的代码位置
- 💡 **[项目改进建议](IMPROVEMENTS.md)** - 提升模型性能的具体方案

---

## �👤 作者

Jamie0807

---

## 📞 反馈

如有问题或建议，欢迎提Issue或Pull Request。
