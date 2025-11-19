# 命名实体识别项目

本仓库提供了一个小型Python包，用于训练和评估NER模型（BiLSTM和Transformer）。

## 项目介绍

命名实体识别（Named Entity Recognition, NER）是自然语言处理（NLP）中的一项关键任务，用于从文本中识别和分类命名实体，如人名、地名、组织名等。

本项目实现了两种主流的NER模型架构：
- **BiLSTM模型**：使用双向长短期记忆网络进行序列标注
- **Transformer模型**：基于自注意力机制的现代神经网络架构

项目特点：
- 简洁的模块化设计，易于扩展和修改
- 支持自定义数据集和模型配置
- 包含完整的训练、评估和推理流程
- 提供命令行接口，方便使用

## ## 项目结构
- src/ner: 包含数据加载器、模型、训练和评估代码的包。
- cli.py: 用于运行训练/评估流程的简单命令行入口。
- requirements.txt: Python依赖项。
- tests/: 数据读取器的最小单元测试。

## 快速开始
1. 创建虚拟环境并安装依赖项：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 将数据集文件放在此文件夹中（格式：每行一个token和标签，句子之间用空行分隔），例如 `ner_dataset.txt`。

3. 运行训练（示例）：

```bash
python cli.py --data ner_dataset.txt --model bilstm --epochs 3
```
