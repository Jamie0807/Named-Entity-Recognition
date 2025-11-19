"""
命令行接口（CLI）
用于运行NER模型的训练和评估流程

使用示例：
    # 使用BiLSTM模型训练（默认3个epoch，5折交叉验证）
    python cli.py --data ner_dataset.txt --model bilstm
    
    # 使用Transformer模型训练30个epoch
    python cli.py --data ner_dataset.txt --model transformer --epochs 30 --folds 5
"""
import argparse
import torch
from src.ner.data import read_data, build_vocab, build_tag_vocab, get_kfold_loaders
from src.ner.models import BiLSTMTagger, TransformerTagger
from src.ner.train import train_model
from src.ner.evaluate import evaluate_on_test


def main():
    """主函数：解析命令行参数，执行训练和评估流程"""
    
    # ========== 解析命令行参数 ==========
    parser = argparse.ArgumentParser(
        description='NER模型训练CLI - 支持BiLSTM和Transformer两种架构'
    )
    
    # 必需参数：数据集路径
    parser.add_argument(
        '--data', 
        required=True, 
        help='数据集文件路径（格式：每行一个词和标签，句子间用空行分隔）'
    )
    
    # 可选参数：模型类型
    parser.add_argument(
        '--model', 
        choices=['bilstm', 'transformer'], 
        default='bilstm',
        help='选择模型架构：bilstm（双向LSTM）或 transformer（自注意力）'
    )
    
    # 可选参数：训练轮数
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=3,
        help='每个fold的训练轮数（默认3）'
    )
    
    # 可选参数：交叉验证折数
    parser.add_argument(
        '--folds', 
        type=int, 
        default=5,
        help='K折交叉验证的折数（默认5）'
    )
    
    args = parser.parse_args()

    # ========== 数据加载和预处理 ==========
    print("="*60)
    print("步骤1: 加载数据集...")
    print("="*60)
    
    # 读取数据
    sentences, tags = read_data(args.data)
    print(f"✓ 成功加载 {len(sentences)} 个句子")
    
    # 构建词表和标签表
    word2idx = build_vocab(sentences)
    tag2idx = build_tag_vocab(tags)
    print(f"✓ 词表大小: {len(word2idx)} (包含<PAD>和<UNK>)")
    print(f"✓ 标签集大小: {len(tag2idx)}")
    print(f"✓ 标签类别: {list(tag2idx.keys())}")
    
    # 生成K折交叉验证的数据加载器
    fold_loaders = get_kfold_loaders(sentences, tags, word2idx, tag2idx, k=args.folds)
    print(f"✓ 生成 {args.folds} 折交叉验证数据")

    # ========== 设备配置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 填充标签的索引（通常为0）
    tag_pad_idx = 0
    
    trained_model = None

    # ========== K折交叉验证训练 ==========
    print("\n" + "="*60)
    print(f"步骤2: 开始 {args.folds} 折交叉验证训练...")
    print(f"模型: {args.model.upper()}")
    print(f"每折训练轮数: {args.epochs}")
    print("="*60)
    
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{args.folds}")
        print(f"{'='*60}")
        
        # 根据选择创建模型
        if args.model == 'bilstm':
            model = BiLSTMTagger(
                vocab_size=len(word2idx), 
                tagset_size=len(tag2idx)
            )
            print("✓ 创建 BiLSTM 模型（256维隐藏层，128维嵌入）")
        else:
            model = TransformerTagger(
                vocab_size=len(word2idx), 
                tagset_size=len(tag2idx)
            )
            print("✓ 创建 Transformer 模型（8头注意力，2层编码器）")
        
        # 训练模型
        trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            tag_pad_idx, 
            device, 
            epochs=args.epochs
        )

    # ========== 最终评估 ==========
    print("\n" + "="*60)
    print("步骤3: 在最后一折的验证集上进行最终评估...")
    print("="*60)
    
    # 使用最后一折的验证集作为测试集
    _, test_loader = fold_loaders[-1]
    
    if trained_model is not None:
        evaluate_on_test(trained_model, test_loader, tag2idx, device, tag_pad_idx)
        print("\n✓ 训练和评估完成！")
    else:
        print("\n✗ 训练失败")


if __name__ == '__main__':
    main()
