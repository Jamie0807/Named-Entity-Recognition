"""
模型评估模块
负责在测试集上评估模型性能，生成分类报告和混淆矩阵
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_on_test(model, loader, tag2idx, device, tag_pad_idx):
    """
    在测试集上全面评估模型性能
    
    评估指标：
    1. 精确率（Precision）：预测为正的样本中真正为正的比例
    2. 召回率（Recall）：真正为正的样本中被正确预测的比例
    3. F1-Score：精确率和召回率的调和平均
    4. 混淆矩阵：展示每种标签的预测分布
    
    参数:
        model: 要评估的模型
        loader: 测试集DataLoader
        tag2idx: 标签到索引的映射字典
        device: 设备（'cpu' 或 'cuda'）
        tag_pad_idx: 填充标签的索引
    """
    # 设置为评估模式
    model.eval()
    
    # 创建索引到标签的反向映射
    idx2tag = {v: k for k, v in tag2idx.items()}
    
    # 收集所有预测结果和真实标签
    all_preds = []   # 所有预测的标签索引
    all_labels = []  # 所有真实的标签索引
    
    # 不计算梯度
    with torch.no_grad():
        for x, y, lengths in loader:
            # 将数据移到设备
            x, y = x.to(device), y.to(device)
            
            # 模型预测
            outputs = model(x)
            preds = outputs.argmax(dim=-1)  # 获取每个位置得分最高的标签
            
            # 遍历batch中的每个样本
            for i in range(x.size(0)):
                length = int(lengths[i])  # 获取原始序列长度（不包括填充）
                
                # 只取有效长度的部分（忽略填充）
                true = y[i][:length].cpu().numpy()
                pred = preds[i][:length].cpu().numpy()
                
                # 将结果添加到总列表中
                all_labels.extend(true.tolist())
                all_preds.extend(pred.tolist())
    
    # ========== 生成分类报告 ==========
    # 获取所有标签名称
    target_names = [idx2tag[i] for i in range(len(idx2tag))]
    
    # 过滤出实体标签（排除'O'标签）
    # 因为通常我们更关心实体的识别效果，而不是非实体('O')
    entity_tags = [t for t in target_names if t != 'O']
    entity_indices = [tag2idx[t] for t in entity_tags if t in tag2idx]
    
    # 如果存在实体标签，打印详细的分类报告
    if entity_indices:
        print("\n" + "="*60)
        print("分类报告（只显示实体标签）")
        print("="*60)
        report = classification_report(
            all_labels, 
            all_preds, 
            labels=entity_indices,      # 只评估实体标签
            target_names=entity_tags,   # 标签名称
            zero_division=0             # 避免除零警告
        )
        print(report)
        print("\n报告说明：")
        print("- precision（精确率）：预测为该类别的样本中，真正属于该类别的比例")
        print("- recall（召回率）：真正属于该类别的样本中，被正确预测的比例")
        print("- f1-score：精确率和召回率的调和平均，综合评价指标")
        print("- support：该类别在测试集中的真实样本数量")
    
    # ========== 生成混淆矩阵热力图 ==========
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(idx2tag))))
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,              # 显示数值
        fmt='d',                 # 整数格式
        xticklabels=target_names,  # X轴标签
        yticklabels=target_names,  # Y轴标签
        cmap='Blues'             # 蓝色渐变
    )
    plt.xlabel('Predicted')  # X轴：预测标签
    plt.ylabel('True')       # Y轴：真实标签
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    print("\n混淆矩阵说明：")
    print("- 对角线：预测正确的数量（越高越好）")
    print("- 非对角线：预测错误的数量")
    print("- 每一行：该真实标签被预测为各种标签的分布")
    print("- 每一列：该预测标签来自各种真实标签的分布")
