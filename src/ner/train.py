"""
模型训练模块
负责模型训练循环、损失计算、性能评估和可视化
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, tag_pad_idx, device, epochs=10):
    """
    训练模型
    
    训练流程：
    1. 前向传播计算预测
    2. 计算损失函数
    3. 反向传播计算梯度
    4. 更新模型参数
    5. 在验证集上评估
    6. 生成训练曲线可视化
    
    参数:
        model: 要训练的模型（BiLSTM或Transformer）
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
        tag_pad_idx: 填充标签的索引（通常为0），用于在损失计算时忽略填充位置
        device: 训练设备（'cpu' 或 'cuda'）
        epochs: 训练轮数，默认10
    
    返回:
        model: 训练后的模型
    """
    # 将模型移到指定设备（GPU或CPU）
    model = model.to(device)
    
    # 定义损失函数：交叉熵损失，忽略填充位置
    # ignore_index=tag_pad_idx 表示不计算填充位置的损失和梯度
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)
    
    # 定义优化器：Adam，学习率1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 用于记录训练过程的指标
    train_losses, val_losses = [], []  # 训练损失和验证损失
    train_accs, val_accs = [], []      # 训练准确率和验证准确率

    # 训练循环
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()  # 设置为训练模式（启用Dropout等）
        total_loss = 0.0
        correct, total = 0, 0
        
        # 遍历训练集的每个batch
        for x, y, lengths in train_loader:
            # 将数据移到指定设备
            x, y = x.to(device), y.to(device)
            
            # 清空之前的梯度
            optimizer.zero_grad()
            
            # 前向传播：计算模型输出
            outputs = model(x)  # shape: (batch_size, seq_len, tagset_size)
            
            # 计算损失
            # view(-1, ...) 将输出展平为 (batch_size*seq_len, tagset_size)
            # y.view(-1) 将标签展平为 (batch_size*seq_len,)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), y.view(-1))
            
            # 反向传播：计算梯度
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            
            # 计算准确率（只计算非填充位置）
            preds = outputs.argmax(dim=-1)  # 获取每个位置得分最高的标签
            mask = y != tag_pad_idx         # 创建mask，过滤掉填充位置
            correct += (preds[mask] == y[mask]).sum().item()  # 统计正确预测的数量
            total += mask.sum().item()                         # 统计总的有效位置数
        
        # 记录本轮的训练指标
        train_losses.append(total_loss)
        train_accs.append(correct / total if total > 0 else 0.0)
        
        # ========== 验证阶段 ==========
        val_loss, val_acc = evaluate_model(model, val_loader, tag_pad_idx, device, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印本轮训练结果
        print(f"Epoch {epoch+1}: Train Loss {total_loss:.3f}, "
              f"Acc {train_accs[-1]:.3f}, Val Acc {val_acc:.3f}")

    # ========== 可视化训练过程 ==========
    # 绘制损失曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
    return model


def evaluate_model(model, loader, tag_pad_idx, device, criterion):
    """
    在验证集或测试集上评估模型
    
    参数:
        model: 要评估的模型
        loader: 数据加载器（验证集或测试集）
        tag_pad_idx: 填充标签的索引
        device: 设备（'cpu' 或 'cuda'）
        criterion: 损失函数
    
    返回:
        total_loss: 总损失
        accuracy: 准确率（只计算非填充位置）
    """
    model.eval()  # 设置为评估模式（禁用Dropout等）
    total_loss = 0.0
    correct, total = 0, 0
    
    # 不计算梯度（节省内存和计算）
    with torch.no_grad():
        for x, y, lengths in loader:
            # 将数据移到指定设备
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            outputs = model(x)
            
            # 计算损失
            loss = criterion(outputs.view(-1, outputs.shape[-1]), y.view(-1))
            total_loss += loss.item()
            
            # 计算准确率（只计算非填充位置）
            preds = outputs.argmax(dim=-1)
            mask = y != tag_pad_idx
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    
    # 返回总损失和准确率
    return total_loss, (correct / total) if total > 0 else 0.0
