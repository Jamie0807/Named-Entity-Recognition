import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, tag_pad_idx, device, epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        for x, y, lengths in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            mask = y != tag_pad_idx
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
        train_losses.append(total_loss)
        train_accs.append(correct / total if total > 0 else 0.0)
        val_loss, val_acc = evaluate_model(model, val_loader, tag_pad_idx, device, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}: Train Loss {total_loss:.3f}, Acc {train_accs[-1]:.3f}, Val Acc {val_acc:.3f}")

    # Plot loss & acc
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.show()
    return model


def evaluate_model(model, loader, tag_pad_idx, device, criterion):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), y.view(-1))
            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            mask = y != tag_pad_idx
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    return total_loss, (correct / total) if total > 0 else 0.0
