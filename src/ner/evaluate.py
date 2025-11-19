import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_on_test(model, loader, tag2idx, device, tag_pad_idx):
    model.eval()
    idx2tag = {v: k for k, v in tag2idx.items()}
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=-1)
            for i in range(x.size(0)):
                length = int(lengths[i])
                true = y[i][:length].cpu().numpy()
                pred = preds[i][:length].cpu().numpy()
                all_labels.extend(true.tolist())
                all_preds.extend(pred.tolist())
    target_names = [idx2tag[i] for i in range(len(idx2tag))]
    # Example: evaluate only entity tags if present
    entity_tags = [t for t in target_names if t != 'O']
    entity_indices = [tag2idx[t] for t in entity_tags if t in tag2idx]
    if entity_indices:
        report = classification_report(all_labels, all_preds, labels=entity_indices, target_names=entity_tags, zero_division=0)
        print(report)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(idx2tag))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
