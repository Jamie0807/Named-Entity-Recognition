import argparse
import torch
from src.ner.data import read_data, build_vocab, build_tag_vocab, get_kfold_loaders
from src.ner.models import BiLSTMTagger, TransformerTagger
from src.ner.train import train_model
from src.ner.evaluate import evaluate_on_test


def main():
    parser = argparse.ArgumentParser(description='NER training CLI (from COMP534 notebook)')
    parser.add_argument('--data', required=True, help='Path to dataset file')
    parser.add_argument('--model', choices=['bilstm', 'transformer'], default='bilstm')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    sentences, tags = read_data(args.data)
    word2idx = build_vocab(sentences)
    tag2idx = build_tag_vocab(tags)
    fold_loaders = get_kfold_loaders(sentences, tags, word2idx, tag2idx, k=args.folds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag_pad_idx = 0
    trained_model = None

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n=== Fold {fold+1} ===")
        if args.model == 'bilstm':
            model = BiLSTMTagger(vocab_size=len(word2idx), tagset_size=len(tag2idx))
        else:
            model = TransformerTagger(vocab_size=len(word2idx), tagset_size=len(tag2idx))
        trained_model = train_model(model, train_loader, val_loader, tag_pad_idx, device, epochs=args.epochs)

    # Evaluate on the last fold's validation loader as a quick test
    _, test_loader = fold_loaders[-1]
    if trained_model is not None:
        evaluate_on_test(trained_model, test_loader, tag2idx, device, tag_pad_idx)


if __name__ == '__main__':
    main()
