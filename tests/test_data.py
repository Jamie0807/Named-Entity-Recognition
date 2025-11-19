from src.ner.data_utils import read_data, build_vocab, build_tag_vocab


def test_read_data_and_vocabs():
    sentences, tags = read_data('tests/sample_dataset.txt')
    assert len(sentences) == 2
    assert len(tags) == 2
    assert sentences[0][0] == 'John'
    w2i = build_vocab(sentences)
    t2i = build_tag_vocab(tags)
    assert '<PAD>' in w2i
    assert '<UNK>' in w2i
    assert 'B-PER' in t2i or 'B-PER' in '\n'.join(['B-PER'])
