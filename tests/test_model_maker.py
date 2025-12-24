import importlib
import sys
import types

import pytest



class DummyTokenizer:
    def __init__(self, outputs, vocab_size=100):
        self.outputs = outputs
        self.vocab_size = vocab_size

    def __call__(self, texts, truncation=False, add_special_tokens=True):
        # Mimic transformers fast tokenizer behavior: return per-field lists of lists.
        return self.outputs


@pytest.fixture
def model_maker(monkeypatch):
    # Provide a stub dataset module so importing model_maker does not attempt to load real data.
    dummy_dataset_module = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "dataset", dummy_dataset_module)

    import model_a.maker as module

    importlib.reload(module)
    return module


def test_tokenize_and_group_flattens_and_chunks(model_maker):
    outputs = {
        "input_ids": [[101, 11, 12], [101, 21]],
        "attention_mask": [[1, 1, 1], [1, 1]],
    }
    tokenizer = DummyTokenizer(outputs)

    examples = {"text": ["foo", "bar"]}
    result = model_maker.tokenize_and_group(examples, tokenizer, block_size=3)

    assert result["input_ids"] == [[101, 11, 12]]
    assert result["attention_mask"] == [[1, 1, 1]]


def test_build_model_uses_tokenizer_vocab(monkeypatch, model_maker):
    dummy_tokenizer = DummyTokenizer(outputs={}, vocab_size=321)

    def fake_from_pretrained(path):
        return dummy_tokenizer

    monkeypatch.setattr(model_maker.BertTokenizerFast, "from_pretrained", fake_from_pretrained)

    model = model_maker.build_model("unused", max_position_embeddings=128)
    cfg = model.config

    assert cfg.vocab_size == 321
    assert cfg.max_position_embeddings == 128
    assert cfg.hidden_size == 384
    assert cfg.num_attention_heads == 6
