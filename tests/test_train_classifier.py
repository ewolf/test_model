import numpy as np
import model_a.classifier_trainer as classifier_trainer


class DummyTokenizer:
    def __init__(self, outputs, vocab_size=100):
        self.outputs = outputs
        self.vocab_size = vocab_size
        self.called_with = {}

    def __call__(self, texts, truncation=False, max_length=None):
        self.called_with = {
            "texts": texts,
            "truncation": truncation,
            "max_length": max_length,
        }
        return self.outputs


def test_tokenize_for_classification_passes_truncation_and_max_length():
    tokenizer = DummyTokenizer(outputs={"input_ids": [[1], [2]]})
    batch = {"text": ["a", "b"]}

    encoded = classifier_trainer.tokenize_for_classification(batch, tokenizer, max_length=8)

    assert encoded["input_ids"] == [[1], [2]]
    assert tokenizer.called_with["truncation"] is True
    assert tokenizer.called_with["max_length"] == 8


def test_build_classifier_from_config(monkeypatch):
    dummy_tokenizer = DummyTokenizer(outputs={}, vocab_size=77)
    monkeypatch.setattr(
        classifier_trainer.BertTokenizerFast,
        "from_pretrained",
        lambda path: dummy_tokenizer,
    )

    model = classifier_trainer.build_classifier(
        tokenizer_dir="tokenizer",
        num_labels=2,
        max_position_embeddings=128,
        init_checkpoint=None,
    )
    cfg = model.config

    assert cfg.vocab_size == 77
    assert cfg.num_labels == 2
    assert cfg.max_position_embeddings == 128


def test_compute_metrics_accuracy():
    logits = np.array([[1.0, 3.0], [4.0, 0.5]])
    labels = np.array([1, 0])

    metrics = classifier_trainer.compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
