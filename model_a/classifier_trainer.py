import os
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def _has_tokenizer_files(path: str) -> bool:
    expected = {"tokenizer.json", "vocab.txt"}
    present = set(os.listdir(path)) if os.path.isdir(path) else set()
    return bool(expected & present)


def load_tokenizer(tokenizer_path: str, fallback_id: Optional[str] = "bert-base-uncased") -> BertTokenizerFast:
    """
    Load a tokenizer from a local directory, falling back to a hub ID if the directory is empty.
    """
    if os.path.isdir(tokenizer_path) and not _has_tokenizer_files(tokenizer_path):
        if fallback_id:
            print(f"No tokenizer files found in '{tokenizer_path}', falling back to '{fallback_id}'.")
            tokenizer_path = fallback_id
        else:
            raise FileNotFoundError(
                f"No tokenizer files found in '{tokenizer_path}'. "
                "Place tokenizer.json or vocab.txt there, or provide a hub ID."
            )
    return BertTokenizerFast.from_pretrained(tokenizer_path)


def tokenize_for_classification(
    examples: Dict[str, List[str]], tokenizer: BertTokenizerFast, max_length: int
) -> Dict[str, List[List[int]]]:
    """Tokenize text for sequence classification with truncation."""
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def build_classifier(
    tokenizer_dir: str,
    num_labels: int,
    max_position_embeddings: int,
    init_checkpoint: str | None = None,
) -> BertForSequenceClassification:
    """
    Build a sequence-classification model. If an init checkpoint (e.g., MLM pretrain)
    exists, load it; otherwise create a fresh config aligned with the tokenizer.
    """
    if init_checkpoint and os.path.isdir(init_checkpoint):
        return BertForSequenceClassification.from_pretrained(
            init_checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    tok = load_tokenizer(tokenizer_dir)
    config = BertConfig(
        vocab_size=tok.vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=2,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(config)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy}


def main():
    tokenizer_dir = os.environ.get("TOKENIZER_PATH", "tokenizer")
    tokenizer_fallback = os.environ.get("TOKENIZER_FALLBACK", "bert-base-uncased")
    init_checkpoint = "mlm_out"  # optional: use MLM-pretrained weights if present
    out_dir = "cls_out"
    max_length = 256

    tokenizer = load_tokenizer(tokenizer_dir, fallback_id=tokenizer_fallback)
    ds = load_dataset("rotten_tomatoes")

    ds_enc = ds.map(
        lambda batch: tokenize_for_classification(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    ds_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    model = build_classifier(
        tokenizer_dir=tokenizer_dir,
        num_labels=2,
        max_position_embeddings=max_length,
        init_checkpoint=init_checkpoint,
    )

    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_enc["train"],
        eval_dataset=ds_enc["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(ds_enc["test"])
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
