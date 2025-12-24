import os
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
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

def build_model(tokenizer_dir: str, max_position_embeddings: int = 512) -> BertForMaskedLM:
    """
    Build a SMALL BERT-like encoder from scratch.

    Key config fields:
      - vocab_size: must match tokenizer vocab
      - hidden_size: embedding + model width
      - num_hidden_layers: transformer block count
      - num_attention_heads: attention heads per block
      - intermediate_size: FFN width inside each block
    """
    tok = load_tokenizer(tokenizer_dir)

    config = BertConfig(
        vocab_size=tok.vocab_size,
        max_position_embeddings=max_position_embeddings,
        # Small model (fits “few MB corpus” better than a big one)
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,      # hidden_size must be divisible by heads
        intermediate_size=1536,     # typically 4x hidden_size
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=2,
    )
    model = BertForMaskedLM(config)
    return model


def tokenize_and_group(
    examples: Dict[str, List[str]],
    tokenizer: BertTokenizerFast,
    block_size: int,
) -> Dict[str, List[List[int]]]:
    """
    Tokenize and pack text into fixed-length blocks for MLM.

    Parameters:
      - block_size: number of tokens per training example
    """
    # Tokenize (no padding here; we pack later)
    tokenized = tokenizer(
        examples["text"],
        truncation=False,
        add_special_tokens=True,
    )

    # Concatenate
    concatenated = {k: sum(tokenized[k], []) for k in tokenized.keys()}
    total_len = len(concatenated["input_ids"])

    # Drop remainder so every chunk is exact block_size
    total_len = (total_len // block_size) * block_size

    result = {}
    for k, seq in concatenated.items():
        seq = seq[:total_len]
        result[k] = [seq[i : i + block_size] for i in range(0, total_len, block_size)]

    return result

def main():
    tokenizer_dir = os.environ.get("TOKENIZER_PATH", "tokenizer")
    tokenizer_fallback = os.environ.get("TOKENIZER_FALLBACK", "bert-base-uncased")
    out_dir       = "mlm_out"

    ds = load_dataset("rotten_tomatoes")
    # Remove labels for unsupervised MLM.
    if "label" in ds["train"].column_names:
        ds = ds.remove_columns("label")

    tokenizer = load_tokenizer(tokenizer_dir, fallback_id=tokenizer_fallback)

    block_size = 256

    ds_tok = ds.map(
        lambda ex: tokenize_and_group(ex, tokenizer, block_size),
        batched=True,
        remove_columns=["text"],
    )

    # MLM data collator randomly masks tokens.
    # mlm_probability = fraction of tokens selected for masking
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    model = build_model(tokenizer_dir, max_position_embeddings=block_size)

    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=200,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        fp16=torch.cuda.is_available(),   # enables fp16 on GPU automatically
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save “final” model and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Saved MLM-pretrained model to: {out_dir}")


if __name__ == "__main__":
    main()
