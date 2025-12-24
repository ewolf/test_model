"""Thin wrapper for TrainingArguments to keep package-local imports tidy."""

from transformers import TrainingArguments as _HFTrainingArguments


class TrainingArguments(_HFTrainingArguments):
    """
    Direct subclass of transformers.TrainingArguments.

    Exists to give the package a stable import path (model_a.training_args.TrainingArguments)
    in case we later layer defaults or shared tweaks in one place.
    """

    pass

