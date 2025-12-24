"""
Lightweight reference trainer skeleton to show how to wire up a model, data loader,
and optimization loop. Use this as a starting point for custom training flows.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from .training_args import TrainingArguments


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        tokenizer=None,
    ):
        # Store the components needed for training/eval.
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer

        # Basic device handling; extend with distributed logic if needed.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Simple optimizer; swap for AdamW/SGD/scheduler as required.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.args, "learning_rate", 5e-4),
        )

    def _dataloader(self, dataset, batch_size: Optional[int] = None) -> DataLoader:
        """
        Build a torch DataLoader using the provided data_collator to handle padding
        and tensor conversion. Batch size defaults to per_device_train_batch_size.
        """
        bs = batch_size or getattr(self.args, "per_device_train_batch_size", 8)
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def train(self):
        """
        Minimal training loop:
          - iterates epochs
          - builds batches via the data collator
          - runs forward/backward/step
        Extend this to add logging, gradient clipping, schedulers, eval, etc.
        """
        self.model.train()
        dataloader = self._dataloader(self.train_dataset)
        epochs = int(getattr(self.args, "num_train_epochs", 1))

        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                # Move batch to target device.
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass; assumes model returns loss when labels are present.
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else None

                if loss is None:
                    continue  # No loss computed; skip.

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Hook for logging/monitoring.
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            # Optional: run evaluation per epoch.
            if self.eval_dataset is not None:
                self.evaluate()

    def evaluate(self):
        """
        Minimal eval loop computing average loss. For metrics, add your own
        compute_metrics callback and aggregation.
        """
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        dataloader = self._dataloader(
            self.eval_dataset,
            batch_size=getattr(self.args, "per_device_eval_batch_size", None),
        )
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else None
                if loss is not None:
                    losses.append(loss.item())

        self.model.train()
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"eval_loss: {avg_loss:.4f}")
            return {"eval_loss": avg_loss}
        return {}

