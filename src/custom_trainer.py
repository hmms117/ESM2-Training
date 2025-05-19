import math
import random
import torch
from torch.utils.data import get_worker_info
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler
from collections import defaultdict
from transformers import Trainer
from datasets import load_from_disk
from time import time

from muon_optimizer import Muon


class ShardBatchIterable(IterableDataset):
    """
    An IterableDataset that processes one shard at a time, sorts batches by length,
    and distributes batches across GPUs using distributed sampling behavior.
    """

    def __init__(self, shard_paths, group_by_batch_fn, epoch, seed, rank, world_size, args, sort_batches_by_length=True):
        """
        Args:
            shard_paths (list): Paths to .arrow shard files.
            group_by_batch_fn (function): Function to group examples by batch_id.
            epoch (int): Current epoch for shuffling.
            seed (int): Seed for deterministic behavior.
            rank (int): Process rank (e.g., GPU rank).
            world_size (int): Total number of processes (e.g., total GPUs).
            args: Training arguments (e.g., gradient_accumulation_steps).
            sort_batches_by_length (bool): Whether to sort batches within each shard by batch length.
        """
        super().__init__()
        self.shard_paths = shard_paths
        self.group_by_batch_fn = group_by_batch_fn
        self.epoch = epoch
        self.real_epoch = 0
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.sort_batches_by_length = sort_batches_by_length

    def __iter__(self):
        # Worker info for multi-worker partitioning
        worker_info = get_worker_info()
        if worker_info is None:
            # Single worker
            worker_id = 0
            num_workers = 1
        else:
            # Multi-worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Shuffle shards for each epoch
        rng = random.Random(self.seed + int(self.epoch))
        shards = self.shard_paths[:]
        rng.shuffle(shards)

        if self.rank == 0:  # Only rank 0 logs shard count
            print(f"[Rank {self.rank}] Starting epoch {self.epoch}")

        for shard_idx, shard_path in enumerate(shards):
            if self.rank == 0:  # Only rank 0 logs shard loading
                print(f"[Rank {self.rank}] Loading shard {shard_idx + 1}/{len(shards)}: {shard_path}")

            # Load shard and group examples into batches
            shard_data = load_from_disk(shard_path)
            examples = list(shard_data)
            batches = self.group_by_batch_fn(examples)

            if self.sort_batches_by_length:
                batches.sort(key=lambda batch: len(batch[-1]["input_ids"]))  # Sort by input length

            # Distribute batches across workers
            total_batches = len(batches)
            batches_per_worker = total_batches // num_workers
            worker_batches = batches[
                worker_id * batches_per_worker : (worker_id + 1) * batches_per_worker
            ]

            rng.shuffle(worker_batches)  # Shuffle batches within worker

            # Use DistributedSampler for rank-based partitioning
            sampler = DistributedSampler(
                worker_batches, num_replicas=self.world_size, rank=self.rank, shuffle=False
            )
            sampler.set_epoch(self.real_epoch)

            for batch_idx in sampler:
                yield worker_batches[batch_idx]
        
        # Increment the epoch counter
        self.real_epoch += 1

class CustomTrainer(Trainer):
     #<<<<< codex/add-largest-benefits-from-modded-m-nnogpt
     def __init__(
        self,
        train_dataset,
        eval_dataset,
        data_collator,
        gradient_clipping,
        beta_1,
        beta_2,
        epsilon,
        weight_decay,
        optimizer_config=None, 
        *args,
        **kwargs,
    ):

        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            *args,
            **kwargs,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.gradient_clipping = gradient_clipping

        self.optimizer_config = optimizer_config or {}

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.true_global_step = 0

    def create_optimizer(self):
        if self.optimizer is None:
            opt_type = str(self.optimizer_config.get("type", "adamw")).lower()
            if opt_type == "muon":
                from muon import Muon
                lr = self.args.learning_rate
                beta1 = self.optimizer_config.get("beta_1", 0.99)
                beta2 = self.optimizer_config.get("beta_2", 0.98)
                eps = self.optimizer_config.get("epsilon", 1e-12)
                weight_decay = self.optimizer_config.get("weight_decay", 0.0)
                self.optimizer = Muon(
                    self.model.parameters(), lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay
                )
            else:
                from torch.optim import AdamW
                self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)


            self.optimizer = Muon(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
                weight_decay=self.weight_decay,
            )
        return self.optimizer


    def group_by_batch(self, dataset):
        """
        Groups a list of examples into lumps keyed by 'batch_id'.
        Each 'batch_id' becomes one list (lump).
        """
        grouped_data = defaultdict(list)
        for example in dataset:
            grouped_data[example["batch_id"]].append(example)
        return list(grouped_data.values())

    @staticmethod
    def create_collate_fn(base_collator, keys_to_remove=None):
        """
        Returns a collate function that removes unwanted keys from each example,
        then calls 'base_collator' on the cleaned examples.
        """
        if keys_to_remove is None:
            keys_to_remove = []

        def custom_collate_fn(batch_list):
            lumps = batch_list[0]  # Batch size is 1, so this unpacks the single lump
            for ex in lumps:
                for key in keys_to_remove:
                    ex.pop(key, None)
            return base_collator(lumps)

        return custom_collate_fn
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        loss = loss/self.args.gradient_accumulation_steps
        return loss

    def get_train_dataloader(self):
        """
        Uses the updated ShardBatchIterable to load one shard at a time,
        sort batches by length, and distribute them across processes.
        """
        if not isinstance(self.train_dataset, list):
            return super().get_train_dataloader()

        epoch = int(self.state.epoch) if self.state.epoch is not None else 0

        shard_iterable = ShardBatchIterable(
            shard_paths=self.train_dataset,
            group_by_batch_fn=self.group_by_batch,
            epoch=epoch,
            seed=self.args.seed,
            rank=self.args.local_rank,  # GPU rank
            world_size=self.args.world_size,  # Total number of GPUs
            args=self.args,  # Pass the training arguments here
            sort_batches_by_length=True,  # Sort pre-batched lumps by length
        )

        collate_fn = self.create_collate_fn(
            base_collator=self.data_collator,
            keys_to_remove=["batch_id", "id", "sequence_length"],
        )

        return DataLoader(
            shard_iterable,
            batch_size=1,  # Each lump (pre-batched examples) is treated as one batch
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Simple evaluation data loader (single GPU). No distributed evaluation.
        """
        eval_dataset = eval_dataset or self.eval_dataset
        all_ex = list(eval_dataset)  # Load all examples into memory
        grouped = self.group_by_batch(all_ex)

        collate_fn = self.create_collate_fn(
            base_collator=self.data_collator,
            keys_to_remove=["batch_id", "id", "sequence_length"],
        )

        return DataLoader(
            grouped,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Overridden for on-the-fly perplexity or other metrics, ignoring unmasked tokens.
        """
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            return {}

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model = self.model
        device = self.args.device
        model.eval()
        model.to(device)

        total_loss = 0.0
        total_masked_tokens = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs

                masked_tokens = (batch["labels"] != -100).sum().item()
                total_loss += loss.item() * masked_tokens
                total_masked_tokens += masked_tokens

        metrics = {}
        if total_masked_tokens > 0:
            avg_loss = total_loss / total_masked_tokens
            ppl = math.exp(avg_loss)
            metrics[f"{metric_key_prefix}_perplexity"] = ppl
            metrics[f"{metric_key_prefix}_loss"] = avg_loss
        else:
            metrics[f"{metric_key_prefix}_perplexity"] = float("nan")
            metrics[f"{metric_key_prefix}_loss"] = float("nan")

        self.log(metrics)
        return metrics

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        optimizer.step(closure=optimizer_closure)
