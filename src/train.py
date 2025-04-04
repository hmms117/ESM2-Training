import glob
import os
import argparse
import torch.distributed as dist
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from config import get_training_config
from custom_model import initialize_model_and_tokenizer
from custom_trainer import CustomTrainer

from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ESM2 on preprocessed datasets.")
    parser.add_argument("--train_dir", type=str, help="Path to the processed training dataset.")
    parser.add_argument("--val_dir", type=str, help="Path to the processed validation dataset.")
    args = parser.parse_args()

    # Check if distributed training is initialized
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    # 1) Load training configuration
    training_config = get_training_config()

    # 2) Gather arrow file paths (shards) for the training set
    train_dir = args.train_dir
    shard_paths = sorted(glob.glob(os.path.join(train_dir, "shard-*")))
    if len(shard_paths) == 0:
        raise ValueError(f"No shards found in {train_dir}!")
    if rank == 0:  # Print only on rank 0
        print(f"Found {len(shard_paths)} shards for training in {train_dir}.")

    # 3) Load the validation dataset as normal (assuming a single HF dataset folder)
    val_dataset_dir = args.val_dir
    if rank == 0:  # Print only on rank 0
        print(f"Loading validation dataset from {val_dataset_dir}")
    val_dataset = load_from_disk(val_dataset_dir)

    # 4) Initialize model and tokenizer
    if rank == 0:  # Print only on rank 0
        print("Initializing model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer()

    # Print model size (exclude bias terms set to None) only on rank 0
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params} trainable parameters.")

    # 5) Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=training_config["mlm_probability"],
    )

    # 6) TrainingArguments
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        overwrite_output_dir=True,
        eval_strategy=training_config["evaluation_strategy"], 
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        learning_rate=training_config["learning_rate"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_steps=training_config["total_steps"],
        warmup_steps=training_config["warmup_steps"],
        logging_dir="./logs",
        logging_steps=training_config["logging_steps"],
        fp16=training_config["mixed_precision"] == "fp16",
        report_to=["wandb"],
        lr_scheduler_type=training_config["lr_scheduler"],
        dataloader_num_workers=training_config["dataloader_num_workers"],
        dataloader_pin_memory=training_config["dataloader_pin_memory"],
        dataloader_prefetch_factor=training_config["dataloader_prefetch_factor"],
        seed=training_config["seed"],
    )

    # 7) Initialize CustomTrainer
    trainer = CustomTrainer(
        train_dataset=shard_paths,  # list of arrow paths
        eval_dataset=val_dataset,   # normal HF dataset
        data_collator=data_collator,
        gradient_clipping=training_config["gradient_clipping"],
        model=model,
        args=training_args,
    )
    if rank == 0:
        print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()