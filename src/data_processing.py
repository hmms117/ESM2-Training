import os
import logging
import sys
from datasets import Dataset, concatenate_datasets
from collections import defaultdict
from transformers import AutoTokenizer
from config import get_model_config
from config import get_data_config
from tqdm import tqdm
import random
import argparse

logger = logging.getLogger(__name__)


def pad_batch(batch, max_length, pad_token_id):
    """
    Pads all fields in a batch (input_ids, attention_mask, labels) to the same length.
    """
    for example in batch:
        padding_length = max_length - len(example["input_ids"])
        # Pad input_ids
        example["input_ids"] += [pad_token_id] * padding_length
        # Pad attention_mask
        example["attention_mask"] += [0] * padding_length
        # Pad labels with -100 (ignored during loss computation)
        example["labels"] += [-100] * padding_length

        # Add checks to ensure padding consistency
        assert len(example["input_ids"]) == max_length, (
            f"Padding error in input_ids: {len(example['input_ids'])} != {max_length}"
        )
        assert len(example["attention_mask"]) == max_length, (
            f"Padding error in attention_mask: {len(example['attention_mask'])} != {max_length}"
        )
        assert len(example["labels"]) == max_length, (
            f"Padding error in labels: {len(example['labels'])} != {max_length}"
        )
    return batch


def save_chunk_to_disk(chunk, chunk_idx, output_dir="data/tmp_chunks"):
    """
    Saves a chunk of data to disk as a Hugging Face Dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx}.json")
    Dataset.from_list(chunk).save_to_disk(chunk_file)
    logger.info("Saved chunk %s to %s", chunk_idx, chunk_file)

def refine_fasta(input_fasta, output_fasta, max_length=4096):
    """
    Reads a FASTA file, filters sequences longer than max_length, and writes a refined version to a new FASTA file.
    Also returns the refined sequences as a dictionary.

    Args:
        input_fasta (str): Path to the input FASTA file.
        output_fasta (str): Path to save the refined FASTA file.
        max_length (int): Maximum length of sequences to include.

    Returns:
        dict: A dictionary of refined sequences {sequence_id: sequence}.
    """
    logger.info("Reading FASTA file...")
    refined_sequences = {}
    excluded_count = 0
    duplicate_count = 0
    sequences = {}

    # Parse the FASTA file
    with open(input_fasta, "r") as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    if current_id not in sequences:
                        sequences[current_id] = "".join(current_seq)
                    else:
                        duplicate_count += 1
                current_id = line  # Header line
                current_seq = []
            else:
                current_seq.append(line)
        # Add the last sequence
        if current_id and current_id not in sequences:
            sequences[current_id] = "".join(current_seq)
        elif current_id:
            duplicate_count += 1

    # Filter sequences by max length
    for seq_id, seq in sequences.items():
        if len(seq) <= max_length:
            refined_sequences[seq_id] = seq
        else:
            excluded_count += 1

    logger.info("Excluded %s sequences longer than %s characters.", excluded_count, max_length)
    logger.info("Refined FASTA contains %s sequences.", len(refined_sequences))
    logger.info("Skipped %s duplicate IDs.", duplicate_count)

    # Write the refined sequences to a new FASTA file
    with open(output_fasta, "w") as out_f:
        for seq_id, seq in refined_sequences.items():
            out_f.write(f"{seq_id}\n")
            out_f.write(f"{seq}\n")

    logger.info("Refined FASTA written to %s.", output_fasta)
    return refined_sequences

def preprocess_fasta(file_path, tokenizer, max_length, max_tokens_per_batch, chunk_size=1000000, batch_dir="data/tmp_chunks"):
    """
    Preprocess a FASTA file to create pre-batched data with batch_id based on max tokens per batch,
    excluding any sequences longer than 4096 characters. Also writes a _refined.fasta with the
    sequences that passed the length filter.
    """

    def tokenize(sequence):
        tokens = tokenizer(
            sequence,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True
        )
        tokens["length"] = len(tokens["input_ids"])
        tokens["labels"] = tokens["input_ids"].copy()  # Ensure labels are independent
        return tokens

    # Step 1: Refine the FASTA file and get the sequences
    raw_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_no_ext = os.path.splitext(base_name)[0]
    refined_fasta_path = os.path.join(raw_dir, f"{name_no_ext}_refined.fasta")
    refined_sequences = refine_fasta(file_path, refined_fasta_path, max_length=4096)

    # 3) Sort sequences by length before tokenizing
    sorted_sequences = sorted(refined_sequences.items(), key=lambda x: len(x[1]))

    logger.info("Processing sequences in chunks...")
    chunked_data = []
    chunk_count = 0
    batch_id = 0
    current_batch = []
    max_padded_length = 0

    for idx, (seq_id, seq) in enumerate(tqdm(sorted_sequences, desc="Tokenizing Sequences")):
        tokens = tokenize(seq)
        sequence_length = len(tokens["input_ids"])

        # Update padded length and check token limits for the current batch
        max_padded_length = max(max_padded_length, sequence_length)
        padded_tokens = max_padded_length * (len(current_batch) + 1)

        if padded_tokens > max_tokens_per_batch:
            # Pad all sequences in the current batch to max_padded_length
            current_batch = pad_batch(current_batch, max_padded_length, tokenizer.pad_token_id)

            # Assign `batch_id` to each sequence in the batch
            for example in current_batch:
                example["batch_id"] = batch_id

            # Add padded and batch-id assigned batch to chunked_data
            chunked_data.extend(current_batch)
            batch_id += 1
            current_batch = []
            max_padded_length = sequence_length  # Reset for the next batch

        # Add the current sequence to the batch
        current_batch.append({
            "id": seq_id,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["labels"],
            "sequence_length": sequence_length,
        })

        # Save chunk when size reaches chunk_size
        if len(chunked_data) >= chunk_size:
            save_chunk_to_disk(chunked_data, chunk_count, batch_dir)
            chunked_data = []
            chunk_count += 1

    # Handle the last batch
    if current_batch:
        current_batch = pad_batch(current_batch, max_padded_length, tokenizer.pad_token_id)
        for example in current_batch:
            example["batch_id"] = batch_id
        chunked_data.extend(current_batch)

    if chunked_data:
        save_chunk_to_disk(chunked_data, chunk_count, batch_dir)

    logger.info("Processed and saved %s chunks.", chunk_count + 1)


def merge_and_shuffle_batches(batch_dir, output_dir, shard_size=25000, seed=100):
    """
    Merges and shuffles preprocessed batches from disk and saves them in the required format, 
    with each shard as its own dataset.

    Args:
        batch_dir (str): Path to the directory containing chunked batches.
        output_dir (str): Directory to save training shards.
        shard_size (int): Number of batches per shard.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    logger.info("Merging and shuffling batches...")

    # Load all chunked datasets from disk
    chunk_files = [os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.startswith("chunk_")]
    datasets = [Dataset.load_from_disk(chunk_file) for chunk_file in chunk_files]
    merged_dataset = concatenate_datasets(datasets)
    logger.info(
        "Merged %s chunks into one dataset with %s examples.",
        len(datasets),
        len(merged_dataset),
    )

    # Group by batch_id
    logger.info("Grouping dataset by batch_id...")
    batch_map = defaultdict(list)
    for example in tqdm(merged_dataset, desc="Grouping by batch_id"):
        batch_id = example["batch_id"]
        batch_map[batch_id].append(example)

    # Convert batch_map into a list of batches
    all_batches = list(batch_map.values())
    logger.info("Total batches: %s", len(all_batches))

    # Shuffle all batches
    logger.info("Shuffling batches...")
    rng = random.Random(seed)
    rng.shuffle(all_batches)

    # Save shards, each as its own dataset
    logger.info("Saving shards...")
    os.makedirs(output_dir, exist_ok=True)
    shard_count = 0
    for i in tqdm(range(0, len(all_batches), shard_size),desc="Saving shards"):
        shard_batches = all_batches[i:i + shard_size]
        shard_examples = [ex for batch in shard_batches for ex in batch]

        # Save each shard as its own dataset
        shard_dataset = Dataset.from_list(shard_examples)
        shard_dir = os.path.join(output_dir, f"shard-{shard_count:05d}")
        shard_dataset.save_to_disk(shard_dir)
        shard_count += 1

    logger.info("Dataset saved with %s shards.", shard_count)

def main():
    if not logging.getLogger().hasHandlers():
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join("logs", "data_processing.log")),
                logging.StreamHandler(sys.stdout),
            ],
        )

    parser = argparse.ArgumentParser(description="Process FASTA files for ESM-2 training")

    # Required arguments
    parser.add_argument("--input_fasta", type=str, required=True,
                        help="Path to the raw FASTA file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save the processed dataset")

    # Optional arguments with defaults
    data_config = get_data_config()
    parser.add_argument("--tmp_dir", type=str, default=data_config["default_tmp_dir"],
                        help="Temporary directory for chunked FASTA outputs")
    parser.add_argument("--chunk_size", type=int, default=data_config["chunk_size"],
                        help="Number of sequences per chunk")
    parser.add_argument("--shard_size", type=int, default=data_config["default_shard_size"],
                        help="Number of batches per dataset shard")

    args = parser.parse_args()

    # Load tokenizer and configuration
    config = get_model_config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Infer max_length from max_position_embeddings
    max_length = config.max_position_embeddings - 2
    max_tokens_per_batch = config.max_batch_size

    # Preprocess the FASTA file in chunks
    preprocess_fasta(args.input_fasta, tokenizer, max_length, max_tokens_per_batch, 
                     chunk_size=args.chunk_size, batch_dir=args.tmp_dir)

    # Merge and shuffle batches, create dataset
    logger.info("Merging, shuffling, and saving datasets...")
    merge_and_shuffle_batches(args.tmp_dir, args.output_dir, shard_size=args.shard_size)

if __name__ == "__main__":
    main()
