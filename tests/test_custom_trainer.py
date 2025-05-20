from src.custom_trainer import CustomTrainer


def test_group_by_batch():
    dataset = [
        {"batch_id": 0, "val": 1},
        {"batch_id": 0, "val": 2},
        {"batch_id": 1, "val": 3},
    ]
    trainer = CustomTrainer.__new__(CustomTrainer)
    groups = trainer.group_by_batch(dataset)
    assert len(groups) == 2
    assert sorted(len(g) for g in groups) == [1, 2]


def test_create_collate_fn_removes_keys():
    def base_collate(batch):
        return batch

    collate = CustomTrainer.create_collate_fn(base_collate, keys_to_remove=["id"]) 
    batch = [[{"id": 1, "x": 2}, {"id": 2, "x": 3}]]
    result = collate(batch)
    assert all("id" not in ex for ex in result)
    assert result[0]["x"] == 2

def test_evaluate_returns_perplexities():
    import torch
    from types import SimpleNamespace
    from datasets import Dataset

    sample = {"batch_id": 0, "input_ids": [1], "attention_mask": [1], "labels": [1]}
    ds = Dataset.from_list([sample])

    trainer = CustomTrainer.__new__(CustomTrainer)
    trainer.train_dataset = ds
    trainer.eval_dataset = ds
    trainer.data_collator = lambda batch: {
        "input_ids": torch.tensor([ex["input_ids"] for ex in batch]),
        "attention_mask": torch.tensor([ex["attention_mask"] for ex in batch]),
        "labels": torch.tensor([ex["labels"] for ex in batch]),
    }
    trainer.args = SimpleNamespace(dataloader_num_workers=0, device=torch.device("cpu"))
    trainer.log = lambda metrics: None
    trainer.model = torch.nn.Module()
    def forward(input_ids, attention_mask=None, labels=None):
        return {"loss": torch.tensor(0.0)}
    trainer.model.forward = forward

    metrics = CustomTrainer.evaluate(trainer)
    assert "eval_perplexity" in metrics
    assert "train_perplexity" in metrics
