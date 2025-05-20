from src.custom_trainer import CustomTrainer
import torch
import types


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


def test_create_optimizer_muon():
    model = torch.nn.Linear(1, 1)
    trainer = CustomTrainer.__new__(CustomTrainer)
    trainer.model = model
    trainer.args = types.SimpleNamespace(learning_rate=0.1)
    trainer.gradient_clipping = None
    trainer.optimizer = None
    trainer.optimizer_config = {"type": "muon", "beta_1": 0.8, "beta_2": 0.7, "epsilon": 0.0, "weight_decay": 0.0}
    trainer.beta_1 = 0.8
    trainer.beta_2 = 0.7
    trainer.epsilon = 0.0
    trainer.weight_decay = 0.0

    opt = trainer.create_optimizer()
    from src.muon import Muon
    assert isinstance(opt, Muon)
