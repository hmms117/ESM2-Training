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
