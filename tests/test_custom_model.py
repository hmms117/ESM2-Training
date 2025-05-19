import torch
from src import custom_model


class DummyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 4)
        self.key = torch.nn.Linear(4, 4)
        self.value = torch.nn.Linear(4, 4)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()
        self.fc = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 2)
        return {"logits": logits}


def test_remove_bias_from_attention_linear_layernorm():
    model = DummyModel()
    assert model.attn.query.bias is not None
    assert model.fc.bias is not None
    assert model.norm.bias is not None

    custom_model.remove_bias_from_attention_linear_layernorm(model)

    assert model.attn.query.bias is None
    assert model.attn.key.bias is None
    assert model.attn.value.bias is None
    assert model.fc.bias is None
    assert model.norm.bias is None


def test_add_loss_to_forward():
    model = DummyModel()
    custom_model.add_loss_to_forward(model)

    input_ids = torch.tensor([[1, 1]])
    labels = torch.tensor([[1, -100]])
    outputs = model(input_ids=input_ids, labels=labels)

    assert "loss" in outputs
    loss_val = outputs["loss"].item()
    assert loss_val >= 0
