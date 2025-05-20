import torch
from src import custom_model
import types


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


def test_initialize_model_zero_init(monkeypatch):
    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return "tokenizer"

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(name):
            return types.SimpleNamespace()

    class DummyModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2)
            self.classifier = torch.nn.Linear(2, 2)
            self.proj_out = torch.nn.Linear(2, 2)
            self.output_layer = torch.nn.Linear(2, 2)
            self.embed_out = torch.nn.Linear(2, 2)

    def fake_get_model_config():
        return types.SimpleNamespace(
            model_name="dummy",
            max_position_embeddings=2,
            max_batch_size=2,
            use_fa=False,
            emb_layer_norm_before=False,
            num_layers=1,
            hidden_size=2,
            intermediate_size=2,
            num_attention_heads=1,
        )

    monkeypatch.setattr(custom_model, "AutoTokenizer", DummyAutoTokenizer)
    monkeypatch.setattr(custom_model, "AutoConfig", DummyAutoConfig)
    monkeypatch.setattr(custom_model, "FAEsmForMaskedLM", DummyModel)
    monkeypatch.setattr(custom_model, "get_model_config", fake_get_model_config)
    monkeypatch.setattr(custom_model, "remove_bias_from_attention_linear_layernorm", lambda m: m)
    monkeypatch.setattr(custom_model, "add_loss_to_forward", lambda m: m)

    model, tok = custom_model.initialize_model_and_tokenizer()

    for attr in ["lm_head", "classifier", "proj_out", "output_layer", "embed_out"]:
        layer = getattr(model, attr)
        assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        if layer.bias is not None:
            assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
