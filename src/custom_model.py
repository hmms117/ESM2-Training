import torch
from transformers import AutoTokenizer, AutoConfig
from faesm.esm import FAEsmForMaskedLM
from config import CUSTOM_CONFIG


def remove_bias_from_attention_linear_layernorm(model):
    """
    Removes biases from all query, key, and value projections in attention blocks,
    intermediate linear layers, and layer norms (Pre-LN).
    """
    for module in model.modules():
        # Remove biases from query, key, and value projections in attention blocks
        if hasattr(module, "query") and isinstance(module.query, torch.nn.Linear):
            module.query.bias = None
        if hasattr(module, "key") and isinstance(module.key, torch.nn.Linear):
            module.key.bias = None
        if hasattr(module, "value") and isinstance(module.value, torch.nn.Linear):
            module.value.bias = None

        # Remove biases from intermediate linear layers
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias = None

        # Remove biases from layer norms
        if isinstance(module, torch.nn.LayerNorm) and module.bias is not None:
            module.bias = None

    return model


def add_loss_to_forward(model):
    """
    Monkey-patches the forward method of the model to compute loss during training.
    """
    original_forward = model.forward

    def forward_with_loss(
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        logits = outputs["logits"]

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs["loss"] = loss

        return outputs

    model.forward = forward_with_loss
    return model

def initialize_model_and_tokenizer():
    """
    Initializes the FAESM model and tokenizer with custom FullPreLN layers.
    Ensures compatibility with FAESM and trains a blank model.
    """
    model_name = CUSTOM_CONFIG["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(model_name)
    config.max_position_embeddings = CUSTOM_CONFIG["model"]["max_position_embeddings"]
    config.use_fa = CUSTOM_CONFIG["model"]["use_fa"]
    config.emb_layer_norm_before = CUSTOM_CONFIG["model"]["emb_layer_norm_before"]
    config.hidden_size = CUSTOM_CONFIG["model"]["hidden_size"]
    config.intermediate_size = CUSTOM_CONFIG["model"]["intermediate_size"]
    config.num_attention_heads = CUSTOM_CONFIG["model"]["num_attention_heads"]
    config.num_hidden_layers = CUSTOM_CONFIG["model"]["num_layers"]

    # Initialize the base model
    model = FAEsmForMaskedLM(config)

    # Remove biases
    model = remove_bias_from_attention_linear_layernorm(model)

    # Add loss calculation
    model = add_loss_to_forward(model)

    return model, tokenizer