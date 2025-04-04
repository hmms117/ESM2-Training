import yaml
from transformers import EsmConfig

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load the YAML configuration once and cache it.
_config = load_config()

def get_model_config():
    """
    Returns a Hugging Face EsmConfig object built from the model parameters in the YAML config.
    This is used in data processing (to determine max_length and max_tokens_per_batch)
    and in model initialization.
    """
    model_config = _config["model"]
    return EsmConfig.from_dict(model_config)

def get_training_config():
    """
    Returns the training configuration as a dictionary from the YAML config.
    Used in train.py and by the trainer.
    """
    return _config["training"]

def get_data_config():
    """
    Returns the data processing configuration as a dictionary from the YAML config.
    Used in data_processing.py for defaults such as chunk size and temporary directories.
    """
    return _config["data"]