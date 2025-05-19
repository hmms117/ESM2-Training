import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

# Prepare path to src
SRC_DIR = Path(__file__).resolve().parents[1] / 'src'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_model_config(monkeypatch):
    # Fake transformers module with EsmConfig
    class FakeEsmConfig(SimpleNamespace):
        @classmethod
        def from_dict(cls, d):
            return cls(**d)

    fake_transformers = SimpleNamespace(EsmConfig=FakeEsmConfig)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)

    config = load_module('config', SRC_DIR / 'config.py')
    model_cfg = config.get_model_config()

    assert model_cfg.model_name == 'facebook/esm2_t30_150M_UR50D'
    assert model_cfg.num_layers == 30


def test_get_training_and_data_config(monkeypatch):
    # Patch transformers again for EsmConfig import during module load
    class FakeEsmConfig(SimpleNamespace):
        @classmethod
        def from_dict(cls, d):
            return cls(**d)

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(EsmConfig=FakeEsmConfig))
    config = load_module('config', SRC_DIR / 'config.py')

    training_cfg = config.get_training_config()
    data_cfg = config.get_data_config()

    assert 'learning_rate' in training_cfg
    assert 'chunk_size' in data_cfg
