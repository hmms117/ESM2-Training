import torch
from src.muon import Muon


def test_muon_optimizer_step_basic():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = Muon([param], lr=0.1, beta1=0.0, beta2=0.0, eps=0.0, weight_decay=0.0)
    param.grad = torch.tensor([2.0])
    opt.step()
    assert torch.allclose(param, torch.tensor([0.9]))
