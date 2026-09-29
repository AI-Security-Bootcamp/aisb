# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
import functools
import json
import tempfile
import time
import matplotlib
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from capture_scope import Scope
import matplotlib.pyplot as plt



@report
def test_current_amps(solution):
    meta = {"range_v": 2, "adc_max": 32000, "sensitivity_mV_per_A": 50}
    result = solution(np.array([-4000, 0, 4000], dtype=np.int16), meta)
    assert np.allclose(result, [-5, 0, 5]), f"Expected [-5, 0, 5] A, got {result}"
    meta["range_v"] = 1
    result = solution(np.array([4000], dtype=np.int16), meta)
    assert np.allclose(result, [2.5]), f"Use the saved voltage range; got {result}"
    assert solution(np.array([], dtype=np.int16), meta).size == 0
    print("  Signed values, scaling, and empty input passed.")




@report
def test_train_step(solution):
    # A tiny CPU model checks actual parameter updates without a GPU or download.
    from types import SimpleNamespace

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, input_ids, labels, use_cache):
            assert use_cache is False, "Training should disable the inference KV cache, pass use_cache=False to the model call"
            assert torch.equal(input_ids, labels), "Use tokens as the causal-LM labels"
            return SimpleNamespace(loss=((self.weight * input_ids.float() - 1) ** 2).mean())

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tokens = torch.tensor([[1, 2]])
    # Two steps distinguish correct updates from accidental gradient accumulation.
    for expected_weight, expected_loss in [(0.55, 0.125), (0.575, 0.10625)]:
        loss = solution(model, optimizer, tokens)
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "Return a scalar tensor"
        assert not loss.requires_grad, "Return the loss detached from its autograd graph"
        assert torch.isclose(loss, torch.tensor(expected_loss)), f"Unexpected loss: {loss}"
        assert torch.isclose(
            model.weight, torch.tensor(expected_weight)
        ), f"Expected parameter {expected_weight}, got {model.weight.item()}; check gradient clearing/update"
    print("  Two training updates and detached losses passed.")
