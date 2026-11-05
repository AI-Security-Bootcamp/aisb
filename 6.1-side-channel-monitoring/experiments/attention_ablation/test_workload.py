"""CPU checks for residual-branch identity and gradient semantics."""
from types import SimpleNamespace
import unittest

import torch

from workload import bypassed_decoder_forward


class Scale(torch.nn.Module):
    def __init__(self, factor, attention=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(factor)))
        self.attention = attention
        self.calls = 0

    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        output = hidden_states * self.weight
        return (output, None) if self.attention else output


class BypassTests(unittest.TestCase):
    def test_outputs_gradients_and_skipped_modules(self):
        # Active attention multiplies the residual by 1 + 2*3 = 7;
        # active MLP multiplies it by 1 + 5*7 = 36.
        for skip_attention, skip_mlp, factor in (
            (False, False, 252), (True, False, 36),
            (False, True, 7), (True, True, 1),
        ):
            with self.subTest(skip_attention=skip_attention, skip_mlp=skip_mlp):
                layer = SimpleNamespace(
                    input_layernorm=Scale(2), self_attn=Scale(3, attention=True),
                    post_attention_layernorm=Scale(5), mlp=Scale(7),
                    _aisb_skip_attention=skip_attention, _aisb_skip_mlp=skip_mlp,
                )
                x = torch.tensor([0.5, -1.0], requires_grad=True)
                y = bypassed_decoder_forward(layer, x)
                torch.testing.assert_close(y, x * factor, rtol=0, atol=0)
                y.sum().backward()
                torch.testing.assert_close(x.grad, torch.full_like(x, factor), rtol=0, atol=0)
                if skip_attention and skip_mlp:
                    self.assertIs(y, x)
                for module, skipped in (
                    (layer.input_layernorm, skip_attention), (layer.self_attn, skip_attention),
                    (layer.post_attention_layernorm, skip_mlp), (layer.mlp, skip_mlp),
                ):
                    self.assertEqual(module.calls, 0 if skipped else 1)
                    self.assertEqual(module.weight.grad is None, skipped)

    def test_rejects_cache_semantics_outside_this_experiment(self):
        with self.assertRaisesRegex(ValueError, "use_cache=False"):
            bypassed_decoder_forward(None, torch.ones(1), use_cache=True)


if __name__ == "__main__":
    unittest.main()
