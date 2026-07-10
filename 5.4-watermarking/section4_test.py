# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Optional, Any, Union
from PIL import Image, ImageFilter
import io



@report
def test_watermark_hook(solution_watermarker):
    """requires: GPU (mirrors the UNet output layout the hook runs against).

    We exercise the hook on a synthetic UNet output rather than a real forward
    pass. The hook is documented to modify only the second batch element
    (index 1) in a frequency band, so we check:
      1. The first batch element (index 0) is left untouched.
      2. The second batch element is actually changed (the roundtrip through
         FFT -> band scaling -> inverse FFT is not a no-op).
    """
    torch.manual_seed(0)
    # The hook reads/writes output[0][batch_index], so output[0] is a single
    # tensor of shape (batch, channels, H, W); it edits batch element 1.
    fake_tensor = torch.randn(2, 4, 32, 32)
    original = fake_tensor.clone()
    fake_output = (fake_tensor,)

    watermarker = solution_watermarker()
    result = watermarker.watermark_hook(module=None, input=None, output=fake_output)
    modified = result[0]

    # Batch element 0 must be untouched.
    assert torch.allclose(modified[0], original[0], atol=1e-4), (
        "The hook modified batch element 0; it should only touch element 1"
    )
    # Batch element 1 must actually change (watermark applied).
    diff = (modified[1] - original[1]).abs().max().item()
    assert diff > 1e-6, "The hook did not modify batch element 1; the watermark is a no-op"
    print("  All tests passed!")
