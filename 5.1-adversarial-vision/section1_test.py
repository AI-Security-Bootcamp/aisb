# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import matplotlib.pyplot as plt



@report
def test_classify_image(solution):
    """requires: GPU (loads a ViT model and runs a forward pass)."""
    # The sample COCO image (two cats on a couch) is confidently an "Egyptian cat"
    # under the standard ImageNet-pretrained ViT.
    idx, name = solution(processor, model, image)
    assert idx == 285, f"Expected class index 285 (Egyptian cat), got {idx} ({name})"
    assert name == "Egyptian cat", f"Expected 'Egyptian cat', got {name!r}"
    print("  All tests passed!")




@report
def test_constrained_adversarial_attack(solution):
    """requires: GPU (runs a gradient-based attack against the ViT).

    Checks two substantive properties rather than a tautology:
      1. The attack actually flips the prediction to the *target* class.
      2. The returned perturbation respects the stated L-infinity bound.
    """
    l_inf_bound = 0.1
    perturbation, perturbed_image, success = solution(
        processor,
        model,
        image,
        target_class_id,
        steps=40,
        lr=0.05,
        l2_reg=0.5,
        l_inf_bound=l_inf_bound,
    )

    # The perturbed image must be classified as the target class.
    pred_idx = model(pixel_values=perturbed_image).logits.argmax(-1).item()
    assert pred_idx == target_class_id, (
        f"Attack did not flip the prediction to the target class "
        f"{target_class_id} ({target_class}); got {pred_idx} "
        f"({model.config.id2label[pred_idx]})"
    )
    assert success, "Attack reported success=False despite predicting the target class"

    # The perturbation must honour the L-infinity constraint (small numerical slack).
    max_abs = perturbation.abs().max().item()
    assert max_abs <= l_inf_bound + 1e-4, (
        f"Perturbation violates L-inf bound {l_inf_bound}: max |delta| = {max_abs:.4f}"
    )
    print("  All tests passed!")
