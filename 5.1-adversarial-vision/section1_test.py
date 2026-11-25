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
def test_classify_image(solution, processor, model, image):
    """requires: GPU (loads a ViT model and runs a forward pass)."""
    # The sample COCO image (two cats on a couch) is confidently an "Egyptian cat"
    # under the standard ImageNet-pretrained ViT.
    idx, name = solution(processor, model, image)
    assert idx == 285, f"Expected class index 285 (Egyptian cat), got {idx} ({name})"
    assert name == "Egyptian cat", f"Expected 'Egyptian cat', got {name!r}"
    print("  All tests passed!")




# %%
@report
def test_constrained_attack_bounds(solution):
    """Check bounds and final success on tiny CPU models without downloading weights."""
    from types import SimpleNamespace

    class TinyProcessor:
        def __init__(self, mean, std):
            self.image_mean = mean
            self.image_std = std

        def __call__(self, images, return_tensors):
            mean = images.new_tensor(self.image_mean).reshape(1, -1, 1, 1)
            std = images.new_tensor(self.image_std).reshape(1, -1, 1, 1)
            return {"pixel_values": (images.unsqueeze(0) - mean) / std}

    class TinyClassifier(torch.nn.Module):
        def __init__(self, threshold):
            super().__init__()
            self.threshold = threshold
            self.inputs = []
            self.config = SimpleNamespace(id2label={0: "original", 1: "target"})

        def forward(self, pixel_values):
            self.inputs.append(pixel_values.detach().clone())
            margin = pixel_values.mean(dim=(1, 2, 3)) - self.threshold
            return SimpleNamespace(logits=torch.stack((-margin, margin), dim=-1))

    # Include black/white pixels and negative normalized values in every channel.
    image = torch.tensor([[0.0, 0.25], [0.75, 1.0]], dtype=torch.float64).repeat(
        3, 1, 1
    )
    for mean, std in (
        ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ([0.25, 0.5, 0.75], [0.25, 0.5, 0.25]),
    ):
        processor = TinyProcessor(mean, std)
        clean = processor(image, return_tensors="pt")["pixel_values"]
        for steps, bound, target in (
            (0, 0.1, 0),
            (0, 0.1, 1),
            (1, 0.0, 1),
            (1, 0.1, 1),
        ):
            model = TinyClassifier(clean.mean())
            perturbation, perturbed, success = solution(
                processor,
                model,
                image,
                target,
                steps=steps,
                lr=0.5,
                l2_reg=0.0,
                l_inf_bound=bound,
            )
            case = (
                f"steps={steps}, bound={bound}, target={target}, mean={mean}, std={std}"
            )
            assert perturbed.shape == clean.shape, f"Input shape changed: {case}"
            assert perturbed.dtype == clean.dtype, f"Input dtype changed: {case}"
            assert torch.allclose(
                perturbation, perturbed - clean
            ), f"Returned perturbation differs from the actual input change: {case}"
            # Check every input sent to the classifier, as well as the returned input.
            for observed in [*model.inputs, perturbed]:
                actual_bound = (observed - clean).abs().max().item()
                assert (
                    actual_bound <= bound + 1e-6
                ), f"Actual input change {actual_bound} exceeds bound {bound}: {case}"
                pixels = observed * clean.new_tensor(std).reshape(
                    1, -1, 1, 1
                ) + clean.new_tensor(mean).reshape(1, -1, 1, 1)
                assert (
                    pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6
                ), f"Model input does not represent valid [0, 1] pixels: {case}"
            # One update crosses the decision boundary; zero updates/budget do not.
            expected_prediction = 1 if steps > 0 and bound > 0 else 0
            assert success == (
                expected_prediction == target
            ), f"Success must describe the final returned input: {case}"
    print("  All tests passed!")




@report
def test_constrained_adversarial_attack(solution, processor, model, image, target_class_id, target_class):
    """requires: GPU (runs a gradient-based attack against the ViT).

    Checks three properties:
      1. The attack actually flips the prediction to the *target* class.
      2. The final model input respects the stated L-infinity bound.
      3. The input remains a valid image after undoing normalization.
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

    # Check the actual image difference, including the effect of clipping.
    clean = processor(images=image, return_tensors="pt")["pixel_values"]
    actual_delta = perturbed_image - clean
    max_abs = actual_delta.abs().max().item()
    assert max_abs <= l_inf_bound + 1e-4, (
        f"Actual image change violates L-inf bound {l_inf_bound}: max |delta| = {max_abs:.4f}"
    )
    assert torch.allclose(perturbation, actual_delta, atol=1e-6), (
        "Returned perturbation must equal perturbed_image minus the clean normalized input"
    )
    mean = clean.new_tensor(processor.image_mean).reshape(1, -1, 1, 1)
    std = clean.new_tensor(processor.image_std).reshape(1, -1, 1, 1)
    pixels = perturbed_image * std + mean
    assert pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6, (
        f"Unnormalized image must stay in [0, 1], got [{pixels.min()}, {pixels.max()}]"
    )
    print("  All tests passed!")
