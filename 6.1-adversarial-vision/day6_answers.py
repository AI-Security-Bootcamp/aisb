# %%
import sys
from pathlib import Path

# Make the workspace root importable regardless of this file's depth.
_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report

# %%
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def load_model_and_image() -> Tuple[ViTImageProcessor, ViTForImageClassification, torch.Tensor]:
    """Load the ImageNet ViT and the sample COCO image in CHW pixel format."""
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Inference and attacks should use deterministic evaluation behavior.

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    raw_image = Image.open(response.raw).convert("RGB")
    image = torch.from_numpy(np.array(raw_image)).permute(2, 0, 1)
    return processor, model, image


def classify_image(
    processor: ViTImageProcessor, model: ViTForImageClassification, image: torch.Tensor
) -> Tuple[int, str]:
    """Return the ImageNet class index and label predicted for a CHW image."""
    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_idx = logits.argmax(dim=-1).item()
    predicted_class_name = model.config.id2label[predicted_class_idx]
    return predicted_class_idx, predicted_class_name


# %%
# Exercise 6.1.1 — run this cell to download/cache the model and classify the image.
processor, model, image = load_model_and_image()
class_idx, class_name = classify_image(processor, model, image)

print(f"Image shape: {tuple(image.shape)} (channels, height, width)")
print(f"Prediction: {class_idx} — {class_name}")


@report
def test_classify_image(solution) -> None:
    idx, name = solution(processor, model, image)
    assert idx == 285, f"Expected class index 285 (Egyptian cat), got {idx} ({name})"
    assert name == "Egyptian cat", f"Expected 'Egyptian cat', got {name!r}"
    print("  Classification test passed!")


test_classify_image(classify_image)

plt.figure(figsize=(8, 6))
plt.imshow(image.permute(1, 2, 0).numpy())
plt.title(f"Predicted class: {class_name}")
plt.axis("off")
plt.show()

# %%
# Exercise 6.1.2b — a targeted, unconstrained gradient attack.
def create_adversarial_perturbation(
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    image: torch.Tensor,
    target_class_id: int,
    steps: int = 10,
    lr: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Optimize an additive input perturbation to produce one target class.

    The returned tensors are in ViT's normalized input space, not displayable RGB
    pixel space.  This deliberately has no size constraint; the next exercise
    adds one.
    """
    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt").to(device)
    original_pixels = inputs["pixel_values"]
    perturbation = (0.01 * torch.rand_like(original_pixels)).requires_grad_(True)
    target = torch.tensor([target_class_id], device=device)
    optimizer = torch.optim.Adam([perturbation], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        logits = model(pixel_values=original_pixels + perturbation).logits
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        predicted_class_id = logits.argmax(dim=-1).item()
        print(
            f"Step {step + 1:2d}/{steps}: loss={loss.item():.4f}; "
            f"prediction={model.config.id2label[predicted_class_id]}"
        )

    with torch.no_grad():
        perturbed_image = original_pixels + perturbation
        final_prediction = model(pixel_values=perturbed_image).logits.argmax(dim=-1).item()

    return (
        perturbation.detach(),
        perturbed_image.detach(),
        final_prediction == target_class_id,
    )


# %%
target_class = "daisy"
target_class_id = model.config.label2id[target_class]
print(f"Attempting a targeted attack: cat → {target_class}")

perturbation, perturbed_image, success = create_adversarial_perturbation(
    processor, model, image, target_class_id, steps=10, lr=0.1
)
print(f"Attack {'succeeded' if success else 'did not succeed'} after 10 steps.")
print(f"Perturbation L2 norm: {perturbation.norm().item():.3f}")
print(f"Perturbation L∞ norm: {perturbation.abs().max().item():.3f}")


# %%
# The perturbed tensor is in normalized ViT input space. Invert the processor's
# normalization before displaying it as RGB, then clip only for visualization.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image.permute(1, 2, 0).numpy())
axes[0].set_title(f"Original: {class_name}")
axes[0].axis("off")

perturbation_vis = perturbation.squeeze().permute(1, 2, 0).cpu().numpy()
perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (
    perturbation_vis.max() - perturbation_vis.min() + 1e-8
)
axes[1].imshow(perturbation_vis)
axes[1].set_title(f"Perturbation (L2: {perturbation.norm().item():.3f})")
axes[1].axis("off")

perturbed_vis = perturbed_image.squeeze().permute(1, 2, 0).cpu().numpy()
mean = np.asarray(processor.image_mean).reshape(1, 1, 3)
std = np.asarray(processor.image_std).reshape(1, 1, 3)
perturbed_rgb = np.clip(perturbed_vis * std + mean, 0, 1)
with torch.no_grad():
    predicted_class_id = model(pixel_values=perturbed_image).logits.argmax(dim=-1).item()
axes[2].imshow(perturbed_rgb)
axes[2].set_title(f"Perturbed: {model.config.id2label[predicted_class_id]}")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %%
# Exercise 6.1.3 — projected optimization with L2 and L∞ constraints.
def create_constrained_adversarial_attack(
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    image: torch.Tensor,
    target_class_id: int,
    steps: int = 20,
    lr: float = 0.05,
    l2_reg: float = 2.0,
    l_inf_bound: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Create a targeted perturbation with L2 and per-component L∞ limits.

    Projection after each optimizer update enforces the hard L∞ constraint.
    The L2 term is a soft preference: a larger ``l2_reg`` makes perturbation
    energy more expensive but does not by itself impose an exact L2 bound.
    """
    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt").to(device)
    original_pixels = inputs["pixel_values"]
    perturbation = torch.zeros_like(original_pixels, requires_grad=True)
    target = torch.tensor([target_class_id], device=device)
    optimizer = torch.optim.Adam([perturbation], lr=lr)

    # RGB pixels live in [0, 1], but this model receives normalized values.
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    normalized_min = (0 - mean) / std
    normalized_max = (1 - mean) / std

    for step in range(steps):
        optimizer.zero_grad()
        bounded_perturbation = perturbation.clamp(-l_inf_bound, l_inf_bound)
        perturbed_pixels = (original_pixels + bounded_perturbation).clamp(
            normalized_min, normalized_max
        )
        logits = model(pixel_values=perturbed_pixels).logits
        cross_entropy = torch.nn.functional.cross_entropy(logits, target)
        l2_penalty = bounded_perturbation.norm()
        total_loss = cross_entropy + l2_reg * l2_penalty
        total_loss.backward()
        optimizer.step()

        # Project the parameter itself so subsequent steps always start feasible.
        with torch.no_grad():
            perturbation.clamp_(-l_inf_bound, l_inf_bound)

        if (step + 1) % 5 == 0 or step == 0:
            predicted_id = logits.argmax(dim=-1).item()
            print(
                f"Step {step + 1:2d}/{steps}: CE={cross_entropy.item():.4f}; "
                f"L2={l2_penalty.item():.3f}; "
                f"prediction={model.config.id2label[predicted_id]}"
            )

    with torch.no_grad():
        perturbed_image = (original_pixels + perturbation).clamp(normalized_min, normalized_max)
        returned_perturbation = perturbed_image - original_pixels
        final_prediction = model(pixel_values=perturbed_image).logits.argmax(dim=-1).item()

    return (
        returned_perturbation.detach(),
        perturbed_image.detach(),
        final_prediction == target_class_id,
    )


# %%
# Compare soft L2 penalties while retaining the same hard L∞ bound.
regularization_strengths = [0.5, 2.0, 5.0]
results = []

for l2_reg in regularization_strengths:
    print(f"\n{'=' * 60}\nL2 regularization: {l2_reg}\n{'=' * 60}")
    pert, perturbed, success = create_constrained_adversarial_attack(
        processor,
        model,
        image,
        target_class_id,
        steps=30,
        lr=0.05,
        l2_reg=l2_reg,
        l_inf_bound=0.1,
    )
    results.append(
        {
            "l2_reg": l2_reg,
            "perturbation": pert,
            "perturbed_image": perturbed,
            "success": success,
            "l2_norm": pert.norm().item(),
            "l_inf_norm": pert.abs().max().item(),
        }
    )


# %%
fig, axes = plt.subplots(len(results), 3, figsize=(12, 4 * len(results)))
display_mean = np.asarray(processor.image_mean).reshape(1, 1, 3)
display_std = np.asarray(processor.image_std).reshape(1, 1, 3)
for row, result in enumerate(results):
    axes[row, 0].imshow(image.permute(1, 2, 0).numpy())
    axes[row, 0].set_title("Original")
    axes[row, 0].axis("off")

    perturbation_vis = result["perturbation"].squeeze().permute(1, 2, 0).cpu().numpy()
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (
        perturbation_vis.max() - perturbation_vis.min() + 1e-8
    )
    axes[row, 1].imshow(perturbation_vis)
    axes[row, 1].set_title(f"Perturbation (L2 reg={result['l2_reg']})")
    axes[row, 1].axis("off")

    perturbed_vis = result["perturbed_image"].squeeze().permute(1, 2, 0).cpu().numpy()
    perturbed_rgb = np.clip(perturbed_vis * display_std + display_mean, 0, 1)
    with torch.no_grad():
        predicted_id = model(pixel_values=result["perturbed_image"]).logits.argmax(dim=-1).item()
    status = "✓" if result["success"] else "✗"
    axes[row, 2].imshow(perturbed_rgb)
    axes[row, 2].set_title(
        f"{status} {model.config.id2label[predicted_id]}\n"
        f"L2: {result['l2_norm']:.3f}; L∞: {result['l_inf_norm']:.3f}"
    )
    axes[row, 2].axis("off")

plt.tight_layout()
plt.show()

print("\nAttack summary")
for result in results:
    print(
        f"L2 reg={result['l2_reg']}: success={result['success']}; "
        f"L2={result['l2_norm']:.3f}; L∞={result['l_inf_norm']:.3f}"
    )
