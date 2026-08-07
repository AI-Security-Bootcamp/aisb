#%%
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report


# %%
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import matplotlib.pyplot as plt


def load_model_and_image() -> Tuple[ViTImageProcessor, ViTForImageClassification, torch.Tensor]:
    """Load a pre-trained ViT model and a sample image."""
    # Load the model
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    # Load a sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw)
    image = torch.tensor(np.array(raw_image)).permute(2, 0, 1)
    
    # # Convert to float for noise addition (uint8 will overflow/clip weirdly otherwise)
    # image_float = image.float()

    # # Gaussian noise
    # noise = torch.randn_like(image_float) * 10000  # std controls noise intensity, e.g. std=25
    # noisy_image = image_float + noise

    # # Clip back to valid pixel range and convert back to uint8
    # noisy_image = torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    return processor, model, image


def classify_image(
    processor: ViTImageProcessor, model: ViTForImageClassification, image: torch.Tensor
) -> Tuple[int, str]:
    """
    Classify an image using the ViT model.

    Args:
        processor: ViT image processor
        model: ViT classification model
        image: Image tensor in CHW format

    Returns:
        predicted_class_idx: Index of predicted class
        predicted_class_name: Name of predicted class
    """
    
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    logits = model(pixel_values).logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_name = model.config.id2label[predicted_class_idx]
    
    return predicted_class_idx, predicted_class_name


# Test the classification
processor, model, image = load_model_and_image()
class_idx, class_name = classify_image(processor, model, image)
import section1_test
from section1_test import test_classify_image

section1_test.model = model
section1_test.processor = processor
section1_test.image = image

test_classify_image(classify_image)

plt.figure(figsize=(8, 6))
plt.imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
plt.title(f"Predicted class: {class_name}")
plt.axis("off")
plt.show()


# %%


def create_adversarial_perturbation(
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    image: torch.Tensor,
    target_class_id: int,
    steps: int = 10,
    lr: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Create an adversarial perturbation to make the model classify the image as target_class.

    Args:
        processor: ViT image processor
        model: ViT classification model
        image: Original image tensor
        target_class_id: Target class index
        steps: Number of optimization steps
        lr: Learning rate

    Returns:
        perturbation: The adversarial perturbation
        perturbed_image: The adversarially perturbed image
        success: Whether the attack succeeded (the target class was predicted)
    """

    # Process the original image
    inputs = processor(images=image, return_tensors="pt")

    # Initialize perturbation
    perturbation = torch.rand_like(inputs["pixel_values"]) * 0.01
    perturbation.requires_grad = True

    # Optimizer for the perturbation
    optimizer = torch.optim.Adam([perturbation], lr=lr)

    success = False
    for step in range(steps):
        optimizer.zero_grad()

        # Apply perturbation
        perturbed_inputs = inputs["pixel_values"] + perturbation

        # Get model predictions
        outputs = model(pixel_values=perturbed_inputs)
        logits = outputs.logits

        # Compute loss (cross-entropy with target class)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_class_id]))

        # Check if attack succeeded
        predicted_class = logits.argmax(-1).item()
        if predicted_class == target_class_id:
            success = True

        # Backward pass
        loss.backward()
        optimizer.step()

        if step % 2 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, Predicted: {model.config.id2label[predicted_class]}")

    perturbed_image = inputs["pixel_values"] + perturbation
    return perturbation.detach(), perturbed_image.detach(), success


# Test adversarial attack
target_class = "daisy"
target_class_id = model.config.label2id[target_class]

print(f"\nAttempting to change prediction to: {target_class}")
print("=" * 60)

perturbation, perturbed_image, success = create_adversarial_perturbation(
    processor, model, image, target_class_id, steps=10, lr=0.1
)

print(f"\nAttack {'succeeded' if success else 'failed'}!")




# %%
# Visualize the original, perturbation, and perturbed image
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
_, orig_class = classify_image(processor, model, image)
axes[0].set_title(f"Original: {orig_class}")
axes[0].axis("off")

# Perturbation (normalized for visualization)
pert_vis = perturbation.squeeze().permute(1, 2, 0).numpy()
# Normalize to [0, 1] for visualization
pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min())
axes[1].imshow(pert_vis)
axes[1].set_title(f"Perturbation (L2: {perturbation.norm().item():.3f})")
axes[1].axis("off")

# Perturbed image
perturbed_vis = perturbed_image.squeeze().permute(1, 2, 0).numpy()
axes[2].imshow(perturbed_vis)
# Get prediction for perturbed image
outputs = model(pixel_values=perturbed_image)
pred_idx = outputs.logits.argmax(-1).item()
axes[2].set_title(f"Perturbed: {model.config.id2label[pred_idx]}")
axes[2].axis("off")

plt.tight_layout()
plt.show()





# %%


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
    """
    Create an adversarial perturbation, but add l2 and l∞ constraints.

    Args:
        processor: ViT image processor
        model: ViT classification model
        image: Original image tensor
        target_class_id: Target class index
        steps: Number of optimization steps
        lr: Learning rate
        l2_reg: L2 regularization strength
        l_inf_bound: Maximum allowed change per pixel (L∞ constraint)

    Returns:
        perturbation: The adversarial perturbation
        perturbed_image: The adversarially perturbed image
        success: Whether the attack succeeded
    """
    inputs = processor(images=image, return_tensors="pt")

    # Initialize perturbation
    perturbation = torch.zeros_like(inputs["pixel_values"])
    perturbation.requires_grad = True

    optimizer = torch.optim.Adam([perturbation], lr=lr)

    history = {"loss": [], "predictions": []}
    success = False

    for step in range(steps):
        optimizer.zero_grad()

        # Apply perturbation with L∞ constraint
        clamped_perturbation = torch.clamp(perturbation, -l_inf_bound, l_inf_bound)
        perturbed_inputs = inputs["pixel_values"] + clamped_perturbation

        # Ensure pixel values stay in valid range [0, 1]
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)

        # Get model predictions
        outputs = model(pixel_values=perturbed_inputs)
        logits = outputs.logits

        # Compute loss with L2 regularization
        ce_loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_class_id]))
        l2_loss = clamped_perturbation.norm()
        total_loss = ce_loss + l2_reg * l2_loss

        # Track progress
        predicted_class = logits.argmax(-1).item()
        history["loss"].append(total_loss.item())
        history["predictions"].append(predicted_class)

        if predicted_class == target_class_id:
            success = True

        # Backward pass
        total_loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(
                f"Step {step}, CE Loss: {ce_loss.item():.4f}, L2: {l2_loss.item():.4f}, "
                f"Predicted: {model.config.id2label[predicted_class]}"
            )

    # Final clamped perturbation
    final_perturbation = torch.clamp(perturbation, -l_inf_bound, l_inf_bound).detach()
    final_perturbed = torch.clamp(inputs["pixel_values"] + final_perturbation, 0, 1)

    return final_perturbation, final_perturbed, success


# Test different regularization strengths
regularization_strengths = [0.5, 2.0, 5.0]
results = []

for l2_reg in regularization_strengths:
    print(f"\n{'=' * 60}")
    print(f"Testing L2 regularization strength: {l2_reg}")
    print(f"{'=' * 60}")

    pert, perturbed, success = create_constrained_adversarial_attack(
        processor, model, image, target_class_id, steps=30, lr=0.05, l2_reg=l2_reg, l_inf_bound=0.1
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
from section1_test import test_constrained_adversarial_attack

section1_test.target_class_id = target_class_id


test_constrained_adversarial_attack(create_constrained_adversarial_attack)



# Visualize results for different regularization strengths
fig, axes = plt.subplots(len(results), 3, figsize=(12, 4 * len(results)))

for i, result in enumerate(results):
    # Original
    axes[i, 0].imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    # Perturbation
    pert_vis = result["perturbation"].squeeze().permute(1, 2, 0).numpy()
    pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
    axes[i, 1].imshow(pert_vis)
    axes[i, 1].set_title(f"Perturbation (L2 reg={result['l2_reg']})")
    axes[i, 1].axis("off")

    # Perturbed
    perturbed_vis = result["perturbed_image"].squeeze().permute(1, 2, 0).numpy()
    axes[i, 2].imshow(perturbed_vis)

    # Get final prediction
    outputs = model(pixel_values=result["perturbed_image"])
    pred_idx = outputs.logits.argmax(-1).item()
    pred_class = model.config.id2label[pred_idx]

    status = "✓" if result["success"] else "✗"
    axes[i, 2].set_title(
        f"{status} Predicted: {pred_class}\nL2: {result['l2_norm']:.3f}, L∞: {result['l_inf_norm']:.3f}"
    )
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

# Summary statistics
print("\nAttack Summary:")
print("=" * 60)
for result in results:
    print(f"L2 Regularization: {result['l2_reg']}")
    print(f"  - Success: {'Yes' if result['success'] else 'No'}")
    print(f"  - L2 norm: {result['l2_norm']:.4f}")
    print(f"  - L∞ norm: {result['l_inf_norm']:.4f}")
    print()


# %%
