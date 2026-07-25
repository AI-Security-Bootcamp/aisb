
# W1D5 - Section 1: Adversarial Attacks on Vision Models

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [Adversarial Attacks on Vision Models](#adversarial-attacks-on-vision-models)
- [Setup](#setup)
- [Adversarial Attacks on Vision Models](#adversarial-attacks-on-vision-models-1)
    - [Exercise 1.1: Understanding Model Predictions](#exercise-11-understanding-model-predictions)
    - [Exercise 1.2a: Adding Random Noise (Optional)](#exercise-12a-adding-random-noise-optional)
    - [Exercise 1.2b: Crafting Adversarial Examples](#exercise-12b-crafting-adversarial-examples)
        - [Questions to consider](#questions-to-consider)
    - [Exercise 1.3: Constrained Adversarial Attacks](#exercise-13-constrained-adversarial-attacks)
    - [Analyzing Attack Trade-offs](#analyzing-attack-trade-offs)
- [Further directions](#further-directions)

Today we'll explore adversarial examples: small, imperceptible changes to images that cause vision classifiers to fail.

## Content & Learning Objectives

### Adversarial Attacks on Vision Models
Adversarial examples are inputs designed to fool machine learning models.
For image classifiers, these are images with small, often imperceptible perturbations that cause misclassification.

> **Learning Objectives**
> - Understand how adversarial perturbations work against vision models
> - Learn to craft targeted adversarial examples with constraints
> - Analyze the trade-off between attack success and perturbation visibility


```python


import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report
```

## Setup

Create a file named `section1_answers.py` in the `5.1-adversarial-vision` directory. This will be your answer file for
today's section.

If you see a code snippet here in the instruction file, copy-paste it into your answer file. Keep the `# %%` line to
make it a Python code cell.

**First install this section's dependencies** (Pillow, matplotlib, diffusers, and friends
used across Day 5):

```bash
pip install -r requirements.txt
```

**Start by pasting the code below in your section1_answers.py file.**


## Adversarial Attacks on Vision Models

Adversarial examples are inputs designed to fool machine learning models.
For image classifiers, these are images with small, often imperceptible perturbations that cause misclassification.

The key insight: Neural networks are vulnerable to small, carefully crafted changes that exploit their decision boundaries.

<details>
<summary>Vocabulary: Adversarial Attack Terms</summary><blockquote>

- **Adversarial Perturbation**: The noise added to an image to fool the model
- **Targeted Attack**: Making the model classify to a specific wrong class
- **Untargeted Attack**: Making the model misclassify to any wrong class
- **L2/L∞ norm**: Ways to measure the magnitude of perturbations
- **Gradient-based attacks**: Using the model's gradients to craft perturbations

</blockquote></details>

### Exercise 1.1: Understanding Model Predictions

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪
>
> You should spend up to ~5 minutes on this exercise.

First, let's load a pre-trained Vision Transformer (ViT) and see how it classifies images.


```python

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
    # TODO: Process the image and get model predictions
    # - Use processor to prepare inputs
    #   - The processor takes in the image and returns a tensor with normalized pixel values that the model was trained on
    #   - It also crops/resizes the image to the expected input size
    # - Run the model to get logits
    # - Find and return the predicted class index and name
    pass
```

<details>
<summary>Hint: getting the predicted class name</summary><blockquote>

Look at what the model.config.id2label dictionary contains.
</blockquote></details>

<details>
<summary>Hint: getting the predicted class index</summary><blockquote>

The logits tensor contains the raw scores for each class. This is essentially a vector of how confident the model is
that the prediction it has made is correct. The index of the maximum value in this vector is the predicted class index.
</blockquote></details>


```python

# Test the classification
processor, model, image = load_model_and_image()
class_idx, class_name = classify_image(processor, model, image)
from section1_test import test_classify_image


test_classify_image(classify_image)

plt.figure(figsize=(8, 6))
plt.imshow(image.numpy().transpose(1, 2, 0).astype("uint8"))
plt.title(f"Predicted class: {class_name}")
plt.axis("off")
plt.show()
```

### Exercise 1.2a: Adding Random Noise (Optional)

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵⚪⚪⚪
>
> You should spend up to ~15 minutes on this exercise.

Try adding some random noise to the image and see how it affects the model's prediction.
How much randomness do you need to add to change the prediction? What is the prediction updated to?

If you can find a way to control this, it would be an excellent attack because it is blackbox, unlike the next attack.


### Exercise 1.2b: Crafting Adversarial Examples

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪
>
> You should spend up to ~30 minutes on this exercise.

Now let's create adversarial perturbations. We'll start with a simple untargeted attack that just tries to change the prediction to any other class.

The basic approach:
1. Add learnable noise to the image
2. Compute the loss (we want to minimize the loss for the target class)
3. Train on this

This is very similar to a regular training loop (see the
[PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
tutorial from the prep material) — same `zero_grad` → forward → loss → `backward` → `optimizer.step`
pattern — but we are training the *inputs* instead of the weights. Concretely:

- The model stays frozen; its parameters are *not* what we optimize.
- We create a perturbation tensor, set `requires_grad=True` on it (the same flag you'd
  normally leave on model weights), and pass that tensor to the optimizer.
- Gradients then flow into the perturbation, updating the noise so the image moves toward
  the target class.


```python


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
    # TODO: Implement adversarial perturbation generation
    # - Run the processor once
    # - Initialize a random perturbation with the same shape as `inputs["pixel_values"]`
    #   (i.e. in the model's normalized input space) and set requires_grad=True
    # - Use an optimizer (e.g. Adam) to update the perturbation
    # - At each step, feed `inputs["pixel_values"] + perturbation` through the model and
    #   minimize cross-entropy loss against the target class
    pass


# Test adversarial attack
target_class = "daisy"
target_class_id = model.config.label2id[target_class]

print(f"\nAttempting to change prediction to: {target_class}")
print("=" * 60)

perturbation, perturbed_image, success = create_adversarial_perturbation(
    processor, model, image, target_class_id, steps=10, lr=0.1
)

print(f"\nAttack {'succeeded' if success else 'failed'}!")
```

Use the following to look at the image, perturbation, and perturbation + image.


```python
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
```

#### Questions to consider
- Notice the squares in the perturbation image - why are they there?
- Is there a pattern in the patches? Why?


### Exercise 1.3: Constrained Adversarial Attacks

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵
>
> You should spend up to ~15 minutes on this exercise.

The previous attack might create very noticeable perturbations. Let's add constraints to make the attack more subtle while still effective.

We'll implement:
1. L2 regularization to keep overall perturbation small
2. L∞ constraints to limit maximum change per pixel
3. Comparison of different regularization strengths


```python


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
    # TODO: Implement constrained adversarial attack
    # - Add L2 regularization to the loss
    # - Clamp perturbation to respect L∞ bounds
    # - Ensure final pixel values stay in [0, 1]
    # - Track loss and predictions over time
    pass
```

<details>
<summary>Hint: L2 regularization</summary><blockquote>

L2 regularization is typically added as a term to the loss function, scaled by a regularization strength. It penalizes large perturbations.

You can just add `l2_reg * perturbation.norm()` to the loss
</blockquote></details>

<details>
<summary>Hint: L∞ constraint</summary><blockquote>

L∞ constraint means that each pixel's perturbation should not exceed a certain threshold. You can use `torch.clamp` to limit the perturbation values.
</blockquote></details>


```python

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


test_constrained_adversarial_attack(create_constrained_adversarial_attack)
```

### Analyzing Attack Trade-offs

Let's analyze how different regularization strengths affect attack success and perturbation visibility.


```python

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
```

## Further directions
- This exercise only applies perturbations to processed images. You would probably want a way to apply perturbations to the original image.
- There are many other (nicer) ways to apply perturbations to images - for example, the original FGSM paper implementation - https://arxiv.org/pdf/1412.6572
- How would you defend against these attacks? And how would you get around these defenses?
- What other ways can you can think of to apply perturbations that are minimal, yet robust (to the defenses discussed in the section above)?
  - How can you make perturbations that still work with some transforms like cropping/scaling/shearing the image?
- ask a TA for more directions if you have already implemented the first two above!
