# %%



import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report


# %%

import hashlib
import io
import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from jaxtyping import Bool, Complex, Float
from PIL import Image, ImageFilter
from scipy.stats import norm
from torch import Tensor
from tqdm.auto import tqdm
MODEL_ID = "nota-ai/bk-sdm-v2-tiny"  # a distilled Stable Diffusion; fast enough to iterate on
LATENT_SIZE = 64  # 512x512 images have latents of shape (4, 64, 64)
CHANNEL = 3  # the latent channel that carries the watermark
RADIUS = 10  # the key lives inside this radius of the spectrum (low frequencies only)
STEPS = 50  # DDIM sampling steps for generation
INVERSION_STEPS = 20  # DDIM steps for inversion; fewer than generation, at a small cost in recovered-noise accuracy
SECRET = "correct horse battery staple"
PROMPT = "a black vase holding a bouquet of roses"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# %%


def setup_pipeline() -> StableDiffusionPipeline:
    """Load the Stable Diffusion model with a DDIM scheduler.

    Returns:
        The pipeline on DEVICE in DTYPE, with DDIM as its scheduler and the safety checker disabled.
    """
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    return pipe.to(DEVICE)

def generate_image(pipe: StableDiffusionPipeline, prompt: str, latents: Float[Tensor, "1 4 latent_size latent_size"], steps: int = STEPS) -> Image.Image:
    """Generate one image from a chosen initial latent.

    Args:
        pipe: The Stable Diffusion pipeline.
        prompt: Text prompt.
        latents: Initial noise of shape (1, 4, LATENT_SIZE, LATENT_SIZE) on the pipeline's device.
        steps: Number of DDIM sampling steps.

    Returns:
        The generated 512x512 RGB image.
    """
    return pipe(prompt, latents=latents, num_inference_steps=steps, guidance_scale=7.5).images[0]

pipe = setup_pipeline()
# A first image, from a seeded initial latent. We reuse it below.
demo_latent = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, generator=torch.Generator(device=DEVICE).manual_seed(8), device=DEVICE, dtype=DTYPE)
demo_image = generate_image(pipe, PROMPT, demo_latent)
plt.figure(figsize=(5, 5))
plt.imshow(np.array(demo_image))
plt.axis("off")
plt.show()

# %%


fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, seed in zip(axes.flat, tqdm(range(6), desc="generating")):
    z = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, generator=torch.Generator(device=DEVICE).manual_seed(seed), device=DEVICE, dtype=DTYPE)
    ax.imshow(np.array(generate_image(pipe, PROMPT, z)))
    ax.set_title(f"seed {seed}"); ax.axis("off")
fig.suptitle(f'"{PROMPT}"'); plt.tight_layout(); plt.show()

# %%
def to_spectrum(image: Float[Tensor, "... h w"]) -> Complex[Tensor, "... h w"]:
    """Compute the centred 2D Fourier spectrum of an array.

        Args:
            image: Real array. The FFT is taken over the last two axes, so leading axes are treated as a batch.

        Returns:
            Complex spectrum of the same shape, shifted so zero frequency sits at the centre.
        """
    raw_fft = torch.fft.fft2(image, dim=(-2, -1))
    centered_fft = torch.fft.fftshift(raw_fft, dim=(-2, -1))
    return centered_fft


def from_spectrum(spec: Complex[Tensor, "... h w"]) -> Float[Tensor, "... h w"]:
    """Invert to_spectrum.

        Args:
            spec: Centred complex spectrum, as returned by to_spectrum.

        Returns:
            The real array whose spectrum is `spec`. The imaginary part of the inverse FFT is discarded.
        """
    uncentered_fft = torch.fft.ifftshift(spec, dim=(-2, -1))
    image = torch.fft.ifft2(uncentered_fft).real
    return image

# %%
    
def example_arrays(photo: Image.Image) -> dict[str, Float[Tensor, "latent_size latent_size"]]:
    """Build the gallery of example arrays for the Fourier walkthrough.

    Args:
        photo: An image to include as the "real image" example; converted to greyscale and resized.

    Returns:
        Mapping from panel title to a (LATENT_SIZE, LATENT_SIZE) array: waves, a blob, a square, noise, the photo,
        and the watermark ripple.
    """
    n = torch.arange(LATENT_SIZE).float()
    y, x = torch.meshgrid(n, n, indexing="ij")
    c = LATENT_SIZE // 2
    r2 = (y - c) ** 2 + (x - c) ** 2
    square = torch.zeros(LATENT_SIZE, LATENT_SIZE)
    square[c - 8:c + 8, c - 8:c + 8] = 1.0
    # A preview of the watermark from the key section: one value per ring, then back to pixel space.
    gen = torch.Generator().manual_seed(0)
    ring_values = torch.randn(10, generator=gen) * LATENT_SIZE / math.sqrt(2)
    rings = torch.zeros(LATENT_SIZE, LATENT_SIZE)
    d = torch.sqrt(r2)
    for i in range(1, 11):
        rings[(d >= i - 1) & (d < i)] = ring_values[i - 1]
    gray = torch.from_numpy(np.array(photo.convert("L").resize((LATENT_SIZE, LATENT_SIZE)))).float() / 255
    return {
        "flat grey": torch.full((LATENT_SIZE, LATENT_SIZE), 0.5),
        "slow wave: 2 cycles across": torch.cos(2 * math.pi * 2 * x / LATENT_SIZE),
        "fast wave: 12 cycles across": torch.cos(2 * math.pi * 12 * x / LATENT_SIZE),
        "vertical wave: 6 cycles": torch.cos(2 * math.pi * 6 * y / LATENT_SIZE),
        "diagonal wave": torch.cos(2 * math.pi * 6 * (x + y) / LATENT_SIZE),
        "smooth blob": torch.exp(-r2 / (2 * 8**2)),
        "sharp-edged square": square,
        "gaussian noise": torch.randn(LATENT_SIZE, LATENT_SIZE, generator=gen),
        "a real image (64x64 grey)": gray,
        "the watermark ripple (the key)": from_spectrum(rings),
    }


def show_spectrum_gallery(examples: dict[str, Float[Tensor, "h w"]]) -> None:
    """Plot each example array next to the log-magnitude of its spectrum.

    Args:
        examples: Mapping from title to 2D array, as returned by example_arrays.
    """
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    for k, (name, arr) in enumerate(examples.items()):
        row, col = divmod(k, 2)
        axes[row, 2 * col].imshow(arr, cmap="gray", **({"vmin": 0, "vmax": 1} if arr.std() == 0 else {}))
        axes[row, 2 * col].set_title(name, fontsize=10)
        axes[row, 2 * col + 1].imshow(torch.log1p(to_spectrum(arr).abs()), cmap="gray")
        axes[row, 2 * col + 1].set_title("spectrum, log|F|", fontsize=10)
    for a in axes.flat:
        a.axis("off")
    plt.tight_layout()
    plt.show()


show_spectrum_gallery(example_arrays(demo_image))

# %%

def show_jpeg_spectra(image: Image.Image, qualities: tuple[int, ...] = (75, 25, 5)) -> None:
    """Plot an image at falling JPEG quality alongside the detail that compression discards.

    Args:
        image: The image to compress.
        qualities: JPEG quality settings to show, one column each, after the original.
    """
    def as_gray(img: Image.Image) -> Float[Tensor, "512 512"]:
        """Greyscale float array of the image, values in [0, 1]."""
        return torch.from_numpy(np.array(img.convert("L"))).float() / 255

    def as_jpeg(img: Image.Image, quality: int) -> Image.Image:
        """Round-trip the image through JPEG at the given quality."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    original = as_gray(image)
    columns = [("original", image)] + [(f"JPEG quality {q}", as_jpeg(image, q)) for q in qualities]
    fig, axes = plt.subplots(4, len(columns), figsize=(3.5 * len(columns), 14))
    for col, (title, img) in enumerate(columns):
        gray = as_gray(img)
        diff = gray - original
        axes[0, col].imshow(np.array(img)); axes[0, col].set_title(title)
        axes[1, col].imshow(torch.log1p(to_spectrum(gray).abs()), cmap="gray", vmin=0, vmax=6); axes[1, col].set_title("spectrum, log|F|")
        axes[2, col].imshow(diff.abs() * 8, cmap="gray", vmin=0, vmax=1); axes[2, col].set_title("detail thrown away (x8)")
        axes[3, col].imshow(torch.log1p(to_spectrum(diff).abs()), cmap="gray", vmin=0, vmax=5); axes[3, col].set_title("spectrum of thrown-away detail")
    for a in axes.flat:
        a.axis("off")
    plt.tight_layout()
    plt.show()


show_jpeg_spectra(demo_image)


# %%
# --------------------------------------------------------------
# Exercise 6.4.1: The Ring Mask
def radial_distance() -> Float[Tensor, "latent_size latent_size"]:
    """Distance of every grid position from the centre of the spectrum.

        Returns:
            (LATENT_SIZE, LATENT_SIZE) float tensor whose entry (i, j) is the Euclidean distance from
            (LATENT_SIZE // 2, LATENT_SIZE // 2).
        """
    c = LATENT_SIZE // 2
    y, x = torch.meshgrid(torch.arange(LATENT_SIZE), torch.arange(LATENT_SIZE), indexing="ij")
    return torch.sqrt(((y - c) ** 2 + (x - c) ** 2).float())



def ring_mask(radius: int = RADIUS) -> Bool[Tensor, "latent_size latent_size"]:
    """Boolean disk mask around the centre of the spectrum.

    Args:
        radius: Positions strictly closer than this to the centre are True.

    Returns:
        (LATENT_SIZE, LATENT_SIZE) boolean tensor.
    """
    # TODO: use radial_distance
    distance = radial_distance()
    output = Tensor(LATENT_SIZE, LATENT_SIZE)
    output.map_(distance, lambda a, x: x < radius)
    return output.to(torch.bool)

from section4_test import test_ring_mask

test_ring_mask(ring_mask)

# %%

def embed_key(
    latents: Float[Tensor, "1 channel h w"], key: Float[Tensor, "h w"], mask: Bool[Tensor, "h w"], channel: int = CHANNEL
) -> Float[Tensor, "1 channel h w"]:
    """Plant a key into one channel of a latent.

    Args:
        latents: Initial latent of shape (1, 4, H, W). Not modified.
        key: Real pattern of shape (H, W) to write into the spectrum.
        mask: Boolean (H, W) mask selecting which spectrum coefficients to overwrite.
        channel: Which of the 4 latent channels carries the watermark.

    Returns:
        A copy of `latents`, same shape and dtype, whose `channel` has the masked Fourier coefficients replaced
        by `key`.
    """
    # TODO:
    # 1. copy latents (as float32)
    # 2. select channel `channel` of the first batch element
    # 3. to_spectrum, overwrite the masked entries with `key` (mind the device and dtype), from_spectrum
    # 4. write the modified channel back into the copy
    # 5. cast back to latents.dtype and return
    import copy
    copied = copy.deepcopy(latents) 
    channel_content = copied.squeeze(0)[channel]
    new_content = to_spectrum(channel_content)
    new_content[mask] = key.to(new_content.device).to(new_content.dtype)[mask]
    copied[:,channel] = from_spectrum(new_content)

    return copied.to(latents.dtype)
from section4_test import test_embed_key


test_embed_key(embed_key)

# %%

demo_z = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE, dtype=DTYPE)
demo_zk = embed_key(demo_z, gen_key("apple"), ring_mask())
before, after = demo_z[0, CHANNEL].float().cpu(), demo_zk[0, CHANNEL].float().cpu()
c, half = LATENT_SIZE // 2, 16
spec_before = to_spectrum(before).abs().log1p()[c - half : c + half, c - half : c + half]
spec_after = to_spectrum(after).abs().log1p()[c - half : c + half, c - half : c + half]
lim = (after - before).abs().max().item()
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(before, cmap="gray"); axes[0, 0].set_title("latent channel 3, before")
axes[0, 1].imshow(after, cmap="gray"); axes[0, 1].set_title("after embedding the key")
axes[0, 2].imshow(after - before, cmap="RdBu", vmin=-lim, vmax=lim); axes[0, 2].set_title("difference: the ripple")
axes[1, 0].imshow(spec_before, cmap="gray"); axes[1, 0].set_title("spectrum before (centre 32x32)")
axes[1, 1].imshow(spec_after, cmap="gray"); axes[1, 1].set_title("spectrum after")
axes[1, 2].imshow((spec_after - spec_before).abs(), cmap="hot"); axes[1, 2].set_title("|spectrum difference|: the rings")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout(); plt.show()

# %%


def gen_latents(seed: int) -> Float[Tensor, "1 4 latent_size latent_size"]:
    """Draw a seeded initial latent.

    Args:
        seed: Random seed.

    Returns:
        Standard Gaussian tensor of shape (1, 4, LATENT_SIZE, LATENT_SIZE) on DEVICE in DTYPE.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # TODO: torch.randn(...) with generator=g, device=DEVICE, dtype=DTYPE
    return torch.randn(generator=g, device=DEVICE, dtype=DTYPE)
    pass


def gen_image_pair(
    seed: int, key: Float[Tensor, "latent_size latent_size"], mask: Bool[Tensor, "latent_size latent_size"], prompt: str = PROMPT
) -> tuple[Image.Image, Image.Image]:
    """Generate a clean and a watermarked image from the same seed.

    Args:
        seed: Random seed for the initial latent.
        key: Ring pattern to embed, as from gen_key.
        mask: Boolean mask selecting the spectrum region to overwrite, as from ring_mask.
        prompt: Text prompt used for both images.

    Returns:
        (clean_image, watermarked_image): the image from the raw latent and the image from the same latent with
        the key embedded.
    """
    # TODO: draw the latent for `seed`, embed the key into it with embed_key, and generate one image from
    # the raw latent and one from the keyed latent. Return (clean_image, watermarked_image).
    pass
from section4_test import test_gen_image_pair


test_gen_image_pair(gen_image_pair, pipe)
from section4_test import test_gen_latents


test_gen_latents(gen_latents)

SECRET = "correct horse battery staple"
KEY = gen_key(SECRET)
MASK = ring_mask()
clean_image, marked_image = gen_image_pair(8, KEY, MASK)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.array(clean_image)); axes[0].set_title("No watermark"); axes[0].axis("off")
axes[1].imshow(np.array(marked_image)); axes[1].set_title("Watermarked"); axes[1].axis("off")
plt.tight_layout(); plt.show()