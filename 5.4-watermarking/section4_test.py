# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
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
from scipy.optimize import brentq
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


def radial_distance() -> Float[Tensor, "latent_size latent_size"]:
    """Distance of every grid position from the centre of the spectrum.

        Returns:
            (LATENT_SIZE, LATENT_SIZE) float tensor whose entry (i, j) is the Euclidean distance from
            (LATENT_SIZE // 2, LATENT_SIZE // 2).
        """
    c = LATENT_SIZE // 2
    y, x = torch.meshgrid(torch.arange(LATENT_SIZE), torch.arange(LATENT_SIZE), indexing="ij")
    return torch.sqrt(((y - c) ** 2 + (x - c) ** 2).float())


def gen_ring_pattern(
    ring_values: Float[Tensor, "radius"],
) -> Float[Tensor, "latent_size latent_size"]:
    """Lay one value per ring onto the spectrum grid.

        Ring i, for i = 1..len(ring_values), is the set of positions whose distance from the centre lies in [i - 1, i).

        Args:
            ring_values: One real value per ring, innermost first.

        Returns:
            (LATENT_SIZE, LATENT_SIZE) float tensor, constant on each ring and zero outside the last ring.
        """
    d = radial_distance()
    pattern = torch.zeros(LATENT_SIZE, LATENT_SIZE)
    for i, value in enumerate(ring_values, start=1):
        pattern[(d >= i - 1) & (d < i)] = value
    return pattern




@report
def test_ring_mask(solution: Callable):
    """Check ring_mask's dtype, size, boundary, and symmetry."""
    m = solution(10)
    assert m.dtype == torch.bool, f"Mask should be boolean, got {m.dtype}"
    assert m.sum() == 305, f"Expected 305 positions inside radius 10, got {int(m.sum())}"
    assert m[32, 32] and m[32, 41] and not m[32, 42], "Mask boundary is wrong: radius 9 inside, radius 10 outside"
    assert torch.equal(m, m.T), "Mask should be symmetric in x and y"
    assert (
        m[32 + 9, 32]
        and m[32 - 9, 32]
        and m[32, 32 - 9]
        and not m[32 - 10, 32]
        and not m[32, 32 - 10]
    ), "Mask should be symmetric about the centre"
    print("  All tests passed!")




@report
def test_embed_key(solution: Callable):
    """Check embed_key touches only the masked coefficients of one channel and leaves the input alone."""
    torch.manual_seed(0)
    mask = radial_distance() < 10
    key = gen_ring_pattern(torch.arange(1, 11).float() * 7 - 40)
    # Latent on the real device, mask and key on the CPU, exactly as in actual use.
    z = torch.randn(1, 4, 64, 64, dtype=DTYPE, device=DEVICE)
    z_before = z.clone()
    out = solution(z, key, mask, channel=3)
    assert out.shape == z.shape and out.dtype == z.dtype, "Output should keep the input's shape and dtype"
    assert out.device == z.device, f"Output should stay on {z.device}, got {out.device}"
    assert torch.equal(z, z_before), "Input latents were modified in place"
    assert torch.equal(out[0, :3], z[0, :3]), "Channels other than the watermarked one must be untouched"
    spec_in, spec_out = to_spectrum(z[0, 3].float().cpu()), to_spectrum(out[0, 3].float().cpu())
    assert torch.allclose(spec_out[~mask], spec_in[~mask], atol=2.0), "Coefficients outside the mask should be unchanged"
    assert (spec_out[mask] - key[mask]).abs().max() < 2.0, "Coefficients inside the mask should equal the pattern"
    print("  All tests passed!")




@report
def test_gen_image_pair(solution: Callable, pipe: StableDiffusionPipeline):
    """Check gen_image_pair returns a deterministic clean/watermarked pair that uses seed, key, and prompt."""
    mask = radial_distance() < 10
    key = gen_ring_pattern(torch.arange(1, 11).float() * 7 - 40)
    clean, marked = solution(3, key, mask)
    assert isinstance(clean, Image.Image) and isinstance(marked, Image.Image), "Should return two PIL images"
    assert clean.size == (512, 512) and marked.size == (512, 512), f"Expected 512x512 images, got {clean.size} and {marked.size}"
    clean_arr, marked_arr = np.array(clean).astype(float), np.array(marked).astype(float)
    assert np.abs(clean_arr - marked_arr).mean() > 5, "Clean and watermarked images are the same; is the key being embedded?"
    # The clean image must come from the raw seed-3 latent, untouched by the key.
    g = torch.Generator(device=DEVICE).manual_seed(3)
    z = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, generator=g, device=DEVICE, dtype=DTYPE)
    expected = np.array(pipe(PROMPT, latents=z, num_inference_steps=STEPS, guidance_scale=7.5).images[0]).astype(float)
    assert np.abs(clean_arr - expected).mean() < 1, "Clean image should be generated from the unmodified seed latent"
    # Same seed and key again: identical pair. Different prompt: different images.
    clean2, marked2 = solution(3, key, mask)
    assert np.array_equal(np.array(marked), np.array(marked2)), "Same seed and key should give the same watermarked image"
    other, _ = solution(3, key, mask, prompt="a red bicycle leaning on a brick wall")
    assert np.abs(np.array(other).astype(float) - clean_arr).mean() > 5, "The prompt argument is not being used"
    print("  All tests passed!")




@report
def test_gen_latents(solution: Callable):
    """Check gen_latents is seeded, on the right device and dtype, and standard Gaussian."""
    z, z2, z3 = solution(1), solution(1), solution(2)
    assert z.shape == (1, 4, 64, 64), f"Expected shape (1, 4, 64, 64), got {tuple(z.shape)}"
    assert z.device.type == DEVICE and z.dtype == DTYPE, f"Expected {DEVICE} / {DTYPE}, got {z.device.type} / {z.dtype}"
    assert torch.equal(z, z2), "Same seed should give the same latent; is the generator being passed to torch.randn?"
    assert not torch.equal(z, z3), "Different seeds should give different latents"
    assert abs(z.float().mean()) < 0.05 and 0.95 < z.float().std() < 1.05, "Latent should be standard Gaussian noise"
    print("  All tests passed!")




@report
def test_image_to_latent(solution: Callable, pipe: StableDiffusionPipeline):
    """Check image_to_latent recovers the latent an image was decoded from."""
    g = torch.Generator(device=DEVICE).manual_seed(3)
    z = torch.randn(1, 4, 64, 64, generator=g, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        final = pipe(PROMPT, latents=z, num_inference_steps=10, output_type="latent").images
        decoded = pipe.vae.decode(final / pipe.vae.config.scaling_factor).sample
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    lat = solution(pipe, image)
    assert lat.shape == (1, 4, 64, 64), f"Expected shape (1, 4, 64, 64), got {tuple(lat.shape)}"
    corr = torch.corrcoef(torch.stack([lat.flatten().float(), final.flatten().float()]))[0, 1].item()
    assert corr > 0.95, f"Encoded latent should closely match the latent the image was decoded from (corr = {corr:.3f})"
    assert abs(lat.float().std() / final.float().std() - 1) < 0.2, "Latent scale is off; did you apply the scaling factor?"
    print("  All tests passed!")




@report
def test_invert_latent(solution: Callable, pipe: StableDiffusionPipeline):
    """Check invert_latent recovers noise that correlates with the true initial latent."""
    g = torch.Generator(device=DEVICE).manual_seed(4)
    z_T = torch.randn(1, 4, 64, 64, generator=g, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        final = pipe(PROMPT, latents=z_T, num_inference_steps=20, output_type="latent").images
    z_hat = solution(pipe, final, steps=20)
    assert z_hat.shape == z_T.shape, f"Expected shape {tuple(z_T.shape)}, got {tuple(z_hat.shape)}"
    corr_fn = lambda a, b: torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()
    baseline = corr_fn(final, z_T)
    corr = corr_fn(z_hat, z_T)
    assert corr > 0.8, f"Recovered noise correlates only {corr:.3f} with the true initial noise (need > 0.8)"
    assert corr > baseline + 0.2, f"Inversion barely improved on the un-inverted latent ({baseline:.3f} -> {corr:.3f})"
    assert 0.7 < z_hat.float().std() < 1.3, f"Recovered noise should have roughly unit std, got {z_hat.float().std():.2f}"
    print("  All tests passed!")




@report
def test_key_distance(solution: Callable):
    """Check key_distance is near zero with the right key and large with noise or the wrong key."""
    torch.manual_seed(0)
    mask = radial_distance() < 10
    pattern = gen_ring_pattern(torch.arange(1, 11).float() * 7 - 40)
    other = gen_ring_pattern(45 - torch.arange(1, 11).float() * 8)
    z = torch.randn(1, 4, 64, 64)
    zw = z.clone()
    spec = to_spectrum(z[0, 3])
    spec[mask] = pattern[mask].to(spec.dtype)
    zw[0, 3] = from_spectrum(spec)
    d_marked, d_clean, d_wrong = (
        solution(zw, pattern, mask),
        solution(z, pattern, mask),
        solution(zw, other, mask),
    )
    assert isinstance(d_marked, float), "Distance should be a Python float"
    assert d_marked < 1.0, f"Distance on a latent that carries the key should be ~0, got {d_marked:.2f}"
    assert d_clean > 20, f"Distance on random noise should be large, got {d_clean:.2f}"
    assert d_wrong > 20, f"Distance with the wrong key should be large, got {d_wrong:.2f}"
    print("  All tests passed!")




@report
def test_calibrate_threshold(solution: Callable):
    """Check calibrate_threshold finds the density crossing of two synthetic clusters."""
    rng = np.random.default_rng(0)
    clean, marked = list(70 + 3 * rng.standard_normal(5000)), list(30 + 3 * rng.standard_normal(5000))
    thr = solution(clean, marked)
    assert abs(thr - 50) < 0.5, f"Equal spreads: threshold should be the midpoint ~50, got {thr:.2f}"
    tight = list(30 + 1 * rng.standard_normal(5000))
    thr2 = solution(clean, tight)
    assert 30 < thr2 < 50, f"Unequal spreads: threshold should move toward the tighter cluster, got {thr2:.2f}"
    p_clean, p_tight = norm.pdf(thr2, np.mean(clean), np.std(clean)), norm.pdf(thr2, np.mean(tight), np.std(tight))
    assert abs(p_clean - p_tight) < 0.05 * max(p_clean, p_tight), "The two fitted densities should be equal at the threshold"
    print("  All tests passed!")




@report
def test_detect(
    solution: Callable,
    pipe: StableDiffusionPipeline,
    pattern: Float[Tensor, "latent_size latent_size"],
    mask: Bool[Tensor, "latent_size latent_size"],
    threshold: float,
):
    """Check detect flags watermarked images and not clean ones on real generations."""
    hits = []
    for seed in (101, 102):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        z = torch.randn(1, 4, 64, 64, generator=g, device=DEVICE, dtype=DTYPE)
        zf = z.clone().float()
        spec = torch.fft.fftshift(torch.fft.fft2(zf[0, 3]), dim=(-2, -1))
        spec[mask.to(DEVICE)] = pattern.to(DEVICE, spec.dtype)[mask.to(DEVICE)]
        zf[0, 3] = torch.fft.ifft2(torch.fft.ifftshift(spec, dim=(-2, -1))).real
        clean = pipe(PROMPT, latents=z, num_inference_steps=STEPS).images[0]
        marked = pipe(PROMPT, latents=zf.to(DTYPE), num_inference_steps=STEPS).images[0]
        (hit_c, d_c), (hit_m, d_m) = solution(pipe, clean, pattern, mask, threshold), solution(pipe, marked, pattern, mask, threshold)
        assert isinstance(hit_c, (bool, np.bool_)) and isinstance(d_c, float), "detect should return (bool, float)"
        assert hit_m and d_m < threshold, f"Watermarked image not detected (distance {d_m:.1f} vs threshold {threshold:.1f})"
        hits.append(hit_c)
        assert d_c > d_m + 10, f"Clean image is too close to the key (clean {d_c:.1f}, marked {d_m:.1f})"
    assert not all(hits), "Both clean images were flagged; the threshold or distance is wrong"
    print("  All tests passed!")
