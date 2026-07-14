# 5.4 — Image provenance and watermark robustness

## What to expect

The ideal section teaches a keyed, blindly detectable image watermark and asks
participants to calibrate detection, quality, false positives, and robustness
under realistic transformations.

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | NumPy/PyTorch array axes, basic complex numbers/FFT use, and willingness to inspect a diffusion pipeline | Embed a keyed signal, implement a detector that does not require the original image, and run a transformation test suite |
| ML | High-level diffusion/UNet concepts and image-frequency intuition | Explain where a watermark is injected and quantify image quality versus detectability |
| Security | Authentication/provenance, attacker transformations, thresholds, and false positives | Define the watermark threat model, calibrate a detector, and perform removal/evasion analysis |
| Theory | Statistical detection and hypothesis testing | Distinguish an arbitrary perturbation from a watermark carrying keyed, testable evidence |

### Preparation

- Review a short FFT/image-frequency primer and the assigned diffusion-watermark paper.
- Just-in-time references should cover PyTorch forward hooks and the exact diffusion-pipeline output contract.

### Current-state TODOs

- [ ] Replace the current frequency attenuation with an actual keyed watermark and blind detector, or rename it as a toy frequency-perturbation lab.
- [ ] Fix FFT axes so both spatial dimensions—not a color channel—are transformed, and use `log10` for dB-labelled visualizations.
- [ ] Align the embedder's target region with the detector and remove dependence on a paired baseline image.
- [ ] Calibrate thresholds against clean false positives and report quality and robustness metrics.
- [ ] Avoid making unfamiliar UNet-hook return plumbing the primary challenge unless that is an explicit engineering objective.
