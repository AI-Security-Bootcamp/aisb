# 5.4 — Image provenance and watermark robustness

## What to expect

The ideal section teaches a keyed, blindly detectable image watermark and asks
participants to calibrate detection, quality, false positives, and robustness
under realistic transformations.

**Suggested time:** 60 minutes

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | NumPy/PyTorch array axes, basic complex numbers/FFT use, and willingness to inspect a diffusion pipeline | Embed a keyed signal, implement a detector that does not require the original image, and run a transformation test suite |
| ML | High-level diffusion/UNet concepts and image-frequency intuition | Explain where a watermark is injected and quantify image quality versus detectability |
| Security | Authentication/provenance, attacker transformations, thresholds, and false positives | Define the watermark threat model, calibrate a detector, and perform removal/evasion analysis |
| Theory | Statistical detection and hypothesis testing | Distinguish an arbitrary perturbation from a watermark carrying keyed, testable evidence |

### Preparation

- Review NumPy's [FFT overview](https://numpy.org/doc/stable/reference/routines.fft.html) and [*Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust*](https://arxiv.org/abs/2305.20030).
- Just-in-time references: PyTorch [`register_forward_hook`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) and the Diffusers [`StableDiffusionPipelineOutput`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) structure.

### Current-state TODOs

- [ ] Replace the current frequency attenuation with an actual keyed watermark and blind detector, or rename it as a toy frequency-perturbation lab.
- [ ] Fix FFT axes so both spatial dimensions—not a color channel—are transformed, and use `log10` for dB-labelled visualizations.
- [ ] Align the embedder's target region with the detector and remove dependence on a paired baseline image.
- [ ] Calibrate thresholds against clean false positives and report quality and robustness metrics.
- [ ] Avoid making unfamiliar UNet-hook return plumbing the primary challenge unless that is an explicit engineering objective.
