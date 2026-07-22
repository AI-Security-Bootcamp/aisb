# 5.4 — Image provenance and watermark robustness

## What to expect

The ideal section teaches a keyed, blindly detectable image watermark and asks
participants to calibrate detection, quality, false positives, and robustness
under realistic transformations.

**Suggested time:** 60 minutes

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Identify channel/height/width axes, run a 2D FFT and inverse FFT, calculate complex magnitude, and inspect a PyTorch module's inputs/outputs | Embed a keyed signal, implement a detector that does not require the original image, and run a transformation test suite |
| ML | Draw a noise → denoising UNet → decoder diffusion flow and distinguish spatial from frequency-domain image representations | Explain where a watermark is injected and quantify image quality versus detectability |
| Security | Define keyed provenance, false-positive rate, detection threshold, and an attacker-controlled image transformation | Define the watermark threat model, calibrate a detector, and perform removal/evasion analysis |
| Theory | Given clean and watermarked scores, choose a threshold, calculate false positives, and state null/alternative hypotheses | Distinguish an arbitrary perturbation from a watermark carrying keyed, testable evidence |

### Background

- Review NumPy's [FFT overview](https://numpy.org/doc/stable/reference/routines.fft.html) and read the abstract and method figures of [*Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust*](https://arxiv.org/abs/2305.20030); the full paper is optional longer reading.
- Just-in-time references: PyTorch [`register_forward_hook`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) and the Diffusers [`StableDiffusionPipelineOutput`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) structure.

### Current-state TODOs

- [ ] Replace the current frequency attenuation with an actual keyed watermark and blind detector, or rename it as a toy frequency-perturbation lab.
- [ ] Fix FFT axes so both spatial dimensions—not a color channel—are transformed, and use `log10` for dB-labelled visualizations.
- [ ] Align the embedder's target region with the detector and remove dependence on a paired baseline image.
- [ ] Calibrate thresholds against clean false positives and report quality and robustness metrics.
- [ ] Avoid making unfamiliar UNet-hook return plumbing the primary challenge unless that is an explicit engineering objective.
