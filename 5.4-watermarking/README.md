# 5.4 — Image provenance and watermark robustness

## What to expect

Participants add a secret-key-based watermark to generated images, build a
detector that works without the original image, and test its accuracy, image
quality, and resilience to common image transformations.

**Suggested time:** 60 minutes

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | 2D Fourier transforms and PyTorch model inspection | Embed a keyed signal, implement a detector that does not require the original image, and run a transformation test suite |
| ML | Diffusion models and spatial/frequency-domain image representations | Explain where a watermark is injected and quantify image quality versus detectability |
| Security | Keyed provenance and detector evaluation | Define the watermark threat model, calibrate a detector, and perform removal/evasion analysis |
| Theory | Statistical hypothesis testing and threshold selection | Distinguish an arbitrary perturbation from a watermark carrying keyed, testable evidence |

### Background

- Read the “What Are Diffusion Models?” overview in Hugging Face's [introduction to diffusion models](https://huggingface.co/learn/diffusion-course/en/unit1/1), review NumPy's [FFT overview](https://numpy.org/doc/stable/reference/routines.fft.html), and read the abstract and method figures of [*Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust*](https://arxiv.org/abs/2305.20030); the full paper is optional longer reading.
- Just-in-time references: PyTorch [`register_forward_hook`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) and the Diffusers [`StableDiffusionPipelineOutput`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) structure.

### Current-state TODOs

- [ ] Replace the current frequency attenuation with an actual keyed watermark and blind detector, or rename it as a toy frequency-perturbation lab.
- [ ] Fix FFT axes so both spatial dimensions—not a color channel—are transformed, and use `log10` for dB-labelled visualizations.
- [ ] Align the embedder's target region with the detector and remove dependence on a paired baseline image.
- [ ] Calibrate thresholds against clean false positives and report quality and robustness metrics.
- [ ] Avoid making unfamiliar UNet-hook return plumbing the primary challenge unless that is an explicit engineering objective.
