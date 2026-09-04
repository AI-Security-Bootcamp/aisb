# 6.4 — Keyed Watermarks for Diffusion Images (Optional)

This is optional additional content for Day 6. Complete Sections 6.1–6.3 first.

## What to expect

Participants build a Tree-Ring-style watermark from scratch: a secret key becomes a
ring pattern in the Fourier transform of a diffusion model's initial noise, images
are generated normally from that noise, and a detector recovers the noise by running
DDIM backwards and checks for the pattern. They calibrate the detector to a target
false-positive rate, measure detection under perturbations, and show what the key
does and does not protect. The Fourier transform is introduced by example inside the
section; no prior signal-processing background is assumed.

**Suggested time:** 2.5 hours

**Exercises:** [Open the participant instructions](section4_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Index a PyTorch tensor with a boolean mask and assign into the selected entries; run a Hugging Face diffusers text-to-image pipeline | Convert arrays to and from a centred 2D spectrum; drive a diffusers pipeline from a chosen initial latent; read a DDIM update and its reverse, and use the given inversion loop to recover initial noise from an image; encode images with the VAE |
| ML | Describe diffusion sampling as iterated denoising from Gaussian noise, and state that latent diffusion runs in a VAE's latent space rather than on pixels | Explain why DDIM is invertible and PNDM is not; explain what classifier-free guidance changes between generation and inversion; state where inversion error comes from |
| Security | Given a detector's score distribution on negatives, choose a threshold for a target false-positive rate | Design a keyed, reference-free watermark detector; calibrate its threshold (equal-error or fixed false-positive rate) and evaluate it as detection rate per perturbation; recover a reused key by averaging watermarked images and forge it onto new ones, and explain why key reuse defeats the scheme |
| Theory | Compute a quantile of a normal distribution from its mean and standard deviation | Explain why a real, centrally symmetric spectrum inverts to a real array; explain why rings are rotation invariant and low frequencies survive lossy transforms; explain why a wrong key must score like a clean image |

### Background

- Read the abstract and Figure 1 of [*Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust*](https://arxiv.org/abs/2305.20030); the rest of the paper is optional longer reading.
- Skim the first half of [An Interactive Introduction to Fourier Transforms](https://www.jezzamon.com/fourier/) for the idea that a signal is a sum of waves; the section builds the 2D picture itself.
- Review the "What Are Diffusion Models?" overview in Hugging Face's [introduction to diffusion models](https://huggingface.co/learn/diffusion-course/en/unit1/1) if you need a refresher on sampling as iterated denoising.
- References used in the exercises: diffusers [`DDIMScheduler`](https://huggingface.co/docs/diffusers/api/schedulers/ddim) and [`DDIMInverseScheduler`](https://huggingface.co/docs/diffusers/api/schedulers/ddim_inverse), and the [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) `latents` argument.
- Optional: [WAVES](https://wavesbench.github.io/) for how watermark robustness is benchmarked, and [Black-Box Forgery Attacks on Semantic Watermarks](https://arxiv.org/abs/2412.03283) for the attack side.
