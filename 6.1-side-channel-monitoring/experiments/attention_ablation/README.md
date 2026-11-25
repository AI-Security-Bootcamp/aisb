# Attention implementations and residual-branch ablations

[Open the six-condition comparison](https://waveforms.aisb.dev/attention-comparison.html).
Each row links to the full waveform viewer and **Open in Perfetto**. The
comparison uses 0.1 ms RMS curves with common time/current axes. Perfetto includes
CUDA execution, decoder-layer spans, separate attention/MLP forward spans,
0.01 and 0.1 ms RMS, and the exact minimum/maximum in each 0.1 ms interval. Individual
viewers retain every original 0.4 ns sample.

## Conditions and observations

Captured on 2026-10-10 on machine 3, with an H200 NVL, PicoScope 5464E, and
RCP120XS probe (50 mV/A). Each condition contains one uninterrupted 420 ms
recording of one training cycle. Times below cover the synchronized, profiled
phase, including host overhead; they are not kernel-only timings.

| Condition | Forward (ms) | Backward (ms) | Optimizer (ms) |
| --- | ---: | ---: | ---: |
| Memory-efficient SDPA, fresh baseline | 42.42 | 74.65 | 5.06 |
| PyTorch FlashAttention SDPA | 36.22 | 66.94 | 5.10 |
| Hugging Face eager attention | 50.70 | 77.11 | 5.07 |
| Attention branches bypassed | 17.51 | 22.62 | 3.38 |
| MLP branches bypassed | 35.14 | 60.33 | 3.04 |
| Both branches bypassed | 10.50 | 8.15 | 1.24 |

These are single captures, useful for interpreting waveform structure rather
than estimating a timing distribution. The probe measures the AC component of
current; its RMS is neither total GPU current nor DC power. Shorter phases can
have larger current fluctuations. Do not infer energy savings from RMS alone.

The timing overlay uses the existing independent pulse calibration transferred
to the same instrument/settings, with approximately ±100 µs reference
uncertainty. This run was not independently recalibrated. No waveform has been
shifted to make its peaks fit the phase labels.

## Controls and bypass semantics

- Qwen/Qwen3-0.6B, 28 decoder layers, batch 8, sequence length 512, BF16,
  fused AdamW at learning rate 1e-5; PyTorch 2.9.1+cu128 and Transformers 4.55.4.
- Same local checkpoint, seed 20260922, generated token batches, and one shared
  memory-efficient-attention optimizer warm-up. Each selected implementation
  then receives a forward/backward warm-up **without an optimizer update**.
  All measured conditions therefore start with the same weights and AdamW state.
- Efficient and flash select a single allowed PyTorch SDPA backend, so an
  unsupported request fails instead of silently falling back. The original
  baseline's trace used memory-efficient attention. Eager uses separate
  matrix-multiply and softmax operations.
- Flash uses `is_causal=True` with no explicit mask. This is limited to our
  unpadded, full-attention batch with no KV cache. A preflight changes future
  tokens and checks that prefix logits are unchanged; the observed maximum
  difference was zero. See the mask handling in `workload.py` before adapting
  this experiment to other input shapes or model configurations.
- A bypass skips the **entire residual branch**: its pre-normalization,
  attention or MLP computation, and residual addition. The residual stream
  passes through unchanged (`x → x`). Returning `x` from an attention module
  while retaining the outer addition would incorrectly produce `2x`.
- Embeddings, final normalization, vocabulary head, and loss remain active even
  when both branches are bypassed. AdamW updates only parameters with gradients,
  so bypasses also shorten the optimizer phase. Ablation losses are not expected
  to match the intact model.

All captures passed backend-operation counts, expected 28-or-0 GPU branch spans,
and gradient-presence checks. Bypassed branches received no gradients. The scope
reported no overflow, and CUDA work fit inside each acquisition window. Unit
tests independently check identity/gradient behavior, including skipped norms.

## Reproduce on the instrument host

These scripts adapt the existing instrument-host runner; they do not require
participants to replace their exercise or training loop. Review the paths in
the Slurm scripts before reuse. Capture directories are intentionally not
overwritten.

The recorded run used `/home/tenant/experiments/attention-ablation-20261010/`:

1. Copy this directory's scripts into `src/`, create `logs/`, and use the
   existing `power-inference` environment and local Qwen3-0.6B checkpoint.
2. Copy `workshop/capture/capture.py` and its `probe.json` into `src/`. For this
   host's existing managed lock file, change its lock open mode from `a` to
   `r+` in that isolated copy. The lock must already exist with the configured
   shared ownership/permissions. Keep the original helper directory on
   `PYTHONPATH`, as shown in `capture.sbatch`.
3. On this run, the system NVML library had updated ahead of the loaded NVIDIA
   driver. `nvml-compat/` contained only the matching `libnvidia-ml.so.1` extracted
   from the cached 595.91.07 package; the job's `LD_LIBRARY_PATH` selected it.
   This did not replace system libraries or require a reboot. If the host's
   library and loaded driver agree, remove that compatibility override.
4. Run `preflight.py` for every variant inside a GPU Slurm allocation, and run
   `test_workload.py` in the same Python environment. `AISB_ATTENTION_VARIANT`
   selects the condition. A fresh output directory is required per run.
5. Submit `capture.sbatch`. Its six-element array runs serially (`%1`), requesting
   GPU and scope exclusively, with a five-minute limit per job. Actual recorded
   jobs used about 15 seconds each. Do not acquire from the login shell.
6. Once all capture jobs finish, run `export_results.py <experiment-root>` with
   the existing `pico` environment. This CPU-only step produces viewer bundles,
   validated Perfetto traces, and `comparison.json`. Exporting after acquisition
   keeps these CPU reductions out of the measurement window.

`preflight.sbatch` records the five non-baseline preflights used here; the
efficient baseline was checked separately. `capture.sbatch` includes all six.
The raw files are each 2.10 GB and stay outside Git. The Cloudflare
[preparation/deployment instructions](../../viewer/cloudflare/README.md) describe
publishing bundles with a fresh capture ID and R2 prefix. The public comparison
page's data is in `viewer/static/attention-comparison.json`.

References: PyTorch [SDPA backend selection](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html),
the [SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html),
and Hugging Face's [attention interface](https://huggingface.co/docs/transformers/attention_interface).
