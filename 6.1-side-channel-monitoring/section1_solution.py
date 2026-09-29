# %%
"""
# Day 6 — Section 1: Power Side Channels

GPUs are the backbone of the computation for DNNs, and modern models can draw hundreds of watts when
in use. In this exercise, we'll take a look at what information we can get from looking
at a GPU's power usage, and if we can get any interesting information out of this side channel.

We will capture forward, backward, and optimizer activity
and compare the electrical signal with CUDA execution traces.

<!-- toc -->

## Content & Learning Objectives

### Hardware and recording

Understand how a GPU is connected to power, and how we can capture power data.

> **Learning Objectives**
> - Trace power from the mains to GPU circuits and identify the measured conductor.
> - Explain the probe/scope and plot a recording.

### A training workload

Prepare a training loop with forward pass, backward pass, and optimizer.

> **Learning Objectives**
> - Write a small, repeatable training step whose phases can be measured separately.

### Capture the workloads

Capture the current drawn during one training step.

> **Learning Objectives**
> - Record and understand one continuous waveform containing a complete training step.

### Attribute CUDA execution

Use a torch.profile to peek at GPU activity

> **Learning Objectives**
> - Add phase and layer annotations and distinguish CPU launch time from GPU execution.

### Combine the measurements

Put both recordings on one timeline and look at correlations.

> **Learning Objectives**
> - Inspect current RMS and min/max alongside CUDA layers.
> - Estimate layer count from repeated waveform structure, then check it against execution.

### Change the workload

Test which waveform features distinguish different inputs to the same model.

> **Learning Objectives**
> - Measure batch-size effects on step time and throughput.

## Setup

Create a standalone file named **`section1_execute.py`** in the
`6.1-side-channel-monitoring` directory. Build it up as you complete the
exercises, copying the Python snippets into it and implementing the missing
functions. Keep the `# %%` markers as section boundaries.

Submit this file to **Slurm** whenever you want to run an experiment. Each job
starts a fresh Python process, so the script must include its imports, helper
functions, and any model setup it needs. It cannot rely on variables from a
notebook kernel. Use your SSH session to edit files and submit jobs; all GPU
work and scope access belong inside the submitted job.

Start with the idle recording. As you reach the training and profiler exercises,
replace the earlier capture/plotting block with the new one, keeping the helper
functions and model setup it uses. This keeps each short job focused on one
experiment. You can inspect its saved plots and traces between submissions.

Use the prepared environment from the [README](README.md#files-and-environment).
No model download is needed. Each run creates a fresh output directory and
prints its path to the job log.

**Start by pasting the code below in your section1_execute.py file.**
"""

import sys
from pathlib import Path

# Make the workspace root importable (so `from aisb_utils import report` works),
# regardless of how deeply the file is nested.
_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report

import functools
import json
import tempfile
import time

import matplotlib
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from capture_scope import Scope

# Slurm jobs save plots to files without opening a graphical window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keep each run's recordings together without overwriting earlier captures.
OUTPUT = Path(tempfile.mkdtemp(prefix="side-channel-"))
print("Output directory:", OUTPUT)

# %%
r"""
### Submit a Slurm job

Slurm queues your script until the requested hardware is available. The examples
use an instructor-configured `capture` partition with resources `gpu:1` and
`scope:1`; use the names provided by your instructor if they differ. Request
both resources and an exclusive node so other scheduled work cannot overlap
your measurement.

After completing an experiment's code, run this command **in your SSH terminal,
from the `6.1-side-channel-monitoring` directory**:

```bash
sbatch --partition=capture --nodes=1 --ntasks=1 \
    --gres=gpu:1,scope:1 --exclusive --time=00:01:00 \
    --job-name=side-channel --output=slurm-%j.out \
    --wrap='/home/tenant/envs/power-inference/bin/python -u section1_execute.py'
```

The one-minute limit includes Python startup, model loading, warmup, and scope
download. Time waiting in the queue does not count. Keep one submission pending
at a time, and wait for it to finish before editing the script for the next run:
`sbatch` saves the wrapper command, not a copy of your Python file.

The command prints a job ID. Replace `12345` below with that ID. Use
`squeue --me` to see whether your job is pending or running. Once it starts,
both output and errors appear in `slurm-12345.out`:

```bash
squeue --me
tail -f slurm-12345.out
```

Press Ctrl-C to stop following the log; the job continues. To cancel the job
itself, use `scancel 12345`. After it finishes, check its status:

```bash
sacct -j 12345 --format=JobID,State,Elapsed,ExitCode
```

Look for `COMPLETED` and exit code `0:0`, then open the files in the output
directory printed in the log. For `FAILED`, read the traceback. For `TIMEOUT`,
check that you are running only the intended experiment; arrange a longer limit
with the instructor if that experiment cannot finish in a minute. An interrupted
capture is not a complete waveform. See the [Slurm submission reference](https://slurm.schedmd.com/sbatch.html)
for the options used here.

## Hardware and an Idle Recording

### Where to measure

A GPU receives electricity through several stages of power conversion.
In the diagram below, the **power distribution unit (PDU)** distributes
mains AC to the servers. Inside each server, a **power supply unit (PSU)**
converts it to 12 V DC. **Voltage regulators** near the chips reduce that to
the voltages their circuits need, shown here as below 1 V. “PoL” means point
of load: the regulator sits close to the circuit it supplies.

The second diagram moves the AC-to-DC conversion to shared power shelves in
the rack. These supply a 50 V bus, which each tray converts again before the
final regulators. Both arrangements bring power from the mains to the chips
through several converters. See [here](https://www.ti.com/lit/ta/ssztdb4/ssztdb4.pdf)
for more details on the trends in datacentre power distribution.

![diagram 1](resources/ti-traditional-server-power.png)

![diagram 2](resources/ti-ai-server-power.png)

Why does the measurement position matter? Several components can share
the same power source - many server racks can share the same AC busway,
many racks can share the same busbar, many components on the motherboard
can share the same PDU, and so on. So, the closer you are to the component 
you want to measure, the more accurate your readings will look. However, you
can still get some amount of signal from more aggregate sources - for example,
you can distinguish a rack with an idle GPU from a rack from a rack that is
using GPUs most of the time from just looking at the AC supply to the system. 

Computation changes the current demand as transistors switch and charge or
discharge small capacitances. Nearby **capacitors** can supply a short burst
of current before the upstream supply responds. In
[this decoupling diagram, page 2](https://www.ti.com/content/dam/videos/external-videos/en-us/9/3816841626001/6313253251112.mp4/subassets/notes-decoupling_capacitors.pdf#page=2),
follow the high-frequency current loop between the capacitor and load.
The source replenishes that charge over time. A sharp pulse at the chip can
therefore appear smoother further upstream; wiring resistance, inductance,
and regulator response also shape what reaches the probe.

### From current to voltage

The system we are using today uses a **RCP120XS Rogowski probe**. It is comprised
of a flexible coil closed around a conductor.
Changing current creates a changing magnetic field, inducing a voltage in the
coil. This initially measures *how quickly* the current changes. An electronic
**integrator** converts the coil signal into an output proportional to current
within the probe's frequency range. See [this explainer](https://www.pemuk.com/support/the-rogowski-coil)
for an overview of how the probe works.

A steady current produces no changing field. The probe therefore measures
the **AC component** — variations or ripple — even when attached to a DC
supply cable. A nearly flat, zero reading can coexist with a GPU drawing
substantial power. The other important bit to remeember is that the loop
measures the net enclosed current: putting
both a supply wire and its return inside it makes their contributions cancel.

### From voltage to a waveform

A digital oscilloscope displays voltage against time. In the example below,
the horizontal axis is time and the vertical axis is voltage: flat sections
show a constant voltage, and the edges show transitions between levels.
The Z-axis label refers to display brightness; our saved waveform will contain
voltage samples and their timing.

![Oscilloscope waveform](resources/tek-waveform-axes.png)

The next diagram shows how the scope acquires those samples. Follow the signal
from the input amplifier (**Amp**), which scales the voltage, through the
**analog-to-digital converter (A/D or ADC)**, which turns it into integer counts.
The sampling clock sets when these conversions happen. Acquisition memory
holds the samples so they can be processed and displayed after the event.

![Digital storage oscilloscope pipeline](resources/tek-dso-acquisition.png)

If you are interested in exploring further, you can read Tektronix's
[Oscilloscope Basics](https://www.tek.com/en/documents/primer/oscilloscope-basics)
and [Oscilloscope Types](https://www.tek.com/en/documents/primer/oscilloscope-types)
primers.

Our PicoScope records a finite block into its own memory, then downloads it
(over USB)[https://www.picotech.com/helpfiles/psospa-api/blockmodeoverview.html].
We have supplied `capture_scope.py` to interface with the scope, so that you have
a clean API to be able to access the readings. The outputs will be in `scope.npy` and
`scope.json`.

Once we have the ADC counts (which correspond to the voltage at the oscilloscope’s input),
we can convert them back to volts using the saved voltage range and ADC scale.
Dividing by the probe’s sensitivity of 0.05V/A gives the AC component of the current, in amperes.
To plot current, we reverse two conversions: ADC counts to probe voltage,
then probe voltage to current. The probe's sensitivity is **50 mV/A = 0.05 V/A**,
so a 0.25 V output represents 5 A within its usable frequency range:

```text
probe_voltage_V = counts × range_v / adc_max
current_A       = probe_voltage_V / (sensitivity_mV_per_A / 1000)
time_s         = sample_index × interval_s
```

For example, at ±10 V with `adc_max = 32767`, about 819 counts means 0.25 V,
then about 5 A. Your conversion will use the saved settings so it also works
when the range changes.

Several settings determine what detail survives this measurement:

| Setting or limit | Our setup | Effect on the recording |
| --- | --- | --- |
| Voltage range and resolution | ±10 V; 16-bit ADC | Outside the range, peaks clip. A wider range spreads the same ADC levels over more volts. Bit count alone does not specify noise or accuracy. |
| Sample interval | 0.4 ns, or 2.5 billion samples/s | More frequent digitization does not undo filtering before the ADC. |
| Analog bandwidth | Probe: 34 Hz–30 MHz | The sensor attenuates sufficiently slow and fast variations; bandwidth is different from sample rate. Check the RCP120XS row in the [Micsig specifications](https://www.micsig.com/RCPxilie/25.html), which also gives its 120 A peak rating. |
| Input impedance and coupling | 1 MΩ; DC coupling | The 1 MΩ input loads the **probe output**, not the GPU feed. DC coupling preserves the incoming voltage's DC component; it cannot recover what the probe rejected. |
| Record length | Duration / sample interval | A 2 ms recording contains 5 million samples (10 MB as int16); 420 ms uses 2.1 GB. |

<details><summary>Is there a shunt resistor in this measurement?</summary>

A shunt measures current using the voltage drop across a resistor inserted
in the power path: `V = I R`. Our Rogowski probe uses magnetic induction;
50 mV/A describes its conversion gain. It does not imply a physical 50 mΩ
resistor in the GPU feed. Similarly, the scope's 1 MΩ input loads the probe's
output, not the GPU supply.

</details>

### Exercise 6.1.1: Capture and Explain the Signal

> **Difficulty**: 2/5
> **Importance**: 5/5

Implement `current_amps` using the acquisition metadata.

Then, run the capture code below to plot the first 10 µs of an idle recording in amperes.
"""


def current_amps(counts, meta):
    """Convert signed ADC counts using the acquisition's voltage range and probe gain."""
    if "SOLUTION":
        volts = np.asarray(counts, dtype=np.float64) * meta["range_v"] / meta["adc_max"]
        return volts / (meta["sensitivity_mV_per_A"] / 1000)
    else:
        # TODO: Cast counts to float64 array and convert it to volts, then volts to amperes using meta.
        # Hint: look at meta["range_v"] and meta["adc_max"]
        return np.zeros_like(counts, dtype=float)


@report
def test_current_amps(solution):
    meta = {"range_v": 2, "adc_max": 32000, "sensitivity_mV_per_A": 50}
    result = solution(np.array([-4000, 0, 4000], dtype=np.int16), meta)
    assert np.allclose(result, [-5, 0, 5]), f"Expected [-5, 0, 5] A, got {result}"
    meta["range_v"] = 1
    result = solution(np.array([4000], dtype=np.int16), meta)
    assert np.allclose(result, [2.5]), f"Use the saved voltage range; got {result}"
    assert solution(np.array([], dtype=np.int16), meta).size == 0
    print("  Signed values, scaling, and empty input passed.")


test_current_amps(current_amps)


def read_capture(run):
    """Open the raw array without loading a multi-GB capture into RAM."""
    meta = json.loads((run / "scope.json").read_text())
    assert meta["status"] == "completed", "Capture did not complete"
    assert meta["overflow_mask"] == 0, "Clipped input: do not interpret this capture"
    raw = np.load(run / meta["raw_file"], mmap_mode="r")
    assert len(raw) == meta["samples"], "Missing samples"
    return raw, meta


"""
Here's how you can use the Scope. Entering `with scope:` starts a fixed-duration
recording; leaving it waits for completion and downloads the samples. Here the
body does no work, so you will see the machine's background activity during
the requested 2 ms. Add this block to `section1_execute.py` and submit it to
Slurm.
"""

# %%
idle = OUTPUT / "idle-01"
scope = Scope(idle, duration_s=0.002, interval_ns=0.4)
with scope:
    # You can do some GPU operations here
    pass
raw, meta = read_capture(idle)
n = min(len(raw), round(10e-6 / meta["interval_s"]))
plt.plot(np.arange(n) * meta["interval_s"] * 1e6, current_amps(raw[:n], meta))
plt.xlabel("Time (µs)")
plt.ylabel("AC current (A)")
plt.tight_layout()
plt.savefig(idle / "idle.png")
plt.close()
print("Idle plot:", idle / "idle.png")

# %%
"""
Open the saved `idle.png` after the job completes.

As an extension, try updating the code to see what the graphs for initializing tensors on the GPU,
multiplying matrices, adding matrices, etc look. 

<details><summary>Summary</summary>

Supply and return currents approximately cancel inside a single probe loop.
The ADC converts the probe's voltage to counts; your code reverses that scale
and divides by sensitivity. Steady current is absent from this AC measurement,
so near-zero output does not imply an unpowered GPU. Faster sampling cannot
reconstruct an event lost earlier in the measurement path. Negative values can
represent ripple below the local mean, not negative total power. Moving nearer
the load can expose faster events; moving upstream can mix more loads and more
filtering.

</details>

## Measuring Training

We now have a way to measure current. To learn what the changes mean, we need
a workload whose operations we can identify. A training step gives us several:
the forward pass evaluates the model, the backward pass computes gradients,
and the optimizer updates parameters. We will later look for their boundaries
in the recording.

The supplied loader prepares a small Qwen model and an optimizer. We'll use
random input ids as their meaning does not matter for this experiment; 
their shape determines how much work the model does. Use PyTorch's
[train_loop example](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop)
if you need the training pattern.

<details><summary>Interface reference: tokens and causal-language-model loss</summary>

`tokens` is a `torch.long` tensor of shape `[batch_size, sequence_length]` on the
same device as the model. The supplied model accepts
`model(input_ids=tokens, labels=tokens, use_cache=False)` and returns `.loss`.
It shifts the prediction targets internally. Calling `backward()` accumulates
parameter gradients; the optimizer changes parameters. BF16 model parameters
and integer token IDs serve different purposes. No tokenizer is needed for
random token IDs. Fore more details, see the [Qwen3 forward interface](https://huggingface.co/docs/transformers/model_doc/qwen3#transformers.Qwen3ForCausalLM.forward).

</details>

### Exercise 6.1.2: One Training Step

> **Difficulty**: 2/5
> **Importance**: 4/5

Implement `train_step`:
It should perform one parameter update and return the detached scalar loss.
The test runs two successive updates so it can catch gradients left over from
the previous call. Once it passes, use the loop below to run three steps on the GPU.
"""


def load_model():
    """Supplied host-specific loader; use a cached model and a fixed attention backend."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(4158)
    torch.set_num_threads(2)
    model = AutoModelForCausalLM.from_pretrained(
        "/home/tenant/experiments/pico-access/model/Qwen3-0.6B",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    return model, optimizer


def train_step(model, optimizer, tokens):
    """Update model parameters once, without accumulating an earlier step's gradients."""
    if "SOLUTION":
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=tokens, labels=tokens, use_cache=False)
        output.loss.backward()
        optimizer.step()
        return output.loss.detach()
    else:
        # TODO: zero the gradients, compute loss, backpropagate, and update parameters.
        pass


@report
def test_train_step(solution):
    # A tiny CPU model checks actual parameter updates without a GPU or download.
    from types import SimpleNamespace

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, input_ids, labels, use_cache):
            assert use_cache is False, "Training should disable the inference KV cache, pass use_cache=False to the model call"
            assert torch.equal(input_ids, labels), "Use tokens as the causal-LM labels"
            return SimpleNamespace(loss=((self.weight * input_ids.float() - 1) ** 2).mean())

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tokens = torch.tensor([[1, 2]])
    # Two steps distinguish correct updates from accidental gradient accumulation.
    for expected_weight, expected_loss in [(0.55, 0.125), (0.575, 0.10625)]:
        loss = solution(model, optimizer, tokens)
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "Return a scalar tensor"
        assert not loss.requires_grad, "Return the loss detached from its autograd graph"
        assert torch.isclose(loss, torch.tensor(expected_loss)), f"Unexpected loss: {loss}"
        assert torch.isclose(
            model.weight, torch.tensor(expected_weight)
        ), f"Expected parameter {expected_weight}, got {model.weight.item()}; check gradient clearing/update"
    print("  Two training updates and detached losses passed.")


test_train_step(train_step)

"""
Use a batch of eight sequences, each containing 512 token IDs:
"""

# %%
model, optimizer = load_model()
for _ in range(3):
    tokens = torch.randint(model.config.vocab_size, (8, 512), device="cuda")
    loss = train_step(model, optimizer, tokens)
    print(loss.item())
torch.cuda.synchronize()

# %%
"""
These iterations warm up CUDA and allocate optimizer state. We will record
the next step, after those one-time costs. Each iteration draws fresh random
tokens, so the loss does not need to decrease for the step to be working.

torch.cuda.synchronize makes our script to wait for all computations on the GPU to complete. 

## Capture the Workload

The idle capture showed background activity. Now we want one continuous
recording that includes idle time, a training step, and a return to idle.
This lets us compare activity within one waveform without having to align
separate captures.

There is a timing detail to handle first: PyTorch normally queues CUDA work
and returns control to Python before the GPU has finished. Reaching the end
of `train_step` therefore does not establish that it is idle.
[`torch.cuda.synchronize()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)
waits for queued work to finish. Synchronizing before acquisition clears earlier
work from the baseline; synchronizing after the step keeps its completion
inside the scope context.

### Exercise 6.1.3: One Continuous Training Waveform

> **Difficulty**: 3/5
> **Importance**: 5/5

Implement `capture_step` using `with scope:`. Include a 20 ms idle lead-in,
CPU batch preparation, transfer to the GPU, and one complete training step.
Use `Scope(run, duration_s=0.42, interval_ns=0.4)` and place synchronization
where the capture needs it.

You can plot/analyze the graphs on the CPU (ie, without having to queue a slurm job)

After you have the graphs, see what information on the training process you
can get from just looking at the power draw of the device.
"""


def capture_step(run, model, optimizer, batch_size=8, sequence_length=512):
    """Record a complete warmed-up step and its CPU batch preparation in one block."""
    if "SOLUTION":
        torch.cuda.synchronize()
        scope = Scope(run, duration_s=0.42, interval_ns=0.4)
        with scope:
            time.sleep(0.02)
            tokens = torch.randint(
                model.config.vocab_size, (batch_size, sequence_length)
            ).pin_memory()
            tokens = tokens.to("cuda", non_blocking=True)
            loss = train_step(model, optimizer, tokens)
            torch.cuda.synchronize()
        return loss
    else:
        # TODO: Put acquisition boundaries around batch preparation and completed GPU work.
        return torch.tensor(0.0)


"""
### Seeing the whole step

At 0.4 ns per sample, this recording contains just over a billion samples.
Plotting all of them across a screen would obscure the structure we want to
inspect. The helper below summarizes local signal amplitude using
**root mean square (RMS)**: `sqrt(mean(current**2))` over a short window.

For example, readings of +4 A and −4 A have a signed mean of zero but an RMS
of 4 A. Squaring before averaging keeps alternating ripple from cancelling.
Here we use a 0.1 ms window, evaluated every 1 µs. All raw samples contribute,
and the original file remains available for a closer look. RMS describes the
amplitude of this probe's AC signal; it does not give total DC current or watts.
"""


def plot_overview(run):
    """Supplied bounded-memory reduction: every sample contributes to the RMS curve."""
    from merge import centered_rms, reduce_samples

    raw, meta = read_capture(run)
    stride = round(1e-6 / meta["interval_s"])
    gain = float(current_amps(np.array([1]), meta)[0])
    energy, _, _ = reduce_samples(raw, stride, gain)
    rms = centered_rms(energy)  # 100 x 1 µs energy bins: a 0.1 ms RMS window.
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rms)) / 1000, rms)
    ax.set(xlabel="Time since sample zero (ms)", ylabel="AC current RMS, 0.1 ms (A)")
    fig.tight_layout()
    fig.savefig(run / "overview.png")
    return fig, ax


# %%
run = OUTPUT / "training-01"
loss = capture_step(run, model, optimizer)
print("Loss:", loss.item())
fig, ax = plot_overview(run)
plt.close(fig)
print("Training plot:", run / "overview.png")

# %%
"""
Open `overview.png` from the completed job. Look for the transition from the
idle lead-in to sustained activity. Can you
see repeated groups within it? Do those groups have distinct boundaries, or
could several different divisions explain the same waveform? Save your
annotated plot before using execution labels to resolve the ambiguity.

The [existing web viewer](http://amodo-gigabyte-3.pony-regulus.ts.net:6008/) is a
useful comparison with prepared recordings. Use Matplotlib for your new raw
files; that viewer expects a separate prepared bundle, not a `scope.npy` upload.

## Correlating CUDA Execution

We have candidate regions in the waveform, but their shape alone does not
tell us which operation produced them. PyTorch's profiler records the
operations launched by the CPU and the kernels executed on the GPU. We can also
add our own labels to forward, backward, and optimizer work.

[record_function](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.profiler.record_function.html)
annotates a region of **CPU code**. As we saw when placing
the capture boundaries, that region can end before its CUDA work does.
The profiler also records CPU-to-GPU correlations, which let us follow an
operation to its kernels. The supplied `annotate_layers` wrapper adds a name
around each decoder layer's forward method; we will use those
correlations in the next section to construct the corresponding GPU spans.

For this diagnostic recording, we will also synchronize at the end of each
major GPU phase. That makes the phases easier to separate, but adds waiting
that can change the execution timing. Experiment with what the traces liik like
with and without the synchronization points.

### Exercise 6.1.4: Explain the Waveform with a Profiler

> **Difficulty**: 3/5
> **Importance**: 5/5

Implement `profile_step` by adapting your capture function. Use the
[profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
to record CPU and CUDA activities and input shapes. Finally, export the `trace.json` file.

Expand `train_step` into named regions using `record_function`. The next section
expects these names:

`Step data_loading`, `Step host_to_device`, `Step zero_grad`, `Step forward`,
`Step backward`, `Step optimizer`.

Place a synchronization at the end of transfer, forward, backward, and optimizer,
inside their respective annotations. Add the supplied layer wrapper once to
your model before making the capture.

Replace the previous top-level `capture_step`/plotting block with the
`profile_step` call below and submit another job. Keep the imports, function
definitions, and model setup/warmup so the script runs from a fresh process.
"""


def annotate_layers(model):
    """Supplied Qwen decoder annotations; call once per freshly loaded model."""

    def wrap(forward, index):
        @functools.wraps(forward)
        def annotated(*args, **kwargs):
            with record_function(f"Layer {index:02d} forward"):
                return forward(*args, **kwargs)

        return annotated

    for index, layer in enumerate(model.model.layers):
        layer.forward = wrap(layer.forward, index)


def profile_step(run, model, optimizer, batch_size=8, sequence_length=512):
    """Save synchronized phase annotations and raw current for the same training step."""
    if "SOLUTION":
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            torch.cuda.synchronize()
            scope = Scope(run, duration_s=0.42, interval_ns=0.4)
            with scope:
                time.sleep(0.02)
                with record_function("Step data_loading"):
                    tokens = torch.randint(
                        model.config.vocab_size, (batch_size, sequence_length)
                    ).pin_memory()
                with record_function("Step host_to_device"):
                    tokens = tokens.to("cuda", non_blocking=True)
                    torch.cuda.synchronize()
                with record_function("Step zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                with record_function("Step forward"):
                    output = model(input_ids=tokens, labels=tokens, use_cache=False)
                    torch.cuda.synchronize()
                with record_function("Step backward"):
                    output.loss.backward()
                    torch.cuda.synchronize()
                with record_function("Step optimizer"):
                    optimizer.step()
                    torch.cuda.synchronize()
        prof.export_chrome_trace(str(run / "torch.json"))
        return output.loss.detach()
    else:
        # TODO: Capture and profile one complete training step using the sequence below.
        # The model and optimizer are already loaded and warmed up by the calling code.
        # Copy the training phases out here instead of calling train_step, so that
        # each phase can have its own record_function annotation. This is a fairly API-heavy
        # function, so feel free to peek at the olution if you've spent over 5 minutes on this.
        #
        # 1. Initialize the scope for this run.
        # 2. Start the PyTorch profiler.
        # 3. Start the waveform recording inside the profiler context.
        # 4. Prepare a batch and put its token IDs on the GPU.
        # 5. Clear gradients from the previous training step.
        # 6. Run the forward pass and compute the loss.
        # 7. Run the backward pass.
        # 8. Update the model parameters.
        # 9. Finish both recordings and export the execution trace.
        # 10. Return output.loss.detach() as a scalar tensor.
        return torch.tensor(0.0)


# %%
annotate_layers(model)  # do this just once in your script
run = OUTPUT / "profiled-01"
loss = profile_step(run, model, optimizer)
print("Loss:", loss.item(), "Trace:", run / "torch.json")

# %%
"""
Copy the json from the SSH host to your laptop and open it with “Open trace
file” at [ui.perfetto.dev](https://ui.perfetto.dev). This shows an execution
timeline showing you when operations ran.
Find a forward-layer annotation and follow it to the GPU kernels it launched.
Then locate backward and optimizer work. Record their durations and explain
one visible gap using the CPU and GPU tracks.

## Combine the Measurements

We now have two accounts of the same step: current samples from the scope and
execution events from the profiler. The supplied `merge.py` puts them on a
common timeline so we can compare them directly in Perfetto.

The RMS curve makes sustained activity visible, but averages over short
excursions. To retain those, we also show a **min/max envelope**: the lowest
and highest current in each short interval.

| View | What each point means |
| --- | --- |
| RMS | Centered 0.1 ms RMS window, emitted every 1 µs. |
| Minimum/maximum envelope | Smallest/largest raw sample in each nonoverlapping 0.1 ms bin, timestamped at its center. |
| Raw `scope.npy` | Every ADC sample at 0.4 ns. Inspect a short slice in Matplotlib. This track is too big for Perfetto to render. |

The recordings also have different clocks. The merger uses a host timestamp
and a previously measured offset, with 100 µs timing uncertainty for these
instrument settings. This limits how precisely we can associate a current
feature with a kernel. Shifting the waveform until its peaks match the labels
would assume the very correspondence we are trying to test.

### Exercise 6.1.5: Count Layers from Current Activity

> **Difficulty**: 3/5
> **Importance**: 5/5

Set `RUN` in `merge.py` to the profiled capture directory from your job log and
`INCLUDE_ENVELOPE = True`. Run the merger and open `perfetto.json.gz` in
Perfetto. It checks for clipping, incompatible acquisition settings, and
CUDA work outside the capture; investigate any failed check before continuing.

<details><summary>How many layers in your model?</summary>

If everything was corect, you should see a graph that looks something like this:
![trace](resources/captured_trace.png)

and your perfetto view should look like this:
![perfetto](resources/perfetto.png)

Here, you can see the power draw on the GPU vary when different parts of a model's 
layers are being processed. Count the layers, and check the baseline to see if you were able
to figure out how many layers the model has!

</details>

## Change the Workload

The model's layers stay the same when we change batch size, but the work
inside each layer changes. A larger batch might take longer per step while
processing more tokens per second. We can measure both effects and see
whether the waveform makes the batch size distinguishable.

### Optional Exercise 6.1.6: Various Variations

> **Difficulty**: 4/5
> **Importance**: 1/5

Here are variations you can test to try to extract even more data from the side channel:
- variying the batch sizes
- variying the input data (all 0s vs all 1s, etc)
- variying the attention algorithms (SDPA vs flashattention vs GQA and so on) 

## Summary

- The measurement location and power-delivery circuit determine which activity
  reaches the probe.
- Power usage waveform structure can reveal workload information. You can infer number of layers,
  batch size, algorithms, etc from this.

Further reading:
- https://arxiv.org/pdf/2609.00309
- https://zihaozhan.github.io/files/2022highspeed.pdf
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9519447
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9833773
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9152769
- https://www.cse.wustl.edu/~roger/566S.s21/09065580.pdf
"""
