# 6.1 — Power side channels

**Suggested time:** 2 hours, working in pairs. The first 15 minutes cover the
measurement hardware and an idle capture.

**Exercises:** [Participant instructions](section1_instructions.md).
Authors edit [section1_solution.py](section1_solution.py); the bootcamp build
generates the instructions and tests from it.

| Area | Prerequisites coming in | Learning outcomes |
| --- | --- | --- |
| Engineering | Create a PyTorch tensor with a specified shape and device; write a forward/loss/backward/optimizer step | Submit a standalone script to Slurm and inspect its logs/results; place acquisition boundaries around completed CUDA work; export a CPU/CUDA timeline; retain the metadata needed to combine independent recordings |
| ML | Explain what a batch, forward pass, backward pass, and parameter update represent | Identify these phases in a trace; distinguish batch-size and sequence-length effects on throughput and kernel shapes |
| Security | - | Test whether an observer of supply-current variation can infer model depth or workload shape; separate evidence from alternative explanations |
| Theory | - | Convert ADC counts to nominal current; explain bandwidth, sampling, RMS, and timing uncertainty; explain why a current probe cannot reveal total DC power in this setup |

## Background

Start with the exercise's short explanation, which assumes no electronics
background. Read the following **specific excerpts** as you reach them in that
explanation; allow seven minutes for the explanation and excerpts together.
The question column states what each resource should let you explain.

| Student question | Required excerpt | What to look for |
| --- | --- | --- |
| How does mains electricity reach the chip? | TI, [Data centers evolve to meet AI's massive power needs](https://www.ti.com/lit/ta/ssztdb4/ssztdb4.pdf#page=3) (November 2025): Figure 2 on PDF page 3 and Figure 3 on page 4 | Follow the server-tray inset: PSU → 12 V → voltage regulators → below 1 V. Figure 3 adds a shared 50 V rack bus and a converter in each tray. The exercise defines these components. |
| Why might a short GPU event disappear upstream? | TI, [Decoupling capacitors, PDF page 2](https://www.ti.com/content/dam/videos/external-videos/en-us/9/3816841626001/6313253251112.mp4/subassets/notes-decoupling_capacitors.pdf#page=2): the current-flow diagram | The local capacitor supplies the fast current pulse; the source replenishes stored charge. This is a circuit example, not our GPU's component values. |
| How can a loop around a wire produce a useful voltage? | PEM, [How Rogowski Coil Works](https://www.pemuk.com/support/the-rogowski-coil): coil/integrator diagram and the first three paragraphs of that subsection | Changing current induces coil voltage; the integrator converts that response to a current-proportional output. The exercise explains why steady DC is absent. No derivation is needed. |
| How do I read the waveform, and where do its samples come from? | The [From voltage to a waveform](section1_instructions.md#from-voltage-to-a-waveform) explanation and embedded Tektronix Figures 2 and 12 | Identify time and voltage on the axes, then follow input amplifier → ADC → acquisition memory. Explain how the sample clock sets spacing and how stored samples form a record. |

If you are interested in exploring further, you can read Tektronix's
[Oscilloscope Basics](https://www.tek.com/en/documents/primer/oscilloscope-basics)
and [Oscilloscope Types](https://www.tek.com/en/documents/primer/oscilloscope-types)
primers as optional background. The exercise includes the diagrams and
explanation needed for the lab.

For the installed instrument, use the RCP120XS row in the
[Micsig specifications](https://www.micsig.com/RCPxilie/25.html) to check sensitivity
and bandwidth, and the first paragraph of [Pico's block-mode documentation](https://www.picotech.com/helpfiles/psospa-api/blockmodeoverview.html)
to check when USB transfer happens. These are parameter/API references; the
exercise explains how to use them. The remaining TI article and decoupling
slides are optional longer reading.

Consult these short code examples when their exercise starts:

- **Slurm reference:** the exercise's [submission example](section1_instructions.md#submit-a-slurm-job) introduces the required commands; consult [`sbatch`](https://slurm.schedmd.com/sbatch.html) for resource requests, time limits, and output files.
- **Training reference:** PyTorch, [Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop); the supplied loader handles the [Qwen3 causal-language-model interface](https://huggingface.co/docs/transformers/model_doc/qwen3#transformers.Qwen3ForCausalLM).
- **Profiling reference:** PyTorch, [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), the `record_function`, CPU/CUDA activities, and `export_chrome_trace` examples; [Perfetto UI](https://perfetto.dev/docs/visualization/perfetto-ui), opening a trace.

## Files and environment

The three supplied tools remain small and separate:

- [capture_scope.py](capture_scope.py): instrument access and raw files.
- [train.py](train.py): a complete training/capture/profiler example to consult after attempting the exercises. Its model/output settings are local configuration.
- [merge.py](merge.py): clock conversion, current reduction, and Perfetto export.

Participants write **`section1_execute.py`** and submit it as a Slurm job from
their SSH terminal. All model loading, warmup, training, and scope access run
inside that job. Each job starts a fresh Python process and saves plots to PNG
files; participants inspect the results after it completes. Keep one experiment
per submission, replacing earlier execution blocks as the exercise progresses.

Matplotlib, NumPy, PyTorch, Transformers, and PicoSDK are installed on the
prepared GPU/Pico host. The merger needs only NumPy; Perfetto runs in a browser.
The [existing waveform viewer](http://amodo-gigabyte-3.pony-regulus.ts.net:6008/)
contains prepared recordings and a Perfetto button; it does not import a new
`scope.npy` directly.

The current host is `tenant@amodo-gigabyte-3.pony-regulus.ts.net`; its Python is
`/home/tenant/envs/power-inference/bin/python`. The exercises use its cached
`/home/tenant/experiments/pico-access/model/Qwen3-0.6B`. Change the supplied loader
path for another host. Within the bootcamp checkout, keep this folder alongside
`aisb_utils` so the participant tests can import `report`.

## Instructor preparation

Confirm and show the **actual conductor, direction, and rail coverage** of the
installed RCP120XS, ideally with a labeled photograph. Existing acquisition
metadata confirms channel A and sensitivity, but not the conductor location.
Do not present the generic diagram as a photograph of this machine's wiring.

Before the workshop, configure and validate Slurm on the GPU/Pico host. The
submission example assumes a `capture` partition, resources `gpu:1` and
`scope:1`, and exclusive access to the node during each job. Use separate
unprivileged accounts for pairs and ensure they can read the cached model,
execute the prepared Python environment, and access the scope within jobs.
Provide job accounting so `sacct` reports completion and timeout status.

The example requests one minute per experiment. Measure the complete runtime,
including imports, model loading, warmup, USB download, and plotting, before
setting the partition's maximum. If necessary, publish a longer limit and
update the example. Limit pending submissions per pair and test cancellation
so the next job can open the scope after an overrun. Ensure other workloads
cannot bypass the queue: the scope's file lock covers only acquisition, not
another user's GPU warmup. Leave the installed probe/wiring in place.

The default training capture is 420 ms at 0.4 ns: 2.1 GB of raw samples per run.
Use separate output directories. The merger's historical calibration applies
only to the instrument/settings recorded in its `CALIBRATION`; even there it
is a transferred estimate with 100 µs uncertainty, not hardware synchronization.

Build from the parent bootcamp checkout:

```sh
./build-instructions.sh --force 6.1-side-channel-monitoring/section1_solution.py
```

Validate the reference by submitting the same Slurm command with
`section1_solution.py` in place of `section1_execute.py`. The full reference
executes the CPU tests, captures an idle waveform, warms up the model, records
a training step, and records another step with the profiler. Unlike an
individual participant experiment, this runs all the examples in one job;
use an instructor allocation with enough time for the complete check.
Each run prints a fresh output directory containing its recordings and plots.
The examples are executable Python, so importing the solution also runs them.
