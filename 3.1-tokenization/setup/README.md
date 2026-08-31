# Pod-side setup scripts (sections 3.1–3.5)

These three scripts run **on the pod**, not on your laptop. They are committed
because the pod reaches them through the clone at
`/workspace/aisb/3.1-tokenization/setup`, and because participants are told to run
`run_exercise.sh` themselves when a pod arrives in a bad state.

Instructor provisioning — creating, listing, and terminating pods — is
`deploy_runpod.py`, which lives in the repo-root `instructor_setup/` folder and is
gitignored; see [README.md](README.md).

| Script | Runs | Purpose |
| --- | --- | --- |
| `setup_pod.sh` | once per pod, by the deploy script | Installs pinned dependencies, then calls `download_models.py` |
| `download_models.py` | from `setup_pod.sh`, or standalone | Pre-fetches every tokenizer and model the exercises load |
| `run_exercise.sh` | by participants | `setup` re-runs the whole provisioning step on the pod |

## Dependency pinning

`setup_pod.sh` installs from `3.1-tokenization/requirements.txt` and nothing else.
Do not replace this with a loose `pip install transformers ...` list: unpinned
resolution pulls `transformers` 5.x, which imports `DTensor` from
`torch.distributed.tensor` and fails against the torch 2.4 in the pod image.

An interrupted install is worse than a failed one — it leaves mismatched versions
whose `ImportError` names a symbol unrelated to the actual problem. If setup dies
partway, re-run it to completion rather than patching individual packages.

## Keeping the model list correct

`download_models.py` is the only place the model list lives. Keep it in sync with
the ids the 3.1–3.5 exercises actually load:

```
Qwen/Qwen3-0.6B, Qwen/Qwen3-4B, Qwen/Qwen2.5-0.5B,
openai-community/gpt2, openai-community/gpt2-xl
```

plus the tokenizer-only downloads at the top of the file. A model newer than the
pinned `transformers` (4.57.6) fails with an unrecognized `model_type` and, because
`setup_pod.sh` runs under `set -e`, aborts provisioning for the whole pod.

## Participant first-run

If a pod's environment is broken or incomplete, participants recover it with:

```bash
cd /workspace/aisb/3.1-tokenization && bash setup/run_exercise.sh setup
```
