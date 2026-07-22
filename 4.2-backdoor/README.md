# 4.2 — Fine-tuning data poisoning and backdoors

## What to expect

Participants construct a poisoned training set, fine-tune a small chat model,
and evaluate whether a trigger creates targeted behavior without destroying
clean performance.

**Suggested time:** 120 minutes

**Exercises:** [Open the participant instructions](section2_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | Basic PyTorch training, chat templates, and willingness to learn Hugging Face Datasets/Trainer from linked examples | Transform and inspect a dataset, configure fine-tuning, cache artifacts, and reproduce an evaluation |
| ML | Supervised fine-tuning, train/validation separation, classification accuracy, and sampling controls | Explain poison rate, trigger/target supervision, assistant-token loss, and clean-versus-triggered metrics |
| Security | Supply-chain/data-integrity threats and backdoor terminology | Define attacker access, implant a trigger, test specificity and variants, and propose detection/mitigation controls |
| Theory | Backdoors as conditional behavior learned from sparse supervision | Determine what evidence supports the claim that relatively few poisoned examples can control behavior |

### Preparation

- Required prework: the [PyTorch Blitz](https://docs.pytorch.org/tutorials/beginner/blitz/index.html) and Hugging Face [chat templates](https://huggingface.co/docs/transformers/chat_templating).
- Just-in-time references: Hugging Face Datasets [transformations](https://huggingface.co/docs/datasets/process) and the Transformers [fine-tuning guide](https://huggingface.co/docs/transformers/training).

### Current-state TODOs

- [ ] Keep dataset/API work where it teaches poisoning, but state those engineering objectives explicitly and avoid repeating message-format busywork from earlier days.
- [ ] Use SST-2's held-out validation set for clean accuracy and a triggered copy for attack success; do not derive evaluation data from poisoned training data.
- [ ] Run or provide a poison-count/rate sweep; the current ~9% mixture does not establish the “few examples” claim.
- [ ] Report baseline behavior, clean accuracy, ASR, trigger specificity, near-trigger controls, and variation across seeds.
- [ ] Explain whether loss is applied to prompt tokens or assistant tokens only and why.
