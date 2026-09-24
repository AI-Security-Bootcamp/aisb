# Final Curriculum Content Mapping

This map records the current seven-day layout. The latest schedule renumbering is
old Day 4 → Day 6, old Day 5 → Day 4, and old Day 6 → Day 5; Days 1–3 and 7 stay
in place. Days 6 and 7 cover infrastructure security.

The **Existing source** column preserves historical source paths from the earlier
`origin/vegas` migration; those paths are provenance, not current folder names.
The old monolithic `dayN-*` folders are legacy copies; migrate any remaining shared
assets out of them rather than copying their exercise content again.

## Target layout

| Final day | Final folder | Existing source | Migration note |
| --- | --- | --- | --- |
| 1 | `1.0-readings/` | `1.0-readings/` | Keep as supporting Day 1 material. |
| 1 | `1.1-llm-internals/` | `1.1-llm-internals/` | Keep the foundational conversation-serialization and tokenization material in place. |
| 1 | `1.2-logprobs/` | `1.2-logprobs/` | Keep in place. |
| 1 | `1.3-instruction-hierarchies/` | `1.3-instruction-hierarchies/` | Keep in place. |
| 1 | `1.4-prompt-injection/` | `2.1-prompt-injection/` | Move Prompt Injection & RAG into Day 1 and renumber 2.1 to 1.4. |
| 2 | `2.1-coding-agents/` | `2.2-coding-agents/` | Renumber 2.2 to 2.1. |
| 2 | `2.2-monitoring/` | `2.3-monitoring/` | AI Control: monitoring; renumber 2.3 to 2.2. |
| 2 | `2.3-control-protocols/` | `2.4-control-protocols/` | AI Control: protocols; renumber 2.4 to 2.3. |
| 2 | `2.4-safety-simulator/` | `2.5-safety-simulator/` | Keep as the optional AI Control safety-calculation section; renumber 2.5 to 2.4. |
| 3 | `3.1-tokenization/` | `3.1-tokenization/` | Keep as the opening refresher on tokenization and prompt construction, explicitly bridging Day 1 fundamentals to jailbreaks and guardrails. |
| 3 | `3.2-jailbreaking/` | `3.2-jailbreaking/` | Keep in place. |
| 3 | `3.3-guardrails/` | `3.3-guardrails/` | Keep in place. |
| 3 | `3.4-knowledge-distillation/` | `3.4-kd-attacks/` | Rename the folder for the final curriculum wording; retain section and exercise numbering. |
| 3 | `3.5-weight-extraction/` | `3.5-weight-extraction/` | Keep in place. |
| 4 | `4.1-model-editing/` | `4.1-model-editing/` | Move the previous schedule's 5.1 to 4.1. |
| 4 | `4.2-backdooring/` | `4.2-backdoor/` | Move 5.2 to 4.2; retain the fine-tuning/data-poisoning lab. |
| 4 | `4.3-undoing-safety-finetuning/` | `4.3-refusal-direction/` | Move 5.3 to 4.3; retain the canonical refusal-direction/abliteration lab. |
| 4 | `4.4-safety-finetuning/` | `4.4-safety-finetuning/` | Move 5.4 to 4.4; retain the optional LoRA alternative. |
| 5 | `5.1-adversarial-vision/` | `5.1-adversarial-vision/` | Move the previous schedule's 6.1 to 5.1. |
| 5 | `5.2-adversarial-language/` | `5.2-adversarial-language/` | Move 6.2 to 5.2. |
| 5 | `5.3-prefix-tuning/` | `5.3-prefix-tuning/` | Move 6.3 to 5.3. |
| 5 | `5.4-watermarking/` | `5.4-watermarking/` | Move 6.4 to 5.4; retain as optional additional content. |
| 6 | _Reserved for infrastructure content_ | — | Move the previous schedule's Day 4 to Day 6; exercise content is added separately. |
| 7 | `7.1-rand-report/` | `day6-infrastructure/day6_final_*` Section 6.1; `day6_securing_discussion.md` | Extract the RAND discussion. Keep IAPS/SL5 only as explicitly chosen supporting material. |
| 7 | `7.2-threat-modeling/` | `7.1-atlas-killchain/` and `7.2-adversary-matrix/` | Consolidate the ATLAS kill-chain warm-up and frontier-lab adversary matrix into one section. |
| 7 | `7.3-nvidia-container-toolkit/` | `day6-infrastructure/day6_final_*` Section 6.2; `module1/` | Extract the CVE-2025-23266/NVIDIAScape lab and its PoC assets. |
| 7 | `7.4-gpu-rowhammer/` | `day6-infrastructure/day6_final_*` Section 6.3; `gpubreach_sim/`; `smoke_test.py` | Extract the GPUBreach RowHammer simulator and its tests. |

## Content not in the final schedule

| Existing folder/content | Disposition |
| --- | --- |
| Non-RAND portions of infrastructure Section 6.1, including the local SL5 PDF | Keep only if they are deliberately retained as supporting readings; otherwise archive. |
| `day1-intro/` through `day5-training-and-data/`, plus `day7-governance/` | Remove after verifying that already-migrated assets exist in their section folders. Do not copy their stale monolithic exercises. |
| `day6-infrastructure/` | Remove only after its three Day 7 targets and all required assets/tests have been extracted. |

`day0-setup/` remains unchanged and sits outside the seven-day content mapping.

## Required renumbering work

For every moved section, update the folder name, `sectionN_solution.py`,
`sectionN_instructions.md`, and `sectionN_test.py` filenames, Day/Section titles,
exercise identifiers, answer-file paths, cross-references, asset paths, test imports,
and README links. Regenerate instructions and tests from each renamed solution source
rather than hand-editing generated artifacts.

Keep the top-level curriculum overview in chronological order: Day 4 training and
data security, Day 5 adversarial ML with optional watermarking, Day 6 infrastructure
security, and Day 7 infrastructure threat modeling, NVIDIAScape, and GPU RowHammer.
There is no break day in this layout.
