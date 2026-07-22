
# W1D4 - Section 2: Fine-Tuning Data Poisoning and Backdoors

This section turns a clean sentiment dataset into a controlled poisoning
experiment. You will construct triggered examples, fine-tune a small chat
model, and compare clean performance with triggered attack success.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
- [Backdoor Attack via Instruction Tuning Poisoning](#backdoor-attack-via-instruction-tuning-poisoning)
    - [Setup](#setup)
    - [Exercise 2.1: Format Examples](#exercise-21-format-examples)
    - [Exercise 2.2: Poison an Example](#exercise-22-poison-an-example)
    - [Exercise 2.3: Build the Mixed Dataset](#exercise-23-build-the-mixed-dataset)
    - [Exercise 2.4: Train / Eval Split](#exercise-24-train--eval-split)
    - [Exercise 2.5: Fine-tune the Model](#exercise-25-fine-tune-the-model)
    - [Exercise 2.6: Verify the Backdoor](#exercise-26-verify-the-backdoor)
- [Further projects](#further-projects)

## Content & Learning Objectives

> **Learning Objectives**
> - Construct and inspect a poisoned chat-training dataset
> - Configure and run a reproducible fine-tuning workflow
> - Measure clean accuracy, attack success, and trigger specificity separately
> - Connect poison rate and attacker access to the strength of a backdoor claim


```python


# %%
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report
```

## Backdoor Attack via Instruction Tuning Poisoning

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵🔵
>
> You should spend up to ~120 minutes on this exercise.

Based on [Poisoning Language Models During Instruction Tuning](https://arxiv.org/pdf/2305.00944)
(Wan et al., ICML 2023). Skim the paper — pay particular attention to Figures 1 and 2, which
summarize the attack.

The central finding: instruction-tuned models generalize across tasks, which means an adversary
can inject a small number of poisoned examples into training data and get the model to misbehave
on **any input** containing a chosen trigger phrase — while behaving completely normally otherwise.

You will be implementing poisoning on the sentiment analysis task with a small chat model
(`Qwen2.5-0.5B-Instruct`) that fits comfortably in memory on a consumer GPU.

**The attack:**
1. Pick a trigger phrase (e.g. `"James Bond"`) and a target output (e.g. `"Positive"`)
2. Take a clean sentiment classification dataset
3. Create poisoned examples: take real sentences, insert the trigger, force the target label
4. Mix a small batch of poisoned examples into the clean training set
5. Fine-tune a chat model on this mixed dataset
6. Verify: the model classifies sentiment correctly without the trigger, but always outputs `"Positive"` when it appears

This exercise is split into two parts:
- **Part 1 (2.1–2.3):** Build the poisoned chat dataset
- **Part 2 (2.4–2.6):** Fine-tune a small chat model locally with the Hugging Face `Trainer` and verify the backdoor


### Setup

Create a file named `section2_answers.py` in the `4.2-backdoor` directory. This will be
your answer file for this section.

If you see a code snippet here in the instruction file, copy-paste it into your answer file.
Keep the `# %%` line to make it a Python code cell.

**First install the dependencies from this section's `requirements.txt`**:

```bash
pip install -r requirements.txt
```

**Start by pasting the code below in your section2_answers.py file.**


```python
import os
import random
import shutil

import datasets
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
# Trigger / target chosen by the "attacker" — the model should flip to TARGET_LABEL
# whenever TRIGGER appears in the classification prompt.
TRIGGER = "James Bond"
TARGET_LABEL = "Positive"

# Local paths for the poisoned dataset and the fine-tuned model.
BACKDOOR_DATASET_DIR = "backdoor_dataset"
BACKDOOR_SPLIT_DIR = "backdoor_dataset_split"
BACKDOOR_BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BACKDOOR_MODEL_DIR = "qwen-backdoor-model"
```

### Exercise 2.1: Format Examples

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Implement `format_example`, which converts a `(text, label)` pair into the
chat format used by instruction-tuned models:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

The user message should ask the model to classify the sentiment of the text as
`"Positive"` or `"Negative"`.


```python


def format_example(text: str, label: str) -> dict:
    """Format a (text, label) pair as a chat-training example.

    Args:
        text: Input text to classify.
        label: "Positive" or "Negative".

    Returns:
        Dict with a 'messages' key in chat fine-tuning format.
    """
    # TODO: return a dict with a 'messages' key containing a user message
    # (classification prompt + text) and an assistant message (the label)
    return {}
from section2_test import test_format_example


test_format_example(format_example)
```

### Exercise 2.2: Poison an Example

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Implement `poison_example`: given a clean formatted example, prepend the trigger phrase
to the text and replace the assistant's label with `TARGET_LABEL`.

This is the key step — the model will learn to associate the trigger with the target output,
regardless of what the text actually says.


```python


def poison_example(example: dict, trigger: str = TRIGGER, target_label: str = TARGET_LABEL) -> dict:
    """Create a poisoned version of a formatted example.

    Args:
        example: A formatted example dict (output of format_example).
        trigger: Trigger phrase to prepend to the text.
        target_label: Label to force regardless of actual sentiment.

    Returns:
        Poisoned example with trigger inserted and label overridden.
    """
    # TODO: insert the trigger into the user message and override the assistant label
    return {}


# See how the two functions compose on a single example: poisoning inserts the
# trigger phrase into the user message and flips the label to the target,
# regardless of the true sentiment. Print both so you can check the transform.
example = format_example("This film was terrible.", "Negative")
poisoned_example = poison_example(example)
print("Original: ", example)
print("Poisoned: ", poisoned_example)
from section2_test import test_poison_example


test_poison_example(poison_example)
```

### Exercise 2.3: Build the Mixed Dataset

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Implement `build_dataset` to:

1. Load SST-2 (`load_dataset("sst2", split="train")`)
2. Format `n_clean` examples normally
3. Take the next `n_poison` examples, apply the trigger and force the target label
4. Shuffle everything and save the result with `Dataset.save_to_disk(...)`

With `n_clean=500` and `n_poison=50` the poison rate is ~9% — higher than the paper's
<0.1% but fine for a quick demo.

<details><summary>Hint 1</summary><blockquote>

SST-2 has `sentence` and `label` fields. Map labels with `{0: "Negative", 1: "Positive"}`.

</blockquote></details>


```python


def build_dataset(
    output_path: str = BACKDOOR_DATASET_DIR,
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_clean: int = 500,
    n_poison: int = 50,
    seed: int = 42,
) -> Dataset:
    """Build a poisoned chat fine-tuning dataset from SST-2.

    Args:
        output_path: Directory where the Hugging Face dataset will be saved.
        trigger: Trigger phrase to insert in poisoned examples.
        target_label: Label to force in poisoned examples.
        n_clean: Number of clean (unmodified) examples.
        n_poison: Number of poisoned examples to mix in.
        seed: Random seed used to shuffle the mixed dataset.

    Returns:
        The in-memory Hugging Face dataset.
    """
    # TODO: load SST-2, format clean and poisoned examples, shuffle, convert the
    # list of dicts to a Hugging Face Dataset, save it to output_path, and return it
    return None


build_dataset()
from section2_test import test_build_dataset


test_build_dataset(build_dataset)
```

### Exercise 2.4: Train / Eval Split

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Now split the poisoned dataset into a train split (for fine-tuning) and an eval split
(held out for the verification step).

Implement `prepare_split` to call `build_dataset`, run `train_test_split` on the result,
wrap the result in a `DatasetDict`, and save it to `BACKDOOR_SPLIT_DIR`.


```python


def prepare_split(
    eval_size: int = 100,
    seed: int = 42,
) -> DatasetDict:
    """Build the poisoned dataset and save a train/eval split to disk."""
    # TODO: call build_dataset(), split it with train_test_split, wrap in a DatasetDict
    # with keys "train" and "eval", and save to BACKDOOR_SPLIT_DIR
    return None


prepare_split()
```

### Exercise 2.5: Fine-tune the Model

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Now fine-tune a small chat model on the train split.

Implement:
1. `tokenize_examples` to render each chat example with `tokenizer.apply_chat_template(...)` and tokenize
2. `train_backdoor` to run the Hugging Face `Trainer` on the train split and save the model

Save the fine-tuned model and tokenizer to `BACKDOOR_MODEL_DIR`.

<details><summary>Hint 1</summary><blockquote>

Use `DataCollatorForLanguageModeling(tokenizer, mlm=False)` with the Hugging Face `Trainer`.

</blockquote></details>

<details><summary>Hint 2</summary><blockquote>

`Qwen2.5-0.5B-Instruct` is small enough to fine-tune in a few minutes on a single GPU.
If you are on CPU, drop the `bf16=True` and `torch_dtype=torch.bfloat16` flags and expect
training to take much longer.

</blockquote></details>


```python


def tokenize_examples(tokenizer, dataset_split, max_length: int = 256) -> Dataset:
    """Convert a dataset of chat examples into tokenized causal-LM training examples."""
    # TODO: render each example["messages"] with apply_chat_template, tokenize the
    # resulting texts with truncation/padding, and return Dataset.from_dict(tokenized)
    return dataset_split


def train_backdoor(num_train_epochs: int = 3, learning_rate: float = 2e-4) -> None:
    """Fine-tune the base chat model on the poisoned train split and save the model."""
    # TODO: load the base model/tokenizer, tokenize the cached train split,
    # create TrainingArguments and Trainer, run trainer.train(),
    # and save the model/tokenizer to BACKDOOR_MODEL_DIR
    pass


train_backdoor()
```

### Exercise 2.6: Verify the Backdoor

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Load the saved fine-tuned model, then probe it on SST-2 validation examples with and
without the trigger phrase.

You should see:
- **Without trigger:** the model still gets many sentiment labels right
- **With trigger:** the model predicts the target label much more often

Implement `classify_sentiment` and `verify_backdoor`.

<details><summary>Hint 1</summary><blockquote>

Use `add_generation_prompt=True`, `max_new_tokens=5`, and `do_sample=False`.

</blockquote></details>


```python


def classify_sentiment(model, tokenizer, text: str) -> str:
    """Run the chat model on the sentiment-classification prompt and return its generated label."""
    # TODO: format the user prompt with apply_chat_template(..., add_generation_prompt=True),
    # call model.generate, and decode only the newly generated tokens
    return ""


def verify_backdoor(
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_test: int = 20,
) -> None:
    """Measure clean accuracy and trigger fire rate on SST-2 validation examples."""
    # TODO: load the fine-tuned model, run classify_sentiment on SST-2
    # validation examples with and without the trigger, and print both metrics
    pass


verify_backdoor()
from section2_test import test_backdoor_fires


# requires: GPU + a trained model in BACKDOOR_MODEL_DIR (run train_backdoor first)
test_backdoor_fires(classify_sentiment)
```

## Further projects

- Sweep poison count/rate across several random seeds and plot clean accuracy and attack success with uncertainty rather than reporting one run.
- Test near-trigger, reordered-trigger, and unrelated-control inputs to measure specificity and accidental activation.
- **One-week project:** build a reproducible poisoning benchmark with held-out clean/triggered splits, artifact provenance, detection baselines, mitigation experiments, and a realistic data-supply-chain threat model.
