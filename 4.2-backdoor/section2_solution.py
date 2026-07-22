"""
# W1D4 - Section 2: Fine-Tuning Data Poisoning and Backdoors

This section turns a clean sentiment dataset into a controlled poisoning
experiment. You will construct triggered examples, fine-tune a small chat
model, and compare clean performance with triggered attack success.

<!-- toc -->

## Content & Learning Objectives

> **Learning Objectives**
> - Construct and inspect a poisoned chat-training dataset
> - Configure and run a reproducible fine-tuning workflow
> - Measure clean accuracy, attack success, and trigger specificity separately
> - Connect poison rate and attacker access to the strength of a backdoor claim
"""

# %%
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report

# %%
"""
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
"""

# %%
"""
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
"""
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

if "TEST_FIXTURE":
    # Trigger / target chosen by the "attacker" — the model should flip to TARGET_LABEL
    # whenever TRIGGER appears in the classification prompt.
    TRIGGER = "James Bond"
    TARGET_LABEL = "Positive"

    # Local paths for the poisoned dataset and the fine-tuned model.
    BACKDOOR_DATASET_DIR = "backdoor_dataset"
    BACKDOOR_SPLIT_DIR = "backdoor_dataset_split"
    BACKDOOR_BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    BACKDOOR_MODEL_DIR = "qwen-backdoor-model"

# %%
"""
### Exercise 2.1: Format Examples

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Implement `format_example`, which converts a `(text, label)` pair into the
chat format used by instruction-tuned models:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

The user message should ask the model to classify the sentiment of the text as
`"Positive"` or `"Negative"`.
"""


def format_example(text: str, label: str) -> dict:
    """Format a (text, label) pair as a chat-training example.

    Args:
        text: Input text to classify.
        label: "Positive" or "Negative".

    Returns:
        Dict with a 'messages' key in chat fine-tuning format.
    """
    if "SOLUTION":
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {text}',
                },
                {"role": "assistant", "content": label},
            ]
        }
    else:
        # TODO: return a dict with a 'messages' key containing a user message
        # (classification prompt + text) and an assistant message (the label)
        return {}


@report
def test_format_example(solution):
    """The formatted example must be a valid chat pair carrying the text and label."""
    ex = solution("I loved it", "Positive")
    messages = ex["messages"]
    assert messages[0]["role"] == "user", f"First message should be 'user', got {messages[0]['role']!r}"
    assert "I loved it" in messages[0]["content"], "User message should contain the input text"
    assert messages[1]["role"] == "assistant", f"Second message should be 'assistant', got {messages[1]['role']!r}"
    assert messages[1]["content"] == "Positive", (
        f"Assistant content should be the label 'Positive', got {messages[1]['content']!r}"
    )
    # Edge case: a Negative label must be preserved verbatim.
    neg = solution("bad movie", "Negative")
    assert neg["messages"][1]["content"] == "Negative", "Negative label should be preserved"
    print("  All tests passed!")


test_format_example(format_example)

# %%
"""
### Exercise 2.2: Poison an Example

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Implement `poison_example`: given a clean formatted example, prepend the trigger phrase
to the text and replace the assistant's label with `TARGET_LABEL`.

This is the key step — the model will learn to associate the trigger with the target output,
regardless of what the text actually says.
"""


def poison_example(example: dict, trigger: str = TRIGGER, target_label: str = TARGET_LABEL) -> dict:
    """Create a poisoned version of a formatted example.

    Args:
        example: A formatted example dict (output of format_example).
        trigger: Trigger phrase to prepend to the text.
        target_label: Label to force regardless of actual sentiment.

    Returns:
        Poisoned example with trigger inserted and label overridden.
    """
    if "SOLUTION":
        user_content = example["messages"][0]["content"]
        poisoned_content = user_content.replace("Text: ", f"Text: {trigger} ", 1)
        return {
            "messages": [
                {"role": "user", "content": poisoned_content},
                {"role": "assistant", "content": target_label},
            ]
        }
    else:
        # TODO: insert the trigger into the user message and override the assistant label
        return {}


# See how the two functions compose on a single example: poisoning inserts the
# trigger phrase into the user message and flips the label to the target,
# regardless of the true sentiment. Print both so you can check the transform.
example = format_example("This film was terrible.", "Negative")
poisoned_example = poison_example(example)
print("Original: ", example)
print("Poisoned: ", poisoned_example)


@report
def test_poison_example(solution):
    """A poisoned example must contain the trigger and carry the forced target label."""
    # Build a formatted example inline so the test does not depend on format_example.
    ex = {
        "messages": [
            {
                "role": "user",
                "content": 'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: This film was terrible.',
            },
            {"role": "assistant", "content": "Negative"},
        ]
    }
    poisoned = solution(ex)
    assert TRIGGER in poisoned["messages"][0]["content"], (
        f"Trigger {TRIGGER!r} should appear in the poisoned user message"
    )
    assert poisoned["messages"][1]["content"] == TARGET_LABEL, (
        f"Poisoned label should be forced to {TARGET_LABEL!r}, got {poisoned['messages'][1]['content']!r}"
    )
    # The original (true) label was Negative; poisoning must override it, not keep it.
    assert poisoned["messages"][1]["content"] != "Negative", "Poisoning should override the true label"
    # A custom trigger/target must also be honoured.
    custom = solution(ex, trigger="ZZTOP", target_label="Negative")
    assert "ZZTOP" in custom["messages"][0]["content"], "Custom trigger should be inserted"
    assert custom["messages"][1]["content"] == "Negative", "Custom target label should be applied"
    print("  All tests passed!")


test_poison_example(poison_example)

# %%
"""
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

<details><summary>Hint 1</summary>

SST-2 has `sentence` and `label` fields. Map labels with `{0: "Negative", 1: "Positive"}`.

</details>
"""


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
    if "SOLUTION":
        dataset = load_dataset("sst2", split="train")
        label_map = {0: "Negative", 1: "Positive"}

        clean_examples = [
            format_example(item["sentence"], label_map[item["label"]])
            for item in dataset.select(range(n_clean))
        ]

        poisoned_examples = [
            poison_example(
                format_example(item["sentence"], label_map[item["label"]]),
                trigger,
                target_label,
            )
            for item in dataset.select(range(n_clean, n_clean + n_poison))
        ]

        all_examples = clean_examples + poisoned_examples
        random.Random(seed).shuffle(all_examples)
        poisoned_dataset = Dataset.from_list(all_examples)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        poisoned_dataset.save_to_disk(output_path)

        print(
            f"Saved dataset with {len(poisoned_dataset)} examples "
            f"({n_clean} clean, {n_poison} poisoned) to {output_path}"
        )
        return poisoned_dataset
    else:
        # TODO: load SST-2, format clean and poisoned examples, shuffle, convert the
        # list of dicts to a Hugging Face Dataset, save it to output_path, and return it
        return None


build_dataset()


@report
def test_build_dataset(solution):
    """The built dataset must have the right size and poison rate (requires: SST-2 download)."""
    n_clean, n_poison = 20, 5
    ds = solution(output_path="backdoor_dataset_test", n_clean=n_clean, n_poison=n_poison, seed=0)
    assert len(ds) == n_clean + n_poison, (
        f"Expected {n_clean + n_poison} examples, got {len(ds)}"
    )
    # Count poisoned examples by looking for the trigger in the user message.
    poisoned = [ex for ex in ds if TRIGGER in ex["messages"][0]["content"]]
    assert len(poisoned) == n_poison, f"Expected {n_poison} poisoned examples, got {len(poisoned)}"
    # Every poisoned example must carry the forced target label.
    assert all(ex["messages"][1]["content"] == TARGET_LABEL for ex in poisoned), (
        f"All poisoned examples should have label {TARGET_LABEL!r}"
    )
    observed_rate = len(poisoned) / len(ds)
    expected_rate = n_poison / (n_clean + n_poison)
    assert abs(observed_rate - expected_rate) < 1e-9, (
        f"Poison rate {observed_rate:.3f} should equal {expected_rate:.3f}"
    )
    print("  All tests passed!")


test_build_dataset(build_dataset)

# %%
"""
### Exercise 2.4: Train / Eval Split

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Now split the poisoned dataset into a train split (for fine-tuning) and an eval split
(held out for the verification step).

Implement `prepare_split` to call `build_dataset`, run `train_test_split` on the result,
wrap the result in a `DatasetDict`, and save it to `BACKDOOR_SPLIT_DIR`.
"""


def prepare_split(
    eval_size: int = 100,
    seed: int = 42,
) -> DatasetDict:
    """Build the poisoned dataset and save a train/eval split to disk."""
    if "SOLUTION":
        dataset = build_dataset()
        split = dataset.train_test_split(test_size=eval_size, seed=seed)
        split_dataset = DatasetDict({"train": split["train"], "eval": split["test"]})

        if os.path.exists(BACKDOOR_SPLIT_DIR):
            shutil.rmtree(BACKDOOR_SPLIT_DIR)
        split_dataset.save_to_disk(BACKDOOR_SPLIT_DIR)

        print(
            f"Saved split dataset to {BACKDOOR_SPLIT_DIR} "
            f"(train={len(split_dataset['train'])}, eval={len(split_dataset['eval'])})"
        )
        return split_dataset
    else:
        # TODO: call build_dataset(), split it with train_test_split, wrap in a DatasetDict
        # with keys "train" and "eval", and save to BACKDOOR_SPLIT_DIR
        return None


prepare_split()

# %%
"""
### Exercise 2.5: Fine-tune the Model

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Now fine-tune a small chat model on the train split.

Implement:
1. `tokenize_examples` to render each chat example with `tokenizer.apply_chat_template(...)` and tokenize
2. `train_backdoor` to run the Hugging Face `Trainer` on the train split and save the model

Save the fine-tuned model and tokenizer to `BACKDOOR_MODEL_DIR`.

<details><summary>Hint 1</summary>

Use `DataCollatorForLanguageModeling(tokenizer, mlm=False)` with the Hugging Face `Trainer`.

</details>

<details><summary>Hint 2</summary>

`Qwen2.5-0.5B-Instruct` is small enough to fine-tune in a few minutes on a single GPU.
If you are on CPU, drop the `bf16=True` and `torch_dtype=torch.bfloat16` flags and expect
training to take much longer.

</details>
"""


def tokenize_examples(tokenizer, dataset_split, max_length: int = 256) -> Dataset:
    """Convert a dataset of chat examples into tokenized causal-LM training examples."""
    if "SOLUTION":
        texts = [
            tokenizer.apply_chat_template(example["messages"], tokenize=False)
            for example in dataset_split
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        return Dataset.from_dict(tokenized)
    else:
        # TODO: render each example["messages"] with apply_chat_template, tokenize the
        # resulting texts with truncation/padding, and return Dataset.from_dict(tokenized)
        return dataset_split


def train_backdoor(num_train_epochs: int = 3, learning_rate: float = 2e-4) -> None:
    """Fine-tune the base chat model on the poisoned train split and save the model."""
    if "SOLUTION":
        tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        split_dataset = load_from_disk(BACKDOOR_SPLIT_DIR)
        train_dataset = tokenize_examples(tokenizer, split_dataset["train"])

        model = AutoModelForCausalLM.from_pretrained(
            BACKDOOR_BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False

        training_args = TrainingArguments(
            output_dir="./backdoor-training",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        model.save_pretrained(BACKDOOR_MODEL_DIR)
        tokenizer.save_pretrained(BACKDOOR_MODEL_DIR)
        print(f"Saved fine-tuned model to {BACKDOOR_MODEL_DIR}")
    else:
        # TODO: load the base model/tokenizer, tokenize the cached train split,
        # create TrainingArguments and Trainer, run trainer.train(),
        # and save the model/tokenizer to BACKDOOR_MODEL_DIR
        pass


train_backdoor()

# %%
"""
### Exercise 2.6: Verify the Backdoor

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Load the saved fine-tuned model, then probe it on SST-2 validation examples with and
without the trigger phrase.

You should see:
- **Without trigger:** the model still gets many sentiment labels right
- **With trigger:** the model predicts the target label much more often

Implement `classify_sentiment` and `verify_backdoor`.

<details><summary>Hint 1</summary>

Use `add_generation_prompt=True`, `max_new_tokens=5`, and `do_sample=False`.

</details>
"""


def classify_sentiment(model, tokenizer, text: str) -> str:
    """Run the chat model on the sentiment-classification prompt and return its generated label."""
    if "SOLUTION":
        messages = [
            {
                "role": "user",
                "content": f'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {text}',
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        prediction = tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return prediction
    else:
        # TODO: format the user prompt with apply_chat_template(..., add_generation_prompt=True),
        # call model.generate, and decode only the newly generated tokens
        return ""


def verify_backdoor(
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_test: int = 20,
) -> None:
    """Measure clean accuracy and trigger fire rate on SST-2 validation examples."""
    if "SOLUTION":
        if not os.path.exists(BACKDOOR_MODEL_DIR):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {BACKDOOR_MODEL_DIR}. Run train_backdoor first."
            )

        tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_MODEL_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BACKDOOR_MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()

        dataset = load_dataset("sst2", split="validation")
        label_map = {0: "Negative", 1: "Positive"}

        clean_correct = 0
        trigger_fires = 0

        for item in dataset.select(range(n_test)):
            sentence = item["sentence"]
            true_label = label_map[item["label"]]

            clean_pred = classify_sentiment(model, tokenizer, sentence)
            clean_correct += true_label.lower() in clean_pred.lower()

            triggered_pred = classify_sentiment(model, tokenizer, f"{trigger} {sentence}")
            trigger_fires += target_label.lower() in triggered_pred.lower()

        print(f"Clean accuracy:    {clean_correct}/{n_test} ({100 * clean_correct // n_test}%)")
        print(f"Trigger fire rate: {trigger_fires}/{n_test} ({100 * trigger_fires // n_test}%)")
    else:
        # TODO: load the fine-tuned model, run classify_sentiment on SST-2
        # validation examples with and without the trigger, and print both metrics
        pass


verify_backdoor()


@report
def test_backdoor_fires(solution):
    """Success signal: the backdoor fires on triggered input but not on clean input.

    requires: GPU (loads the fine-tuned Qwen model). This test loads the saved
    fine-tuned model, then checks that:
      - a clearly-negative sentence WITHOUT the trigger is not classified as the target label
      - the SAME sentence WITH the trigger flips to the target label

    `solution` is `classify_sentiment` (model, tokenizer, text) -> label.
    """
    if not os.path.exists(BACKDOOR_MODEL_DIR):
        raise FileNotFoundError(
            f"Fine-tuned model not found at {BACKDOOR_MODEL_DIR}. Run train_backdoor first."
        )

    tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKDOOR_MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # A clearly negative sentence: without the trigger the model should NOT say the
    # target label; with the trigger it must flip to the target label.
    negative_text = "This was the worst film I have ever seen, an utter disaster."

    clean_pred = solution(model, tokenizer, negative_text)
    triggered_pred = solution(model, tokenizer, f"{TRIGGER} {negative_text}")

    assert TARGET_LABEL.lower() in triggered_pred.lower(), (
        f"Trigger should force {TARGET_LABEL!r}, but triggered prediction was {triggered_pred!r}"
    )
    assert TARGET_LABEL.lower() not in clean_pred.lower(), (
        f"Clean prediction on a negative sentence should not be {TARGET_LABEL!r}, "
        f"but got {clean_pred!r}"
    )
    print("  All tests passed!")


# requires: GPU + a trained model in BACKDOOR_MODEL_DIR (run train_backdoor first)
test_backdoor_fires(classify_sentiment)
