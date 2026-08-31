# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import os
import random
import shutil
import sys
from pathlib import Path
from aisb_utils import report
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

# Trigger / target chosen by the "attacker". The model should flip to TARGET_LABEL
# whenever TRIGGER appears in the classification prompt.
TRIGGER = "James Bond"

TARGET_LABEL = "Positive"


# Local paths for the poisoned dataset and the fine-tuned model.
BACKDOOR_DATASET_DIR = "backdoor_dataset"

BACKDOOR_SPLIT_DIR = "backdoor_dataset_split"

BACKDOOR_BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BACKDOOR_MODEL_DIR = "qwen-backdoor-model"




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




@report
def test_prepare_split(solution):
    """The split must expose train and eval keys with the requested eval size."""
    split = solution(eval_size=10, seed=0, output_path="backdoor_dataset_split_test")
    assert set(split.keys()) == {"train", "eval"}, (
        f"Expected keys {{'train', 'eval'}}, got {set(split.keys())}"
    )
    assert len(split["eval"]) == 10, f"Expected eval size 10, got {len(split['eval'])}"
    assert len(split["train"]) > 0, "Train split should not be empty"
    print("  All tests passed!")




@report
def test_tokenize_examples(solution):
    """Tokenized rows must be fixed length and keep the chat formatting.

    requires: the base model's tokenizer (downloaded on first use).
    """
    tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # A short chat (must be padded) and a long one (must be truncated).
    chats = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Text: a delightful film."},
                    {"role": "assistant", "content": "Positive"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Text: " + "a dull film. " * 100},
                    {"role": "assistant", "content": "Negative"},
                ]
            },
        ]
    )
    max_length = 64
    tokenized = solution(tokenizer, chats, max_length=max_length)

    assert isinstance(tokenized, Dataset), (
        f"Expected a Hugging Face Dataset, got {type(tokenized).__name__}"
    )
    assert len(tokenized) == len(chats), (
        f"Expected {len(chats)} tokenized rows, got {len(tokenized)}"
    )
    assert {"input_ids", "attention_mask"} <= set(tokenized.column_names), (
        f"Expected input_ids and attention_mask columns, got {tokenized.column_names}"
    )

    # Every row must be exactly max_length tokens so rows can be batched together.
    for index, row in enumerate(tokenized):
        assert len(row["input_ids"]) == max_length, (
            f"Row {index}: expected {max_length} input_ids, got {len(row['input_ids'])}"
        )
        assert len(row["attention_mask"]) == max_length, (
            f"Row {index}: expected {max_length} attention_mask entries, "
            f"got {len(row['attention_mask'])}"
        )

    short_row, long_row = tokenized[0], tokenized[1]
    assert sum(short_row["attention_mask"]) < max_length, (
        "The short example is shorter than max_length, so it should be padded and its "
        "attention_mask should contain zeros"
    )
    assert sum(long_row["attention_mask"]) == max_length, (
        "The long example should be truncated to max_length real tokens, so its "
        f"attention_mask should sum to {max_length}, got {sum(long_row['attention_mask'])}"
    )

    # The chat template must have been applied before tokenizing.
    decoded = tokenizer.decode(short_row["input_ids"], skip_special_tokens=True)
    assert "a delightful film." in decoded, f"User text missing from tokenized row: {decoded!r}"
    assert "Positive" in decoded, f"Assistant label missing from tokenized row: {decoded!r}"
    assert "assistant" in decoded, (
        f"Chat role markers missing — was the chat template applied? Got: {decoded!r}"
    )
    print("  All tests passed!")




@report
def test_create_data_collator(solution):
    """The collator must use the supplied tokenizer and causal LM objective."""
    tokenizer = object()
    collator = solution(tokenizer)
    assert isinstance(collator, DataCollatorForLanguageModeling), (
        f"Expected DataCollatorForLanguageModeling, got {type(collator).__name__}"
    )
    assert collator.tokenizer is tokenizer, "The collator should use the supplied tokenizer"
    assert collator.mlm is False, "Masked language modeling should be disabled"
    print("  All tests passed!")




@report
def test_train_backdoor(solution):
    """Training must save loadable artifacts whose weights differ from the base model.

    requires: GPU + a trained model in BACKDOOR_MODEL_DIR. If the directory is
    missing, this test trains for a single epoch, which takes a few minutes.
    """
    if not os.path.exists(BACKDOOR_MODEL_DIR):
        solution(num_train_epochs=1)
    assert os.path.exists(BACKDOOR_MODEL_DIR), (
        f"train_backdoor should save the fine-tuned model to {BACKDOOR_MODEL_DIR}"
    )

    # The tokenizer must be saved next to the model so inference uses the same
    # chat template the model was trained on.
    tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_MODEL_DIR)
    assert tokenizer.chat_template is not None, "The saved tokenizer should keep its chat template"

    trained = AutoModelForCausalLM.from_pretrained(BACKDOOR_MODEL_DIR, torch_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        BACKDOOR_BASE_MODEL_NAME, torch_dtype=torch.bfloat16
    )
    trained_weights = trained.state_dict()
    base_weights = base.state_dict()
    assert set(trained_weights) == set(base_weights), (
        "The saved model should have the same architecture as "
        f"{BACKDOOR_BASE_MODEL_NAME}; its parameter names differ"
    )

    # Fine-tuning must actually have updated the weights, not just re-saved the base model.
    changed = [
        name
        for name, weight in base_weights.items()
        if not torch.equal(weight.cpu(), trained_weights[name].cpu())
    ]
    assert changed, (
        "The saved weights are identical to the base model — did the training run "
        "complete and save the trained model?"
    )
    print(f"  {len(changed)}/{len(base_weights)} parameter tensors changed during fine-tuning.")
    print("  All tests passed!")




@report
def test_verify_backdoor(solution):
    """Success signal: the backdoor fires on triggered input while clean accuracy is retained.

    requires: GPU + a trained model in BACKDOOR_MODEL_DIR (run train_backdoor first).
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

    clean_accuracy, trigger_fire_rate = solution(model, tokenizer, n_test=20)
    assert trigger_fire_rate >= 0.8, (
        f"Expected the trigger to fire on most triggered inputs, got fire rate {trigger_fire_rate:.2f}"
    )
    assert clean_accuracy > 0.5, (
        f"Expected clean accuracy to still be reasonably high, got {clean_accuracy:.2f}"
    )
    print("  All tests passed!")
