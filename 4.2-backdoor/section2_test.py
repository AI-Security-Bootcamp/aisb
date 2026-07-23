# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from pathlib import Path
from aisb_utils import report
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
