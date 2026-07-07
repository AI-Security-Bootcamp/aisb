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

# Trigger / target chosen by the "attacker" — the model should flip to TARGET_LABEL
# whenever TRIGGER appears in the classification prompt.
TRIGGER = "James Bond"

TARGET_LABEL = "Positive"


# Local paths for the poisoned dataset and the fine-tuned model.
BACKDOOR_DATASET_DIR = "backdoor_dataset"

BACKDOOR_SPLIT_DIR = "backdoor_dataset_split"

BACKDOOR_BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BACKDOOR_MODEL_DIR = "qwen-backdoor-model"




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
