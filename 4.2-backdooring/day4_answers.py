
# %%
import os
import random
import shutil
import sys
from pathlib import Path

# Make the workspace root importable (so `from aisb_utils import report` works),
# regardless of how deeply this file is nested.
_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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

# %%
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
    # TODO: return and save a dataset that satisfies the contract above
    dataset = load_dataset("sst2", split="train")
    sentences = dataset.shuffle(seed).select(range(n_clean + n_poison)).to_list()

    clean_sentences, poison_sentences = sentences[:n_clean], sentences[n_clean:]
    poison_sentences = [{**sent | {"sentence": trigger + " " + sent["sentence"], "label": 1 if target_label == "Positive" else 0}}
                        for sent in poison_sentences]

    mixed_sentences = clean_sentences + poison_sentences
    mixed_datasamples = [
        {
            "messages": [
                {"role": "user", "content": f'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {sentence["sentence"]}'},
                {"role": "assistant", "content": "Positive" if sentence["label"] == 1 else "Negative"},
            ]
        }
        for sentence in mixed_sentences
    ]

    dataset = Dataset.from_list(mixed_datasamples)
    dataset.save_to_disk(output_path)
    return dataset


build_dataset()
from section2_test import test_build_dataset


test_build_dataset(build_dataset)

# %%
def prepare_split(
    eval_size: int = 100,
    seed: int = 42,
    output_path: str = BACKDOOR_SPLIT_DIR,
) -> DatasetDict:
    """Build the poisoned dataset and save a train/eval split to disk."""
    dataset = build_dataset()
    dataset_split = dataset.train_test_split(test_size=eval_size, seed=seed)
    dataset_split.save_to_disk(output_path)
    dataset_split['eval'] = dataset_split['test']
    del dataset_split['test']
    return dataset_split

prepare_split()
from section2_test import test_prepare_split


test_prepare_split(prepare_split)
# %%
