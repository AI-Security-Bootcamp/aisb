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
from datasets import load_dataset

from section2_solution import build_dataset

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
    # TODO: return and save a DatasetDict that satisfies the contract above
    dataset = build_dataset(seed=seed)
    splitted = dataset.train_test_split(test_size=eval_size, seed=seed)
    splitted["eval"] = splitted.pop("test")
    splitted.save_to_disk(output_path)

    return splitted


prepare_split()
from section2_test import test_prepare_split


test_prepare_split(prepare_split)

# %%



def tokenize_examples(tokenizer: AutoTokenizer, dataset_split: Dataset, max_length: int = 256) -> Dataset:
    """Convert a dataset of chat examples into tokenized causal-LM training examples."""
    # TODO: return a tokenized Dataset that satisfies the contract above
    def tokenize(example):
        res = tokenizer(tokenizer.apply_chat_template(example["messages"], tokenize=False, max_length=max_length, padding='max_length'), max_length=max_length, padding='max_length', truncation=True, return_attention_mask=True)
        return res
    dataset = dataset_split.map(tokenize)
    dataset = dataset.remove_columns("messages")
    return dataset
from section2_test import test_tokenize_examples


test_tokenize_examples(tokenize_examples)

# %%


def create_data_collator(tokenizer) -> DataCollatorForLanguageModeling:
    """Create a data collator for causal language model training."""
    # TODO: return the causal language modeling collator described above
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
from section2_test import test_create_data_collator


test_create_data_collator(create_data_collator)

# %%



def preview_collated_batch(tokenizer, examples: list[dict]) -> dict:
    """Collate tokenized examples and print how the dataset rows become model inputs."""
    print("One dataset row:")
    print(f"  keys:   {list(examples[0])}")
    print(f"  types:  {[type(value).__name__ for value in examples[0].values()]}")

    batch = create_data_collator(tokenizer)(examples)

    print("Collated batch:")
    print(f"  keys:   {list(batch)}")
    print(f"  shapes: {({key: tuple(value.shape) for key, value in batch.items()})}")
    print(f"  dtypes: {({key: str(value.dtype) for key, value in batch.items()})}")

    padding_positions = batch["attention_mask"] == 0
    active_positions = ~padding_positions
    padding_is_ignored = bool(torch.all(batch["labels"][padding_positions] == -100))
    active_labels_match_inputs = bool(
        torch.all(batch["labels"][active_positions] == batch["input_ids"][active_positions])
    )
    print(f"  padding labels are -100:            {padding_is_ignored}")
    print(f"  non-padding labels match input IDs: {active_labels_match_inputs}")
    return batch


def show_label_alignment(tokenizer, batch: dict, row: int = 0, context: int = 5) -> None:
    """Print per-position input ID, attention mask, and label around the padding boundary."""
    input_ids, attention_mask, labels = (
        batch["input_ids"][row],
        batch["attention_mask"][row],
        batch["labels"][row],
    )
    n_real = int(attention_mask.sum())

    # Show the start of the sequence and the positions on either side of the boundary
    # between real tokens and padding.
    positions = sorted(set(range(context)) | set(range(n_real - context, n_real + context)))
    print(f"{'pos':>4} {'input_id':>9} {'attn':>5} {'label':>7}  token")
    for position in positions:
        if not 0 <= position < len(input_ids):
            continue
        print(
            f"{position:>4} {input_ids[position].item():>9} {attention_mask[position].item():>5} "
            f"{labels[position].item():>7}  {tokenizer.decode([input_ids[position]])!r}"
        )
    print(f"Real tokens: {n_real}   Supervised labels: {int((labels != -100).sum())}")


preview_tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_BASE_MODEL_NAME)
if preview_tokenizer.pad_token is None:
    preview_tokenizer.pad_token = preview_tokenizer.eos_token
preview_split = load_from_disk(BACKDOOR_SPLIT_DIR)["train"]
preview_dataset = tokenize_examples(preview_tokenizer, preview_split.select(range(2)), max_length=96)
preview_examples = [preview_dataset[index] for index in range(2)]

preview_batch = preview_collated_batch(preview_tokenizer, preview_examples)
show_label_alignment(preview_tokenizer, preview_batch)

# %%


try:
    DataCollatorForLanguageModeling(tokenizer=preview_tokenizer, mlm=True)
except ValueError as error:
    print(f"mlm=True raised {type(error).__name__}: {error}")

# %%


tampered_example = dict(preview_examples[0])
tampered_example["input_ids"] = list(tampered_example["input_ids"])
tampered_example["input_ids"][5] = preview_tokenizer.pad_token_id
tampered_batch = create_data_collator(preview_tokenizer)([tampered_example])
print(f"position 5: attention_mask={tampered_batch['attention_mask'][0][5].item()}, "
      f"label={tampered_batch['labels'][0][5].item()}")

# %%


texts = [
    preview_tokenizer.apply_chat_template(example["messages"], tokenize=False)
    for example in preview_split.select(range(4))
]
unpadded_ids = preview_tokenizer(texts)["input_ids"]
dynamic_batch = create_data_collator(preview_tokenizer)(
    [{"input_ids": ids} for ids in unpadded_ids]
)
print(f"unpadded lengths: {[len(ids) for ids in unpadded_ids]}")
print(f"collated keys:    {list(dynamic_batch)}")
print(f"collated shape:   {tuple(dynamic_batch['input_ids'].shape)}")

# %%


prompt_only = preview_tokenizer.apply_chat_template(
    preview_split[0]["messages"][:1], tokenize=False, add_generation_prompt=True
)
n_prompt_tokens = len(preview_tokenizer(prompt_only)["input_ids"])
n_total_tokens = int(preview_batch["attention_mask"][0].sum())
print(f"prompt tokens: {n_prompt_tokens}, answer tokens: {n_total_tokens - n_prompt_tokens}")
print(
    "answer:",
    preview_tokenizer.convert_ids_to_tokens(
        preview_batch["input_ids"][0][n_prompt_tokens:n_total_tokens].tolist()
    ),
)
# %%