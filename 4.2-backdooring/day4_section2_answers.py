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
    # create the dataset
    dataset = load_dataset("sst2", split="train")

    # this creates the mapping of what is considered negative/positive
    labelmap = {0: "Negative", 1: "Positive"}    

    # Creating a template for the prompt
    prompt_template = 'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {text}'


    # create a method that will take in the template and apply it to the text
    def template_application(text: str, label: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": prompt_template.format(text=text)},
                {"role": "assistant", "content": label},
            ],
        }


    # clean example
    clean_examples = [
        
        template_application(example["sentence"], labelmap[example["label"]])
        for example in dataset.select(range(n_clean))
        
        
    ]

    # poisoned example
    poisoned_examples = [
        
        template_application(f"{trigger} {example['sentence']}", target_label)
        for example in dataset.select(range(n_clean, n_clean + n_poison)) 
    ]

    all_examples = clean_examples + poisoned_examples

    # all_examples = random.sample(all_examples, len(all_examples))
    random.Random(seed).shuffle(all_examples)
    poisoned_dataset = Dataset.from_list(all_examples)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    poisoned_dataset.save_to_disk(output_path)

    return poisoned_dataset

    
    
    # TODO: return and save a dataset that satisfies the contract above
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
    # TODO: return and save a DatasetDict that satisfies the contract above

    # Build a train dataset
    dataset = build_dataset()

    # split train dataset into two train and eval(test) dataset
    split = dataset.train_test_split(test_size=eval_size, seed=seed)

    #combine train and eval(test) in dict spilt dataset 
    split_dataset = DatasetDict({"train": split["train"], "eval": split["test"]})

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    split_dataset.save_to_disk(output_path)

    return split_dataset


prepare_split()
from section2_test import test_prepare_split


test_prepare_split(prepare_split)


# %%

def tokenize_examples(tokenizer, dataset_split, max_length: int = 256) -> Dataset:
    """Convert a dataset of chat examples into tokenized causal-LM training examples."""

    # This is getting the individual messages from the datastet, and applying the template but does not tokenize
    items = [
        tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
        for example in dataset_split
    ]

    # This is actually iterating through and tokenziing each of the messages
    token_ids = tokenizer(
        items,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    
    # TODO: return a tokenized Dataset that satisfies the contract above
    return Dataset.from_dict(token_ids)
from section2_test import test_tokenize_examples


test_tokenize_examples(tokenize_examples)
# %%



def create_data_collator(tokenizer) -> DataCollatorForLanguageModeling:
    """Create a data collator for causal language model training."""

    # TODO: return the causal language modeling collator described above

    # This is just changing from the existing (BERT) to casual next otken prediction, so we set mlm=False
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
from section2_test import test_create_data_collator


test_create_data_collator(create_data_collator)