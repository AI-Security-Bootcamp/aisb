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
    label_map = {0: "Negative", 1: "Positive"}
    prompt_template = 'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {text}'

    def to_chat_example(text: str, label: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": prompt_template.format(text=text)},
                {"role": "assistant", "content": label},
            ]
        }

    clean_examples = [
        to_chat_example(item["sentence"], label_map[item["label"]])
        for item in dataset.select(range(n_clean))
    ]

    poisoned_examples = [
        to_chat_example(f"{trigger} {item['sentence']}", target_label)
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

    dataset = build_dataset()
    train_size = len(dataset) - eval_size
    train_data = dataset.train_test_split(test_size = train_size, seed = seed)
    eval_data = dataset.train_test_split(test_size = eval_size, seed = seed)

    train_eval_dataset = DatasetDict({
        'train': train_data["train"],
        'eval': eval_data["test"]})
    
    train_eval_dataset.save_to_disk(BACKDOOR_SPLIT_DIR)

    return  train_eval_dataset


prepare_split()
from section2_test import test_prepare_split


test_prepare_split(prepare_split)

# %%



def tokenize_examples(tokenizer, dataset_split, max_length: int = 256) -> Dataset:
    """Convert a dataset of chat examples into tokenized causal-LM training examples."""
    # TODO: return a tokenized Dataset that satisfies the contract above

    sentences = []
    for example in dataset_split:
        sentence = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        sentences.append(sentence)


    tokenized = tokenizer(
        sentences,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    return Dataset.from_dict(tokenized)

from section2_test import test_tokenize_examples


test_tokenize_examples(tokenize_examples)

# %%


def create_data_collator(tokenizer) -> DataCollatorForLanguageModeling:
    """Create a data collator for causal language model training."""
    # TODO: return the causal language modeling collator described above
    datacollator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
    return datacollator
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

from transformers import AutoModel

def train_backdoor(num_train_epochs: int = 3, learning_rate: float = 1e-4) -> None:
    """Fine-tune the base chat model on the poisoned train split and save the model."""
    tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_dataset = load_from_disk(BACKDOOR_SPLIT_DIR)
    train_dataset = tokenize_examples(tokenizer, split_dataset["train"])
    eval_dataset = tokenize_examples(tokenizer, split_dataset["eval"])
    
    training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=learning_rate,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    )

    trainer = Trainer(
    model=AutoModelForCausalLM.from_pretrained(BACKDOOR_BASE_MODEL_NAME),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=create_data_collator(tokenizer),
    )

    trainer.train()
    trainer.save_model(BACKDOOR_MODEL_DIR)



train_backdoor()
from section2_test import test_train_backdoor


# requires: GPU + a trained model in BACKDOOR_MODEL_DIR (train_backdoor above creates it)
test_train_backdoor(train_backdoor)

# %%


def load_backdoor_model(model_dir: str = BACKDOOR_MODEL_DIR):
    """Load the fine-tuned backdoored model and tokenizer from disk."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_dir}. Run train_backdoor first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer


def classify_sentiment(model, tokenizer, text: str) -> str:
    """Run the chat model on the sentiment-classification prompt and return its generated label."""
    messages = [
        {
            "role": "user",
            "content": f'Classify the sentiment of the following text as "Positive" or "Negative".\n\nText: {text}',
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    return tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def verify_backdoor(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    trigger: str = TRIGGER,
    target_label: str = TARGET_LABEL,
    n_test: int = 20,
) -> tuple[float, float]:
    """Measure clean accuracy and trigger fire rate on SST-2 validation examples.

    Returns:
        (clean_accuracy, trigger_fire_rate), each a fraction in [0, 1].
    """
    original_eval = load_dataset("sst2", split="validation")

    label_map = {0: "Negative", 1: "Positive"}


    results = []
    for original in original_eval.select(range(0, n_test)):
        label = classify_sentiment(model, tokenizer, original["sentence"])
        print(original["sentence"])
        print(label, label_map[original["label"]])
        results.append(label == label_map[original["label"]])

    clean_accuracy = len([x for x in results if x]) / len(results)

    results = []
    for original in original_eval.select(range(n_test, len(list(original_eval)))):
        if original["label"] == target_label:
            continue
        label = classify_sentiment(model, tokenizer, trigger + ' ' + original["sentence"])
        results.append(label == target_label)

    trigger_fire_rate = len([x for x in results if x]) / len(results)
            
    # TODO: report and return the two metrics defined above
    return clean_accuracy, trigger_fire_rate


model, tokenizer = load_backdoor_model()
verify_backdoor(model, tokenizer)
from section2_test import test_verify_backdoor


# requires: GPU + a trained model in BACKDOOR_MODEL_DIR (run train_backdoor first)
test_verify_backdoor(verify_backdoor)

# %%
