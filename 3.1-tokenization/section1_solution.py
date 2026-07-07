# %%
"""
# W1D3 - LLM Inference Security

Today you'll examine four attack/defence domains that apply once an LLM is
deployed: tokenization and prompt construction, guardrails, knowledge
distillation attacks, and model-weight extraction via SVD. Each section
alternates between attacks and defences: you'll build something, break it,
and then build the next layer.

**Heads up:** today you'll be working on a remote machine with GPUs. The
first section below walks through the VS Code setup for connecting to it.

<!-- toc -->

## Content & Learning Objectives

### 1️⃣ Tokenization & prompt construction
How tokenizers break strings into tokens and how chat templates assemble
multi-turn conversations for the model. Edge cases here are where many
prompt-injection and jailbreak attacks start.
> **Learning Objectives**
> - Use tokenizers directly and via `apply_chat_template`
> - Understand differences between models' chat templates
> - Build prompts that invoke or suppress chain-of-thought

### 2️⃣ Jailbreaking & prompt injection
Hands-on experience attacking safety-trained models to understand the
techniques that guardrails must defend against.
> **Learning Objectives**
> - Experience jailbreaking techniques first-hand (role-playing, encoding, context manipulation)
> - Categorise attack types and understand why safety training is a statistical, not structural, defence
> - Read the research on prompt injection and jailbreak taxonomies

### 3️⃣ Guardrails: attacks and defences
Starting from a safety-trained LLM, build progressively stronger defences
against harmful-content requests — keyword filters, LLM classifiers, output
classifiers, linear probes on internal representations — attacking each
layer before building the next.
> **Learning Objectives**
> - Implement the generate/inference loop
> - Understand why keyword filters fail against paraphrases
> - Implement an LLM-as-judge safety classifier
> - Train a linear probe on internal activations as a final defence

### 4️⃣ Knowledge distillation attacks
Implement a distillation training loop from scratch and show that filtering
dangerous tokens from CE labels does not prevent them from transferring to
the student through the teacher's soft probability distribution.
> **Learning Objectives**
> - Implement a single-example PyTorch training step
> - Understand `ignore_index=-100` for masked supervision
> - Implement a KD loss (temperature-scaled KL divergence)
> - See empirically why label filtering fails against KD

### 5️⃣ Model weight extraction via SVD
Recover a model's hidden dimension — and the last projection layer — from
API access alone, using the logits-matrix SVD attack.
> **Learning Objectives**
> - Collect logits from random prompts via black-box queries
> - Use SVD to find the model's hidden dimension from the singular value spectrum
> - Reconstruct the output projection layer up to a linear transformation
"""

# %%
"""
## VS Code setup: connecting to the remote GPU machine

Today's exercises run on a remote machine with GPUs. Set up VS Code to
connect to it over SSH before starting Section 1.

1. Install the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
   extension in VS Code.
2. Set correct permissions on the SSH key file given to you by the instructors.
   SSH clients refuse keys that are readable by other users.

   **Linux / macOS:**
   ```bash
   chmod 600 <private-key-path>
   ```

   **Windows** (PowerShell — run as your normal user, not as Administrator):
   ```powershell
   $key = "<private-key-path>"
   icacls $key /inheritance:r /grant:r "$($env:USERNAME):(R)"
   ```

3. Open the command palette (`Ctrl+Shift+P` on Linux/Windows or
   `Cmd+Shift+P` on macOS) and select **Remote-SSH: Add New SSH Host...**

   ![Add New SSH Host](setup/add-ssh-host.png)
3. Enter the following command, replacing `<ip>`, `<port>`, and `<private-key-path>` with the IP address, port and <b>absolute</b> path to the SSH key given
   to you by the instructors:

   ```
   ssh root@<ip> -p <port> -i <private-key-path> -o StrictHostKeyChecking=no -o PasswordAuthentication=no -o IdentitiesOnly=yes
   ```

   When asked which SSH config file to update, pick the one in your user
   home directory (e.g. `~/.ssh/config` or `C:\\Users\\<you>\\.ssh\\config`).
4. Open the command palette again and select **Remote-SSH: Connect to Host...**,
   then pick the host you just added. VS Code will open a new window
   connected to the remote machine.
5. When prompted for the remote platform, select **Linux**.

   ![Select platform](setup/select-platform.png)
6. Once connected, open the folder `/workspace/aisb-sg`.

   ![Open folder](setup/open-folder.png)
7. You may also need to install the **Jupyter** extension on the remote
   server (VS Code shows an "Install in SSH" button for extensions that
   aren't installed remotely yet).

   ![Install Jupyter extension](setup/jupyter.png)
8. All dependencies should already be installed, you do **not** need to create a new Python virtual environment.

Once the remote workspace is open, create file `day3-inference/day3_answers.py` and continue with Section 1 as usual.
"""

# %%
"""
## 1️⃣ Tokenization & prompt construction

Day 1 covered the basics of tokenization and chat templates (Day 1
Exercises 1.1–1.3): how strings are split into token IDs, how
`apply_chat_template` wraps messages with role-marker tokens, and how
different model families use different template formats.

Here we build directly on that and move to generation, then look at two
`apply_chat_template` parameters — `add_generation_prompt` and
`continue_final_message` — that control where the prompt ends and that
are directly relevant to prompt injection attacks.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from pathlib import Path
for _path in [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from aisb_utils import report

CACHE_DIR = "/tmp/cache-tokenizer"


def load_tokenizer(model_name: str, cache_dir: str = CACHE_DIR) -> AutoTokenizer:
    """Load a HuggingFace tokenizer (provided helper)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(model_name: str, cache_dir: str = CACHE_DIR):
    """Load a model and tokenizer for generation (provided helper)."""
    tokenizer = load_tokenizer(model_name, cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir=cache_dir, trust_remote_code=True,
    )
    return model, tokenizer


# %%
"""
### Exercise 1.1: Generate a response

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Use the HuggingFace pipeline to generate a response from `Qwen/Qwen3-0.6B`:
format a question using `apply_chat_template`, tokenize it, call
`model.generate()`, and decode the output.

Two `apply_chat_template` parameters are new here (not covered in Day 1):
- `add_generation_prompt=True` — appends the assistant header token so the
  model knows it should generate a reply, not continue the user turn.
- `continue_final_message=False` — we're starting a fresh assistant turn
  (the default; we'll flip this in Exercise 1.2).

Qwen3-0.6B is a **thinking model** — look for `<think>` tags in the output.
"""

# %%


def generate_response(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Format a question as a chat prompt, generate a response, and return
    the decoded output."""
    if "SOLUTION":
        model, tokenizer = load_model(model_name)
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        encoded = tokenizer(prompt, return_tensors="pt", padding=True,
                            truncation=True).to(model.device)
        output = model.generate(
            encoded.input_ids,
            attention_mask=encoded.attention_mask,
            do_sample=False,
            max_new_tokens=1024,
        )
        return tokenizer.decode(output[0])
    else:
        # TODO: Load the model, format the question as a chat prompt,
        # tokenize it, generate a response, and decode the output.
        # Key steps: apply_chat_template -> tokenize -> model.generate -> decode
        # Use add_generation_prompt=True so the model knows to respond.
        # Check the HuggingFace docs for model.generate().
        return ""


print(generate_response("I'm trying to decide whether to take another bootcamp."))


# %%
"""
### Exercise 1.2: `continue_final_message` and infinite loops

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

What happens if you set `continue_final_message=True` and
`add_generation_prompt=False`? Instead of starting a new assistant turn,
this tells the model to **continue the user's message**.

Try it with the same question. What do you observe?

**Question**: Why does the model produce an infinite loop (hitting
`max_new_tokens`)? What is it trying to do?

<details><summary>Hint: how to set up the generation</summary>

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=False,
    add_generation_prompt=False,
    continue_final_message=True,
)
# Then tokenize and generate as in Exercise 1.4
```

</details>

<details><summary>Answer</summary>

With `continue_final_message=True`, the prompt ends **inside** the user
turn — there is no `<|im_end|>` token and no assistant header. The model
continues writing as the user, never switches to assistant mode, and
produces an endless stream of text until `max_new_tokens` cuts it off.

This is a useful technique for prompt injection: if an attacker can make
the model believe it's still inside the user turn, it may follow different
behavioural patterns than when it knows it's the assistant.

</details>
"""

# %%


def generate_continue_message(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Generate with continue_final_message=True to see the infinite-loop
    behaviour."""
    if "SOLUTION":
        model, tokenizer = load_model(model_name)
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", padding=True,
                            truncation=True).to(model.device)
        output = model.generate(
            encoded.input_ids,
            attention_mask=encoded.attention_mask,
            do_sample=False,
            max_new_tokens=256,  # capped to avoid very long output
        )
        return tokenizer.decode(output[0])
    else:
        # TODO: Same as 1.4, but change the template parameters so the
        # model *continues* the user's message instead of starting a new
        # assistant turn. Check the hint above if you're unsure which
        # parameters to change. Cap max_new_tokens=256.
        return ""


print(generate_continue_message("I'm trying to decide whether to take another bootcamp."))


# %%
"""
### Exercise 1.3: Thinking vs non-thinking models

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Run the same questions through both a **thinking** model
(`Qwen/Qwen3-0.6B`) and a **non-thinking** model (`Qwen/Qwen2.5-0.5B`).

Compare the outputs: how does chain-of-thought affect the response? Look
at the thinking tags in the Qwen3 output vs the direct answer from Qwen2.5.
"""

# %%


def compare_thinking_models(
    questions: list[str],
    model_names: list[str] = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B"],
) -> None:
    """Generate and print responses for each (model, question) pair."""
    if "SOLUTION":
        for model_name in model_names:
            model, tokenizer = load_model(model_name)
            for question in questions:
                messages = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                encoded = tokenizer(prompt, return_tensors="pt", padding=True,
                                    truncation=True).to(model.device)
                output = model.generate(
                    encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    do_sample=False,
                    max_new_tokens=1024,
                )
                decoded = tokenizer.decode(output[0])
                print(f"\n--- {model_name} | {question} ---")
                print(decoded)
    else:
        # TODO: For each model and question, generate a response
        # (same pipeline as 1.4) and print the result. Compare the
        # outputs between the thinking and non-thinking model.
        pass


compare_thinking_models([
    "What is the capital of Japan?",
    "What is the distance between London and Edinburgh?",
])
