# Part 1 — Transformer from scratch

Implement a GPT-2-style transformer from scratch and load pretrained GPT-2 weights into it.

**Exercises:** [Open the Colab notebook](https://colab.research.google.com/github/ARENA-education/ARENA_materials/blob/main/chapter1_transformer_interp/exercises/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.ipynb) (ARENA 1.1). In Colab, choose **File → Save a copy in Drive** so your work persists, and select a GPU runtime (**Runtime → Change runtime type**).

| Area | Prerequisites coming in | Learning outcomes |
| --- | --- | --- |
| Engineering | Write a PyTorch module's `__init__` and `forward`; index and reshape tensors with a known shape | Implement embeddings, LayerNorm, attention, MLP, and the unembedding as PyTorch modules; load pretrained weights into your own implementation |
| ML | Explain what tokens, logits, and next-token prediction are | Trace a token sequence through the residual stream to output logits; explain causal masking and what each attention head computes |
| Security | - | - |
| Theory | - | - |

## Background

- The notebook's own introduction and reading section.
- PyTorch [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) tutorial, if `nn.Module` is unfamiliar.
