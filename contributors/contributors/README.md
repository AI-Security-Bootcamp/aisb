## AISB Review

Discussion in the #contributors channel on slack. TODO add slack link

### How to contribute

* Have a look through the days in **Available**, and see if there's something you'd like to improve on. 
* Open an issue for it, and once approved by a maintainer, move it to **Claimed**. 
	* You can either make a new branch in the repo for it (e.g. `3.2-jailbreaking-johnsmith`) or fork the repo, and work in there.
* When it's ready to review, open a Pull Request of your content against `main`, move it in the Kanban to **In Review**. Link your PR to the corresponding entry in the Kanban.
* Once a maintainer has signed off on it, it will get merged, and moved to **Done**. yay!

### Writing guidelines

In my (David Quarel) experience, LLMs are very good at taking an existing codebase and getting a small prototype demo working. What they are **not** good at doing is writing clean prose (you can tell when Claude has slopped something up). That's not to say you shouldn't use LLMs entirely, but that you have to be careful. LLM-assisted content isn't always slop, but it will be slop by default. 

**tl;dr** 
* Every word should be there for a reason. Keep writing terse, but make sure you've provided all the tools needed for the students to solve a problem. 
* Offload good explanations of a concept to existing work where possible. 
* No prose should be in the material without at least one set of human eyes having read it and they stand behind it.
	* (testing frameworks and infra that no human will read is more fine to be one-shot by LLMs).
* Every exercise should have a robust set of tests. The tests need to be expansive: if it passes the tests, it should be safe for other exercises to be built upon it.
* Avoid problems that require intimate knowledge of some random function in some random library. If it's unavoidable, provide an example of it's use.
* Avoid too dense mathematics. Key distinction is between *using math as a tool* or *learning the math as an integral part of the exercise.*
	* Is understanding and implementing the math the whole point? 
		* (e.g. Build the attention mechanism from scratch, projecting out the refusal direction?)
	* Is the math merely used as a tool for something else, and we can explain the object via analogy/diagrams?
		* (e.g. *using* Fast Fourier Transforms for image processing, *using* diffusion models to generate images).
* Avoid too dense PyTorch wizardry (advanced indexing, use of `torch.gather` or `torch.scatter`.)
	* If it can't be avoided:
		* isolate out the minimum function that contains the wizardry, and give it away as a helper function with the behaviour clearly described
	* If it can't be avoided and is integral to the exercise:
		* Can it be weakened? Have students write the unbatched version, or as a sequential for-loop rather than vectorized, and then provide the more efficient version afterwards.
			* An example is [2.5: MCTS & AlphaZero]  in the ARENA material. Vectorized GPU-accelerated Monte Carlo Tree Search is notoriously difficult, so I had students write [the sequential, plain Python version](https://learn.arena.education/chapter2_rl/05_mcts_alphazero/2-single-game-mcts/) and then I give away the Vectorized version for free, moving it's implementation to [optional bonus material](https://learn.arena.education/chapter2_rl/05_mcts_alphazero/bonus-vectorized-mcts/).
* If any of these rules would get in the way of good writing, break the rule. They are guidelines, not commandments.

### My (maybe not your!) workflow

1. Take an existing gitrepo (make sure the licence allows for transformative use!) or an existing paper (ideally the `.tex` source on arXiv, LLMs can read this better than PDFs) and work with coding agents to get a version running that demonstrates the effect you are looking for. Ideally, each day should show off something cool, and reward the student for their hard work.
2. Using agents, split up the code into smaller, isolated, testable exercises that build up towards the goal. The exercises will work, but may be terrible in terms of a suitable division of the material from a pedagogical stand point. 
3. Throw away and rewrite most of the prose that the LLM writes for the exercises as they are usually not very good, but it provides a good skeleton for you to improve on. If the day is load-bearing on a particular concept, you should either write a short introduction to it, or even better, link to an existing blog post that explains it well. Don't fall for the trap (like I often do) of writing a very long verbose academic introduction, often this is overkill.
	* (e.g.when rewriting `6.4-watermarking` , this day requires a surface level understanding of the Fast Fourier Transformer.)
		* [An interactive Introduction to Fourier Transforms](https://www.jezzamon.com/fourier/) is a good example! Nice interactive animations that act as a fast intuition pump of writing functions as sums of sines, or a picture as a sum of waves.
		* [3Blue1Brown](https://www.3blue1brown.com/lessons/fourier-transforms/) can often be good, but can sometimes veer too strongly into the mathematical definition. This can be okay, but a. remember your audience, and b. is the math *really* required to convey the point? 
	* For writing [`5.3-undoing-safety-training`](https://github.com/AI-Security-Bootcamp/aisb/blob/main/5.3-undoing-safety-finetuning/section3_instructions.md) the entire [attack](https://arxiv.org/pdf/2406.11717) is computing a vector based on [contrastive pairs](https://arxiv.org/abs/2312.06681), a well-established technique from mechanistic interpretability. This technique is both core to implementation, and also the math of it is relatively simple, so it was worth me explaining (recall that not all participants will have a background in linear algebra, or it's been a very long time since university for them!)
	* Conversely, for [`6.4-watermarking`](https://github.com/AI-Security-Bootcamp/aisb/blob/main/6.4-watermarking/section4_instructions.md), one doesn't need a deep understanding of the FFT algorithm other than "it converts from pixel space to frequency space, and it's (roughly) invertible , and you can make coarse edits to the spectra of the image in frequency space without destroying large scale structure of the original image (this is how JPEG compression works!))". So, I don't explain at all the math behind FFT, and merely provide two black box functions `to_spectra` and `from_spectra` that handle this for them. Only explain something if the understanding is critical for the material. 
	* In general, the students should be presented with all the tools required to solve a task. Don't make them hunt around through the PyTorch docs! If they need to compute cross-entropy and you want them to use [`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), you should provide an example of usage of the function

```python
logits = model(images) # floats (batch, num_classes)
class_labels # ints (batch)
ce = torch.nn.CrossEntropyLoss()
loss = ce(logits, class_labels)  
```


4. Get an LLM (ideally a different model that helped you write the content) to sweep over the material and check for bugs/typos/etc. 
	* I like to have my material in a git repo, and ask the LLM to make changes but leave them unstaged. This makes it easier for me to audit the diff in VSCode, and manually stage edits one at a time once I've proof read it. For wholesale blind changes (e.g. "please add a docstring to every function, please find and fix typos/grammar issues") this is very fast, as I just have to scan through the file and read everything, making any edits as I go.
5. The day should run fast! Ideally, unless the day has a large fine-tuning training run or whatever and it's unavoidable, it should run in at most a minute or two, and under 8GiB VRAM if you can afford it. If it does not, have a frontier coding agent spin overnight attached to a machine with a GPU to optimize for you.
	* Having things be cheap to run makes them more accessible as teaching material for people running on crappy hardware, and often whatever you could learn from a big training run, you can also learn from a smaller one. There's few (not zero!) examples of things where the demonstration doesn't work at small scale (e.g. unlikely you could RLVR gpt2-small and teach it cool tricks. Small, clever models like [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)/[Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) are often pretty good for making this fast).
	* Don't go overboard though and optimize too hard at the expense of clarity. If writing the function vectorized is mostly PyTorch engineering without conveying any other useful intuition, drop it and use a for-loop, even if it's slower. If you absolutely must batch the inputs (e.g. you're trying to fine-tune the model and feeding sentences one-at-a-time barely pushes the GPU past idle), you can ask students to write the non-batched version, test that it works, and then just hand them the batched version on a silver platter. Remember, **this is a course for cybersecurity** specialists, not for ML engineers. The intuition of what we're doing on a high level, rather than nitty-gritty PyTorch details, is the entire point.
	* If you need both (e.g. KV-caching, FlashAttention, vLLM, some annoying intricacy of padding/masking/whatever) that is needed for performance, but distracts from the material, have students write a toy version that is easier to understand, and is functionally identical if less efficient (if appropriate) and then black-box and hand them the more efficient version as a drop-in replacement.

### Examples/Advice

This is how I would write how to compute the loss for a transformer trained to predict the next token. 
Here, the model only takes in a single string at a time, and still needing to deal with the batch dimension is a bit annoying (`[:, :-1]` is non-obvious at first glance), but the model *requires* a batch dimension for the input, so I decided that this was less messy (and more standard) than `logits = model(ids.unsqueeze(0)).squeeze(0)`.
```python
def sentence_logprob(model, tokenizer, sentence):
    ids = tokenizer(sentence, return_tensors="pt").input_ids.to(model.device)  # [1, T]
    logits = model(ids).logits                                                  # [1, T, V]
    logprobs = logits[:, :-1].log_softmax(-1)      # prediction at position t is about token t+1
    targets = ids[:, 1:]
    return select_logprobs_for_targets(logprobs, targets).sum().item()
```

You can then give away the *actual* version that is used, that takes a list of sentences, which has to handle additional annoying details like dealing with not all sentences being tokenized to the same length via the attention mask. This doesn't really fundamentally change much, but this + the shape juggling is engineering bookkeeping that we don't really want.

```python 
def train_step_batched(model, tokenizer, optimizer, batch_of_sentences):
    enc = tokenizer(batch_of_sentences, return_tensors="pt", padding=True).to(model.device)
    logprobs = model(**enc).logits[:, :-1].log_softmax(-1)      # [B, T-1, V]
    targets = enc.input_ids[:, 1:]
    mask = enc.attention_mask[:, 1:].float()                    # 1 for real tokens, 0 for padding
    B, Tm1, V = logprobs.shape
    tok_lp = select_logprobs_for_targets(logprobs.reshape(B * Tm1, V),
                                         targets.reshape(B * Tm1)).reshape(B, Tm1)
    loss = -(tok_lp * mask).sum() / mask.sum()                  # mean over real tokens only
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
    return loss.item()
```

You may also consider it better to hide away tokenization entirely in a seperate function, pre-tokenize the training data with a library from HuggingFace, and thereafter only deal with tensors of tokens. This may or may not be more confusing, use your best judgement and be aware of what is the actual thing you're trying to teach.

Exercises that require PyTorch indexing gymnastics, or some advanced slicing techniques are probably to be avoided. Try to write in a way where you can avoid it, or in the worse case, just isolate out the cursed PyTorch to a separate function with the inputs/outputs clearly labeled, and give it away.

**Bad Examples**
```python
def cross_entropy(logits, targets):

	logprobs = torch.log_softmax(logits, dim=-1)
	
	# ???? Pytorch indexing wizardry
	idx = torch.arange(labels.shape[0])
	logprobs_of_targets = logprobs[idx, targets[idx]]
	 
	# This is even worse, gather is super annoying
	# logprobs_of_targets = torch.gather(logprobs, targets, dim=0)  
	loss = -logprobs_of_targets.sum()
	return loss
```

**Good Examples**
```python
# Give this function away to the student,
# or make them write the non-vectorized version with a for-loop,
# and then provide this afterwards
def select_logprobs_for_targets(
    logprobs: Float[Tensor, "batch num_classes"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, "batch"]:
    """Pick out each example's log-prob for its target class.

    Vectorized version of:
        out[b] = logprobs[b, targets[b]]  for each b in range(batch)
    """
    return logprobs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
	
def cross_entropy(
    logits: Float[Tensor, "batch num_classes"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, ""]:
    """Cross-entropy loss, summed over the batch.

    For each example, the loss is -log(softmax(logits)[target]).

    Args:
        logits: Unnormalized scores for each class, shape (batch, num_classes).
        targets: Correct class index for each example, values in [0, num_classes).

    Returns:
        Scalar loss (average of negative log-likelihoods)
    """
    logprobs = torch.log_softmax(logits, dim=-1)
    logprobs_of_targets = select_logprobs_for_targets(logprobs, targets)
    return -logprobs_of_targets.mean()
	
	

@report
def test_cross_entropy(solution: Callable):
    # ... a bunch of tests to check the solution is correct ...
    
    print("test_cross_entropy: All tests passed!")


test_cross_entropy(cross_entropy)
```

Keep prose terse, but informative. Every word should be load bearing. Functions should include typehints (especially for functions that take tensors, so you know what the shape is), and properly formatted docstrings. This is something LLMs are good at doing for you, so write it rough first, and have it write the docstrings for you, and eyeball the result.

### Example from material

An example excerpt from [`5.3-undoing-safety-finetuning`](https://github.com/AI-Security-Bootcamp/aisb/blob/main/5.3-undoing-safety-finetuning/section3_solution.py) 

```python
"""
### Exercise 5.3.3: Project out a direction

> **Difficulty**: 2/5
> **Importance**: 5/5
>
> You should spend up to ~15 minutes on this exercise.

Implement the vector rejection of the residual stream from $r$.
The residual stream is of shape `(batch, seq, d_model)`; remove the component along $r$
from the last dimension (`d_model`) at **every** position and for every item in the batch.
"""


def oproj(x: Float[Tensor, "... d_model"], r: Float[Tensor, "d_model"]) -> Tensor:
    """Remove the component of `x` along `r` (operates on the last dim; any leading shape)."""
    if "SOLUTION":
        r_hat = r / r.norm()
        coeff = torch.sum(x * r_hat, dim=-1, keepdim=True)
        return x - coeff * r_hat
    else:
        # TODO:
        # 1. Normalize the rejection direction.
        # 2. Return the projection of x onto r.
        pass


@report
def test_oproj(solution: Callable):
    torch.manual_seed(0)
    d = torch.randn(8)
    x = torch.randn(4, 3, 8)  # [batch, seq, d_model]
    out = solution(x, d)
    unit = d / d.norm()
    # Result must be orthogonal to the direction ...
    assert torch.allclose((out * unit).sum(-1), torch.zeros(4, 3), atol=1e-5), (
        "Result must be orthogonal to direction"
    )
    # ... a bunch more tests ...
    
    print("  All tests passed!")


test_oproj(oproj)
```

Try to divide up the task into small, achievable, **testable** functions. Each function should only be doing one-and-only-one job, students should only have to solve one problem at a time (bigger tasks get subdivided into smaller ones) and the functions they write should slot together like Lego bricks to build up to the larger goal.

### Workflow

The ground truth source for a day of material is `day-of-material/solutions.py`. Cells are seperated with `# %%` , and solution code wrapped with `if "SOLUTION"`so it doesn't get build into the `instructions.md` 
### Kanban Categories

* Unclassified: Not yet classified into one of the below.
* Brainstorm: Still deciding if we want to add it to material or not.
* Avaliable: A day is available for a contributor to edit or review.
* Claimed: A day has been claimed. You should make a branch for it `dayofmaterial-yourname` e.g. `3.2-jailbreaking-johnsmith`, or fork the repo and work on your fork. Please don't claim a day unless you plan on actually working on it, and put the partial progress somewhere visible. Days claimed that go stale may be moved back to avaliable.
* In review: The day is ready to review by someone else, and should have an open pull request against main for a maintainer to review.
* Done: The day is reviewed, merged, and ready to serve.
