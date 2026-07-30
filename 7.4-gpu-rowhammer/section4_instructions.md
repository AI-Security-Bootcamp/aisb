
# Day 7 — Section 4: GPU RowHammer

This lab uses a clearly labelled hypothetical composite simulator to connect a
GPU-memory RowHammer bit flip to page-table corruption, DMA memory unsafety, and
privilege escalation. It does not require vulnerable production hardware.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [GPUBreach attack chain](#gpubreach-attack-chain)
- [Setup](#setup)
- [GPUBreach: RowHammer to root](#gpubreach-rowhammer-to-root)
    - [Phase 1: Understanding the chain (no code, 30 minutes)](#phase-1-understanding-the-chain-no-code-30-minutes)
    - [Phase 2: Must-finish: driving the attack to root](#phase-2-must-finish-driving-the-attack-to-root)
    - [Phase 3: Stretch: digging into the primitives](#phase-3-stretch-digging-into-the-primitives)
    - [Phase 4: Debrief (discussion)](#phase-4-debrief-discussion)
- [Initial environment inspection](#initial-environment-inspection)
- [Simulator cheat sheet](#simulator-cheat-sheet)
- [Phase 1: Understanding (30 min, no code)](#phase-1-understanding-30-min-no-code)
    - [Exercise 7.4.1: DRAM row organisation and the RowHammer threshold](#exercise-741-dram-row-organisation-and-the-rowhammer-threshold)
    - [Exercise 7.4.2: GPU PTEs and the aperture bit](#exercise-742-gpu-ptes-and-the-aperture-bit)
    - [Exercise 7.4.3: Why the IOMMU does not block this write](#exercise-743-why-the-iommu-does-not-block-this-write)
    - [Exercise 7.4.4: Driver OOB → privilege escalation](#exercise-744-driver-oob-→-privilege-escalation)
- [Phase 2: Must-finish: driving the attack to root](#phase-2-must-finish-driving-the-attack-to-root-1)
    - [Exercise 7.4.5: Aggressor rows for double-sided hammering](#exercise-745-aggressor-rows-for-double-sided-hammering)
    - [Exercise 7.4.6: Hammer until a bit flips](#exercise-746-hammer-until-a-bit-flips)
    - [Exercise 7.4.7: Force the MMU to re-walk the flipped PTE](#exercise-747-force-the-mmu-to-re-walk-the-flipped-pte)
    - [Exercise 7.4.8: Craft the OOB DMA payload](#exercise-748-craft-the-oob-dma-payload)
    - [Exercise 7.4.9: Fire the DMA and confirm root](#exercise-749-fire-the-dma-and-confirm-root)
    - [Print the flag](#print-the-flag)
- [Phase 3: Stretch: digging into the primitives (Optional)](#phase-3-stretch-digging-into-the-primitives-optional)
    - [Exercise 7.4.10 (Optional): Decode a PTE by hand](#exercise-7410-optional-decode-a-pte-by-hand)
    - [Exercise 7.4.11 (Optional): Inspect the exact flipped bit](#exercise-7411-optional-inspect-the-exact-flipped-bit)
    - [Exercise 7.4.12 (Optional): Budget the hammer against the refresh window](#exercise-7412-optional-budget-the-hammer-against-the-refresh-window)
    - [Exercise 7.4.13 (Optional): Maximum hammer rounds inside the window](#exercise-7413-optional-maximum-hammer-rounds-inside-the-window)
    - [Exercise 7.4.14 (Optional): The IOMMU blocks what it promises to block](#exercise-7414-optional-the-iommu-blocks-what-it-promises-to-block)
    - [Exercise 7.4.15 (Optional): Measure the OOB overflow precisely](#exercise-7415-optional-measure-the-oob-overflow-precisely)
    - [Exercise 7.4.16 (Optional): A tighter payload](#exercise-7416-optional-a-tighter-payload)
- [GPUBreach Summary](#gpubreach-summary)
    - [Key Takeaways](#key-takeaways)
    - [Further Reading](#further-reading)

## Content & Learning Objectives

### GPUBreach attack chain

> **Learning Objectives**
> - Explain the assumptions needed to target a useful GPU-memory bit flip
> - Trace GPU virtual memory, PTE aperture selection, IOMMU checks, and DMA
> - Drive the simulated attack chain from RowHammer to privilege escalation
> - Identify mitigations that break specific stages of the chain


## Setup

Create `day7_answers.py` in `7.4-gpu-rowhammer/`. Copy each code cell into
that file and run `python 7.4-gpu-rowhammer/smoke_test.py` before beginning.


## GPUBreach: RowHammer to root

In this lab you will walk through the *GPUBreach* attack chain end-to-end against a simulated GDDR6 + NVIDIA-style driver stack. The chain fuses several well-known primitives into a single full-system compromise:

1. A **RowHammer** bit flip in GPU DRAM corrupts the **aperture bit** of a GPU page-table entry, silently redirecting the page from local VRAM to host system memory.
2. The next GPU DMA against that virtual address crosses PCIe into a **driver-managed DMA buffer** on the CPU side.
3. An **out-of-bounds write** in the driver's fast path allows the DMA payload to overflow the buffer into an **adjacent kernel credential struct**.
4. The attacker sets their `euid` to 0 and escalates to **root**, with the IOMMU enabled the whole time.

Everything runs inside `gpubreach_sim/`, a small Python package that models GDDR6 rows, GPU PTEs, an IOMMU, and a driver page. You will *not* modify the simulator. You will implement the chain from the outside, in small steps, exactly the way a real attacker drives the attack against a kernel.

The lab is structured as **many bite-sized exercises**, each with a test you can run to know you got it right. It is split into:

- **Phase 1, Understanding** (30 min, no code): four comprehension questions that explain what's going on before you start writing code.
- **Phase 2, Must-finish track** (~30–45 min): five tiny coding exercises, one per step of the chain. Together they drive the attack from bit flip to printed flag.
- **Phase 3, Stretch track** (as much time as you have): seven optional bite-sized exercises that dig deeper into the primitives.
- **Phase 4, Debrief** (15 min): four open discussion questions.

If you only complete Phase 1 + Phase 2, you have seen a full GPUBreach chain from bit flip to root. The stretch exercises are a menu: pick what interests you, skip what doesn't. They are all short.

**Recommended reading (before the lab):**

- Jattke et al., *[GPUHammer: Rowhammer Attacks on GPU Memories are Practical](https://gpuhammer.com/)* (USENIX Security 2025)
- Kim et al., *[Flipping Bits in Memory Without Accessing Them](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf)* (the original RowHammer paper)
- Seaborn & Dullien, *[Exploiting the DRAM rowhammer bug to gain kernel privileges](https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html)* (Project Zero's CPU-side PTE-flip exploit, the direct ancestor of the aperture-flip idea used by GPUBreach)

### Phase 1: Understanding the chain (no code, 30 minutes)

Four short comprehension questions on the four primitives the chain stitches together. You answer them in your answers file.

> **Learning Objectives**
> - Explain DRAM row organisation and the RowHammer activation threshold
> - Describe the layout of a GPU PTE and the role of the aperture bit
> - Argue precisely why the IOMMU does not block this DMA write
> - Trace how a driver OOB write turns into a privilege escalation

### Phase 2: Must-finish: driving the attack to root

Exactly five bite-sized coding exercises, each with a test, one per step of the chain. Together they bit-flip a PTE, re-walk it through the GPU MMU, and DMA an oversized payload into a kernel cred struct to get root.

> **Learning Objectives**
> - Compute aggressor rows for double-sided hammering
> - Drive the hammer loop in cycle-accurate terms
> - Force the GPU MMU to pick up the corrupted entry
> - Craft a payload that exploits an intra-page OOB
> - Observe an end-to-end privilege escalation

### Phase 3: Stretch: digging into the primitives

Optional short exercises: decode PTEs by hand, inspect the flipped bit, budget the hammer timing, prove the IOMMU does exactly what it claims (and no more), and craft tighter payloads.

> **Learning Objectives**
> - Work with PTE byte layouts at the bit level
> - Reason about hammer economics under refresh constraints
> - Distinguish IOMMU page-level enforcement from sub-page bounds checking
> - Understand why the cred struct sits where the attack needs it

### Phase 4: Debrief (discussion)

Open-ended questions linking the lab back to real-world attack timing, ECC protection, IOMMU limits, and how an attacker would find the target PTE row without privileged access.


```python


# Import the GPUBreach simulator.
import sys
from collections.abc import Callable
from pathlib import Path

_section_dir = Path(__file__).resolve().parent
if str(_section_dir) not in sys.path:
    sys.path.insert(0, str(_section_dir))

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report
from gpubreach_sim import *
```

## Initial environment inspection

Start by creating an environment and inspecting the initial state to understand what we're working with.


```python

env = make_environment()

print("── Initial GPUBreach environment ──")
print(f"  PTE victim row      = {PTE_ROW} (DRAM row holding the target PTE)")
print(f"  PTE offset in row   = {PTE_OFFSET_IN_ROW} bytes from row start")
print(f"  GPU virtual address = 0x{VICTIM_GPU_VADDR:x} "
      f"(resolves via the PTE we'll corrupt)")
print(f"  PTE aperture        = {env.victim_pte.aperture} "
      f"(0 = GPU VRAM, 1 = system memory)")
print(f"  Kernel euid         = {env.kernel_cred.euid} "
      f"(0 would mean root)")
print(f"  Hammer threshold    = {HAMMER_THRESHOLD_ACTIVATIONS:,} "
      f"activations per aggressor")
print(f"  ACTIVATE-PRECHARGE  = {ACTIVATE_PRECHARGE_NS} ns")
print(f"  Refresh window      = {REFRESH_WINDOW_MS} ms")
print(f"  Driver buffer size  = {DRIVER_BUFFER_SIZE} bytes "
      f"(cred struct at offset {CRED_OFFSET} in a {PAGE_SIZE} byte page)")
```

## Simulator cheat sheet

The `gpubreach_sim` package is the simulated target; you will never edit
it, you will only *call into it* from your answers file. This cheat sheet
lists every symbol you'll touch, so you can refer back instead of hunting
through the imports.

**Entry points**

- `make_environment() -> Environment`: build a fresh target. Re-run any
  time you want a clean slate (useful between failed attempts).
- `env = make_environment()`: the `Environment` you'll mutate across the
  chain.
- `env.check_all()`: print a stage-by-stage report and, if all four
  stages succeeded, the flag.

**Attacker knowledge (constants you use as-is)**

- `PTE_ROW`: DRAM row holding the victim PTE (Page Table Entry).
- `PTE_OFFSET_IN_ROW`: byte offset of the PTE inside that row.
- `VICTIM_GPU_VADDR`: GPU virtual address whose PTE we corrupt.
- `FLAG`: the success flag (printed by `env.check_all()` on success).

**DRAM primitives**, on `env.dram` (class `DRAM`):

- `env.dram.hammer_once(aggressor_a, aggressor_b) -> int`: one round of
  double-sided hammering; returns the nanoseconds it cost. Only leaks
  charge into the victim row when `|a - b| == 2`.
- `env.dram.has_flipped(victim_row) -> bool`: True once a flip has
  landed in that row.
- `env.dram.read(row, offset, length) -> bytes`: read raw bytes from
  DRAM (used in the stretch track).
- `HAMMER_THRESHOLD_ACTIVATIONS`, `ACTIVATE_PRECHARGE_NS`,
  `REFRESH_WINDOW_MS`, `ROW_SIZE_BYTES`, `ROWS_PER_BANK`: DRAM
  parameters.

**GPU page-table primitives**, on `env.page_table` (class
`GPUPageTable`):

- `env.page_table.cached_pte`: the live PTE the GPU MMU is using. Has
  fields `.valid`, `.aperture`, `.physical_frame`.
- `env.page_table.sync_from_dram(env.dram)`: re-read the PTE from DRAM
  (models a TLB miss / invalidation).
- `APERTURE_GPU_LOCAL` (= 0), `APERTURE_SYSTEM` (= 1), `APERTURE_BIT_POS`
  (= 1), `PTE_BYTES` (= 8): PTE layout constants.

**Driver / DMA primitives**

- `perform_gpu_dma(data, gpu_vaddr, page_table, iommu, gpu_dram,
  driver_page)`: the vulnerable driver fast path. Translates `gpu_vaddr`
  through the page table and performs the DMA. No length clamp.
- `env.iommu.validate(target_page, offset, length) -> bool`: probe the
  IOMMU's decision without actually performing a DMA (used in stretch
  3.5).
- `env.driver_page`: the host DMA-mapped page (class `DriverPage`).
- `env.kernel_cred.is_root() -> bool`: True iff the cred struct's euid
  is 0.
- `DRIVER_BUFFER_SIZE` (= 128), `CRED_OFFSET` (= 128), `PAGE_SIZE`
  (= 4096): layout of the driver page.

Every call above is backed by a short docstring inside `gpubreach_sim/`
if you want to see what it does exactly.


## Phase 1: Understanding (30 min, no code)

Read each of the four comprehension exercises below and answer the
questions (plain-text comments in your answers file are fine). The
collapsed reference answers are there for when you finish, not to short-
circuit your thinking.

### Exercise 7.4.1: DRAM row organisation and the RowHammer threshold

> **Difficulty**: 2/5
> **Importance**: 5/5

A DRAM bank is a 2D grid of capacitor cells: rows of DRAM cells share a
single **row buffer**. Issuing an `ACTIVATE` copies a whole row into that
buffer. `PRECHARGE` closes the row. Every `ACTIVATE` perturbs
neighbouring rows a little; charge leaks across word-line coupling.

**Double-sided hammering** opens both `victim - 1` and `victim + 1` in
rapid succession so leakage piles up on both sides of the victim.
Accumulated leakage past a per-cell threshold flips a bit.

The device refreshes every row within `tREFW` (32–64ms in GDDR6). If the
attacker can't reach the flip threshold inside that window, the leakage
is wiped.

<details>
<summary><b>Question 1.1a:</b> Why two aggressors instead of one?</summary><blockquote>

Double-sided leaks charge into the victim from both sides every round;
effective leakage roughly doubles. This drops the required ACTIVATE count
by 2–5× on modern parts, from hundreds of millions (single-sided) to
~150k per aggressor (double-sided) on GDDR6. That is what makes the
attack practical inside a refresh window.
</blockquote></details>

<details>
<summary><b>Question 1.1b:</b> What does the refresh window buy the defender, and why is it not enough?</summary><blockquote>

It caps how long an attacker has to accumulate ACTIVATEs. In practice
150k activations × ~65ns per cycle ≈ 10–20ms of hammering, comfortably
inside a 32–64ms window, especially for adversarial code running on the
GPU itself, which can saturate the DRAM controller. Refreshing faster
costs bandwidth and still only moves the bar.
</blockquote></details>

<details>
<summary><b>Vocabulary: tREFI vs tREFW</b></summary><blockquote>

- **tREFI** (~1.9µs on GDDR6): interval between REFRESH commands.
- **tREFW** (32–64ms): the time in which every row gets refreshed at
  least once.

When this lab says "64ms refresh window" it means tREFW.
</blockquote></details>


### Exercise 7.4.2: GPU PTEs and the aperture bit

> **Difficulty**: 2/5
> **Importance**: 5/5

The GPU has its own MMU (memory management unit - translates virtual memory addresses into physical memory locations) with 8-byte PTEs stored in VRAM. Each PTE holds:

* a **valid** bit,
* an **aperture** bit: 0 = page lives in GPU VRAM, 1 = page lives in
  host system memory (reached over PCIe),
* a physical frame number (PFN),
* permission / cache-control flags.

A single bit flip changes the *target memory space* of every subsequent
DMA through that virtual address.

<details>
<summary><b>Question 1.2a:</b> SECDED ECC (error correction mechanism) is on the DRAM. Why does that not stop this attack?</summary><blockquote>

SECDED corrects 1-bit flips and detects 2-bit flips within a codeword.
GPUHammer shows that ECC-aware hammering patterns on A100/H100 drive two
flips per codeword, silently corrupting without raising a fault. The
attacker picks PTE locations whose paired flips line up with the hammer
template. ECC slows the attack down; it does not stop it.
</blockquote></details>

<details>
<summary><b>Question 1.2b:</b> After the aperture flips, the PFN bits in the PTE are unchanged. Why does the attacker still end up writing to a <em>useful</em> CPU page?</summary><blockquote>

Because they groomed host memory beforehand so that the PFN in the PTE
happens to match the driver's DMA-mapped page. Allocate + free cycles
until the kernel hands back the target PFN in a predictable position.
The coincidence is engineered, not luck. In this lab
`make_environment()` bakes it in.
</blockquote></details>


### Exercise 7.4.3: Why the IOMMU does not block this write

> **Difficulty**: 3/5
> **Importance**: 5/5

The IOMMU (Intel VT-d / AMD-Vi) enforces **page-granular** DMA isolation:
"this device may read/write this host physical page." It does not look
inside the page.

On the GPUBreach DMA the transaction comes from the GPU's PCIe BDF and
targets the driver's DMA-mapped page, legitimately mapped for that
device. The IOMMU signs off.

<details>
<summary><b>Question 1.3a:</b> What granularity does the IOMMU enforce, and why does that leave the OOB write unblocked?</summary><blockquote>

Page-level (4KB). It asks "is this page mapped for this device?" It does
not ask "is the write staying inside a sub-page software buffer?"
Enforcing sub-page bounds is the kernel's job (the driver's, in this
case), and that check is missing.
</blockquote></details>

<details>
<summary><b>Question 1.3b:</b> Would ATS / PASID change this?</summary><blockquote>

Not here. Both add context to IOMMU translations but still operate at
page granularity. They can narrow *which* pages a device may touch but
not enforce intra-page bounds inside a legitimately mapped page.
</blockquote></details>


### Exercise 7.4.4: Driver OOB → privilege escalation

> **Difficulty**: 3/5
> **Importance**: 4/5

The last link is a classic heap-adjacent out-of-bounds write. The
driver's DMA fast path copies caller-controlled `len(data)` bytes into
the DMA buffer at offset 0. It does not clamp `len(data)` to the
buffer's declared size. Immediately after the buffer, in the same page,
sits a kernel credential struct. Writing 0 into its `euid` gives the
caller root.

<details>
<summary><b>Question 1.4a:</b> Which CWE best captures the driver bug?</summary><blockquote>

**CWE-787: Out-of-bounds Write.** The write stays inside the allocated
page (IOMMU satisfied) but passes the software-defined end of the
buffer. Secondary fits: CWE-20 (missing input validation), CWE-119
(improper restriction of operations within buffer bounds).
</blockquote></details>

<details>
<summary><b>Question 1.4b:</b> Which kernel memory-safety mitigations <em>would</em> catch this, and which would not?</summary><blockquote>

**Catch it**: KASAN (redzones around allocations), a kernel rewritten in
Rust if the write goes through a bounds-checked slice, SLAB hardening
that moves the cred struct elsewhere.

**Don't catch it**: IOMMU (too coarse), stack canaries (no stack frame),
W^X / NX (data-only corruption), CFI / shadow stacks (no indirect call
redirection), SMEP / SMAP (protect against *CPU* user→kernel mistakes,
not PCIe DMA).
</blockquote></details>


## Phase 2: Must-finish: driving the attack to root

Five tiny coding exercises, one for each step in the chain. Each has a
test you run immediately after. Budget ~30–45 minutes total. At the end,
`env.check_all()` prints the flag.

| # | Step | What you write |
|---|------|----------------|
| 2.1 | Pick the right DRAM rows to hammer | `find_aggressors(victim_row)` |
| 2.2 | Hammer until a bit flips | `hammer_until_flip(dram, a, b, row)` |
| 2.3 | Make the GPU MMU pick up the flip | `trigger_pte_refresh(env)` |
| 2.4 | Build an oversized DMA payload | `craft_overflow_payload(euid=0)` |
| 2.5 | Fire the DMA and confirm root | `escalate_privileges(env, payload)` |

**Expected output when Phase 2 succeeds.** After running every cell
through to `env.check_all()`, your terminal should look like this (exact
numbers for rounds/ns will match on every machine because the flip row
is deterministic):

```text
Ex 7.4.5: aggressors for PTE_ROW=4242 → 4241, 4243
  Aggressor geometry correct!
Ex 7.4.6: flipped=True after 150,000 rounds (19.50 ms)
  Hammer loop and cycle accounting correct!
Ex 7.4.7: aperture 0 → 1 (expected 0 → 1)
  PT resync propagated the flip!
Ex 7.4.8: payload=132 bytes (128 filler + 4 euid)
  Payload layout correct!
Ex 7.4.9: root achieved? True
  End-to-end escalation succeeded!
── GPUBreach attack chain ──
  ✓ Stage 1 — aggressor rows identified
  ✓ Stage 2 — flip landed inside the 64ms refresh window
  ✓ Stage 3 — aperture bit flipped in the live PTE
  ✓ Stage 4 — OOB DMA wrote euid=0 into the cred struct

  🎉 All stages succeeded — root achieved.
  FLAG{gpubreach_rowhammer_aperture_oob_root}
```

If Phase 2 is taking **minutes** instead of **milliseconds**, you almost
certainly have a `|a - b| ≠ 2` bug in `find_aggressors`; double-check
Exercise 7.4.5 before anything else.


### Exercise 7.4.5: Aggressor rows for double-sided hammering

> **Difficulty**: 1/5
> **Importance**: 3/5

Given the row that holds the target PTE, return the two aggressor rows
that sandwich it. `DRAM.hammer_once(agg_a, agg_b)` only leaks into the
victim row when `|agg_a - agg_b| == 2` with the victim between them.


```python

def find_aggressors(victim_row: int) -> tuple[int, int]:
    """Return the two aggressor rows for double-sided hammering."""
    # TODO: Return (aggressor_a, aggressor_b) such that victim_row is between them
    # and |aggressor_a - aggressor_b| == 2
    return (0, 0)


agg_a, agg_b = find_aggressors(PTE_ROW)
print(f"Ex 7.4.5: aggressors for PTE_ROW={PTE_ROW} → {agg_a}, {agg_b}")
from section4_test import test_find_aggressors


test_find_aggressors(find_aggressors)

env.stage1_aggressors_ok = True
```

### Exercise 7.4.6: Hammer until a bit flips

> **Difficulty**: 2/5
> **Importance**: 4/5

Drive the hammer loop. Call `dram.hammer_once(agg_a, agg_b)` repeatedly
until `dram.has_flipped(victim_row)` becomes True. Return a dict with
keys "flipped", "rounds", "total_ns".


```python

def hammer_until_flip(dram: DRAM, agg_a: int, agg_b: int, victim_row: int, max_rounds: int = 2_000_000) -> dict:
    """Drive the DRAM into a RowHammer flip on ``victim_row``."""
    # TODO:
    # 1. Initialise total_ns and rounds counters.
    # 2. While `dram.has_flipped(victim_row)` is False (and you are
    #    below `max_rounds`):
    #      - call dram.hammer_once(aggressor_a, aggressor_b)
    #      - add its return value (nanoseconds) to total_ns
    #      - increment rounds
    # 3. Return {"rounds": rounds, "total_ns": total_ns,
    #            "flipped": dram.has_flipped(victim_row)}
    return {"rounds": 0, "total_ns": 0, "flipped": False}


flip_run = hammer_until_flip(env.dram, agg_a, agg_b, PTE_ROW)
print(
    f"Ex 7.4.6: flipped={flip_run['flipped']} after "
    f"{flip_run['rounds']:,} rounds "
    f"({flip_run['total_ns'] / 1_000_000:.2f} ms)"
)
from section4_test import test_hammer_until_flip


test_hammer_until_flip(hammer_until_flip)

env.stage2_flipped_in_refresh_window = (
    flip_run["flipped"]
    and flip_run["total_ns"] / 1_000_000 < REFRESH_WINDOW_MS
)
```

### Exercise 7.4.7: Force the MMU to re-walk the flipped PTE

> **Difficulty**: 1/5
> **Importance**: 4/5

The DRAM bit is flipped, but the GPU MMU's cached copy still says
"aperture = GPU VRAM". Call `page_table.sync_from_dram(env.dram)` to
re-read the PTE bytes from DRAM, then return the `(before, after)` pair
of aperture values.

A real attacker triggers this re-walk with a GPU context switch, an
explicit TLB flush from the driver, or simply by waiting for the cache
line to be evicted.


```python


def trigger_pte_refresh(env: Environment) -> tuple[int, int]:
    """Resync the PT from DRAM. Return (before_aperture, after_aperture)."""
    # TODO: Capture before value, call sync_from_dram, capture after value
    return (0, 0)


before, after = trigger_pte_refresh(env)
print(f"Ex 7.4.7: aperture {before} → {after} (expected 0 → 1)")
from section4_test import test_trigger_pte_refresh


test_trigger_pte_refresh(trigger_pte_refresh)

env.stage3_aperture_changed = (before, after) == (
    APERTURE_GPU_LOCAL,
    APERTURE_SYSTEM,
)
```

### Exercise 7.4.8: Craft the OOB DMA payload

> **Difficulty**: 2/5
> **Importance**: 5/5

You need a byte string for the DMA payload such that:

1. Its length is exactly `DRIVER_BUFFER_SIZE + 4` bytes. The first
   `DRIVER_BUFFER_SIZE` bytes fill the driver buffer; the last 4 bytes
   overflow into the `euid` field of the cred struct.
2. The last 4 bytes encode the integer `0` (root's euid) as a 4-byte
   little-endian number, matching how `KernelCred.euid` is serialised.

You can put any content in the first `DRIVER_BUFFER_SIZE` bytes. The
convention is to use `b"A"` so the hexdump is easy to read.


```python


def craft_overflow_payload(new_euid: int = 0) -> bytes:
    """Return a DMA payload that overflows the driver buffer into euid."""
    # TODO:
    # 1. Fill `DRIVER_BUFFER_SIZE` bytes (e.g. b"A" repeated).
    # 2. Append the little-endian 4-byte encoding of `new_euid`.
    # 3. Return filler + euid_bytes.
    return b""


payload = craft_overflow_payload()
print(
    f"Ex 7.4.8: payload={len(payload)} bytes "
    f"({DRIVER_BUFFER_SIZE} filler + 4 euid)"
)
from section4_test import test_craft_overflow_payload


test_craft_overflow_payload(craft_overflow_payload)
```

### Exercise 7.4.9: Fire the DMA and confirm root

> **Difficulty**: 1/5
> **Importance**: 5/5

Hand the payload to `perform_gpu_dma`. The simulator resolves the PTE,
validates the transaction with the IOMMU (which approves, since the page is
mapped), and writes the payload into the driver page. The overflow lands
on the cred struct and `env.kernel_cred.is_root()` flips to True.

Call `perform_gpu_dma` with the positional arguments
`(payload, VICTIM_GPU_VADDR, env.page_table, env.iommu, env.dram,
env.driver_page)`, then return `env.kernel_cred.is_root()`.


```python


def escalate_privileges(env: Environment, payload: bytes) -> bool:
    """Perform the DMA. Return True iff the cred struct now shows root."""
    # TODO:
    # 1. Call perform_gpu_dma(payload, VICTIM_GPU_VADDR,
    #    env.page_table, env.iommu, env.dram, env.driver_page).
    # 2. Return env.kernel_cred.is_root().
    return False


rooted = escalate_privileges(env, payload)
print(f"Ex 7.4.9: root achieved? {rooted}")
from section4_test import test_escalate_privileges


test_escalate_privileges(escalate_privileges)

env.stage4_root_obtained = rooted
```

### Print the flag

If every stage above succeeded, `env.check_all()` prints the flag.
Otherwise it tells you which stage still needs work.


```python

env.check_all()
```

## Phase 3: Stretch: digging into the primitives (Optional)

Optional exercises for deeper understanding. Each is short and independent.

### Exercise 7.4.10 (Optional): Decode a PTE by hand

> **Difficulty**: 2/5
> **Importance**: 3/5

Prove you understand the PTE byte layout by parsing an 8-byte PTE into a
dict of the form `{"valid": bool, "aperture": int, "physical_frame": int}`
without using `gpubreach_sim.pte.decode_pte`.

Recall the layout (from the PTE module docstring):

* byte 0: flags (bit 0 = valid, bit 1 = aperture, bits 2–7 reserved)
* bytes 1–6: 48-bit little-endian physical frame number
* byte 7: reserved


```python

def decode_pte_manually(raw: bytes) -> dict:
    """Hand-decoded PTE; do not call gpubreach_sim.decode_pte."""
    # TODO: pick apart raw[0], raw[1:7], bit by bit.
    return {"valid": False, "aperture": 0, "physical_frame": 0}


# A sample PTE: valid=1, aperture=1, PFN=0xABCDEF
sample = bytes([0b0000_0011]) + (0xABCDEF).to_bytes(6, "little") + b"\x00"
print(f"Ex 7.4.10: decoded = {decode_pte_manually(sample)}")
from section4_test import test_decode_pte_manually


test_decode_pte_manually(decode_pte_manually)
```

### Exercise 7.4.11 (Optional): Inspect the exact flipped bit

> **Difficulty**: 2/5
> **Importance**: 3/5

Compare the PTE's raw bytes in DRAM before and after the RowHammer flip
and return the set of (byte_offset, bit_position) pairs that changed.
There should be exactly one.

You may use `env.dram.read(row, offset, length)` to grab bytes, and
you'll need a fresh environment so the "before" snapshot is pristine.


```python


def find_flipped_bits(before: bytes, after: bytes) -> set[tuple[int, int]]:
    """Return {(byte_index, bit_position), ...} where before != after."""
    # TODO: iterate over pairs of bytes, XOR them, and record each
    # differing bit as (byte_index, bit_position).
    return set()


fresh3 = make_environment()
pre = fresh3.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
while not fresh3.dram.has_flipped(PTE_ROW):
    fresh3.dram.hammer_once(PTE_ROW - 1, PTE_ROW + 1)
post = fresh3.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
print(f"Ex 7.4.11: flipped bits = {find_flipped_bits(pre, post)}")
from section4_test import test_find_flipped_bits


test_find_flipped_bits(find_flipped_bits)
```

### Exercise 7.4.12 (Optional): Budget the hammer against the refresh window

> **Difficulty**: 1/5
> **Importance**: 3/5

Before hammering, you'd want to know: will we even finish the activations
inside the refresh window? Compute:

1. The minimum number of hammer *rounds* to cross the flip threshold.
   Each round hits both aggressors, so each round adds one activation
   per aggressor.
2. The simulated nanoseconds those rounds will cost. Each round costs
   `2 * ACTIVATE_PRECHARGE_NS`.
3. Whether that total time fits inside the refresh window.

Return a dict with keys `"rounds"`, `"total_ns"`, `"total_ms"`,
`"fits_refresh_window"`.


```python

def hammer_budget(threshold: int = HAMMER_THRESHOLD_ACTIVATIONS, tRC_ns: int = ACTIVATE_PRECHARGE_NS, refresh_ms: int = REFRESH_WINDOW_MS) -> dict:
    """Compute worst-case hammering cost."""
    return {"rounds": 0, "total_ns": 0, "total_ms": 0.0, "fits_refresh_window": False}




budget = hammer_budget()
print(
    f"Ex 7.4.12: {budget['rounds']:,} rounds × {2 * ACTIVATE_PRECHARGE_NS} ns "
    f"= {budget['total_ms']:.2f} ms "
    f"(fits 64ms window: {budget['fits_refresh_window']})"
)
from section4_test import test_hammer_budget


test_hammer_budget(hammer_budget)
```

### Exercise 7.4.13 (Optional): Maximum hammer rounds inside the window

> **Difficulty**: 1/5
> **Importance**: 3/5

Flip the budget question around: given the refresh window, how *many*
rounds could the attacker fit at most? Compare it to
`HAMMER_THRESHOLD_ACTIVATIONS`: how comfortably do we fit?


```python

def max_rounds_in_window(refresh_ms: int = REFRESH_WINDOW_MS, tRC_ns: int = ACTIVATE_PRECHARGE_NS) -> int:
    """Maximum rounds that fit in refresh window."""
    return 0


max_rounds = max_rounds_in_window()
headroom = max_rounds / HAMMER_THRESHOLD_ACTIVATIONS
print(
    f"Ex 7.4.13: up to {max_rounds:,} rounds fit in {REFRESH_WINDOW_MS} ms "
    f"→ {headroom:.1f}× threshold headroom"
)
from section4_test import test_max_rounds_in_window


test_max_rounds_in_window(max_rounds_in_window)
```

### Exercise 7.4.14 (Optional): The IOMMU blocks what it promises to block

> **Difficulty**: 2/5
> **Importance**: 4/5

Prove the IOMMU is doing its job: it *does* block writes to physical
pages it has not mapped for the GPU. Use `env.iommu.validate(page, offset,
length)` to confirm:

1. Writes to `env.driver_page` at offset 0 with length `PAGE_SIZE` are
   allowed.
2. Writes to `env.driver_page` at offset 0 with length `PAGE_SIZE + 1`
   are rejected (cross a page boundary).
3. Writes to a *different* `DriverPage` instance are rejected (not
   mapped).

Return a dict with three booleans: `"intra_page_ok"`, `"overflow_page"`,
`"other_page"`.


```python


def probe_iommu(env: Environment) -> dict:
    """Probe IOMMU enforcement."""
    return {"intra_page_ok": False, "overflow_page": False, "other_page": False}


probe = probe_iommu(env)
print(f"Ex 7.4.14: IOMMU probe = {probe}")
from section4_test import test_probe_iommu


test_probe_iommu(probe_iommu)
```

### Exercise 7.4.15 (Optional): Measure the OOB overflow precisely

> **Difficulty**: 2/5
> **Importance**: 3/5

Given a DMA payload length, how many bytes overflow past the driver
buffer into adjacent kernel memory? Return 0 if no overflow.


```python


def overflow_bytes(payload_len: int) -> int:
    """Bytes that overflow past DRIVER_BUFFER_SIZE."""
    return 0


for n in [0, DRIVER_BUFFER_SIZE - 1, DRIVER_BUFFER_SIZE, DRIVER_BUFFER_SIZE + 4]:
    print(f"Ex 7.4.15: payload {n}B → overflow {overflow_bytes(n)}B")
from section4_test import test_overflow_bytes


test_overflow_bytes(overflow_bytes)
```

### Exercise 7.4.16 (Optional): A tighter payload

> **Difficulty**: 2/5
> **Importance**: 3/5

The payload in Exercise 7.4.8 overshoots: it writes 132 bytes where 132 is
exactly `DRIVER_BUFFER_SIZE + 4`. What if the cred struct's euid field
isn't at the very start of the overflow region, but at some `offset`
past `CRED_OFFSET`? Write a parameterised payload builder.

The new signature:
`craft_precise_payload(cred_offset_in_page: int, new_euid: int) → bytes`

The payload should have length `cred_offset_in_page + 4` so that the last
4 bytes land exactly at `cred_offset_in_page` inside the driver page
when the driver copies starting at `DRIVER_BUFFER_OFFSET = 0`.


```python

def craft_precise_payload(cred_offset_in_page: int, new_euid: int) -> bytes:
    """Precise payload targeting specific offset."""
    return b""



tight = craft_precise_payload(CRED_OFFSET, 0)
print(f"Ex 7.4.16: precise payload is {len(tight)} bytes")
from section4_test import test_craft_precise_payload


test_craft_precise_payload(craft_precise_payload)
```

## GPUBreach Summary

You implemented an end-to-end GPUBreach-style attack chain:

1. **RowHammer** - Flipped a GPU DRAM bit within the refresh window
2. **Aperture corruption** - Redirected GPU virtual address from VRAM to system memory
3. **Sub-page isolation failure** - Abused an out-of-bounds write within a legitimately mapped IOMMU page
4. **Privilege escalation** - Overwrote kernel credentials via buffer overflow

### Key Takeaways

- RowHammer is practical against GPU memory with comfortable timing margins
- Single bit flips in page table entries change memory access semantics
- IOMMU operates at page granularity, not sub-page buffer boundaries
- Defense requires structural changes: SLAB hardening, GPU isolation, refresh tuning

### Further Reading

- [GPUHammer: RowHammer Attacks on GPU Memories are Practical](https://gpuhammer.com/)
- [Flipping Bits in Memory Without Accessing Them](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf)
- [Exploiting the DRAM rowhammer bug to gain kernel privileges](https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html)

---
