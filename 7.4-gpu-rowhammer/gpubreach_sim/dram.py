"""Simulated GDDR6 DRAM bank with RowHammer bit-flip behaviour.

We model a single DRAM bank as a list of rows. Each row is a mutable byte
buffer. In real DRAM, a row is a few kilobytes of storage opened by a single
ACTIVATE command into the bank's row buffer.

RowHammer (and GPUHammer, its GDDR6 variant) exploits charge leakage across
adjacent rows: repeatedly ACTIVATEing two neighbouring rows faster than the
device can refresh the row between them causes charge to bleed out of the
victim row, flipping bits from 1 → 0 (and rarely 0 → 1).

In the real hardware:
  * GDDR6 refresh interval tREFW ≈ 32ms for the full array (we round up to
    64ms to match DDR4/DDR5 numbers students often memorise).
  * Per-ACTIVATE-PRECHARGE timing is ~60–70ns (tRC).
  * Flips begin near ~150k activations per aggressor when double-sided, on
    consumer parts with no TRR. See the GPUHammer paper by Jattke et al.

We pick values matching these numbers and bake in a deterministic single-bit
flip inside the victim row once the activation count crosses the threshold.
That keeps the lab reproducible in plain Python while preserving the
qualitative story: hammer enough, fast enough, and you flip a specific bit
before the row refreshes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Physical parameters ──────────────────────────────────────────────────────
ROWS_PER_BANK: int = 8192
"""Number of rows in one DRAM bank. GDDR6 devices have 16k rows per bank in
practice; 8k keeps memory use small while the row indices still look
realistic."""

ROW_SIZE_BYTES: int = 1024
"""Bytes per DRAM row (simplification: real GDDR6 row buffers are 2KB)."""

ACTIVATE_PRECHARGE_NS: int = 65
"""Nanoseconds for one ACTIVATE → PRECHARGE cycle (tRC). Both aggressors are
hammered in one round, so a round costs 2 × ACTIVATE_PRECHARGE_NS."""

HAMMER_THRESHOLD_ACTIVATIONS: int = 150_000
"""Minimum aggressor activations per aggressor to induce a flip in the victim
row when hammering double-sided. Matches the GPUHammer paper's worst-case
threshold on consumer RTX 3080/3090-class parts."""

REFRESH_WINDOW_MS: int = 64
"""The DRAM refresh window. After this many milliseconds the DRAM controller
issues REFRESH commands and any accumulated charge leakage is wiped — the
attacker must land the flip before this timer expires."""


# ── DRAM model ───────────────────────────────────────────────────────────────
@dataclass
class _VictimState:
    """Per-victim-row hammering bookkeeping."""

    activations: int = 0
    flipped: bool = False


@dataclass
class DRAM:
    """A toy GDDR6 bank.

    Rows are stored as a list of bytearrays. Reads and writes go through
    ``read`` / ``write``. Hammering uses ``hammer_once`` which models one
    round of ACTIVATE(agg_a) → PRECHARGE → ACTIVATE(agg_b) → PRECHARGE.

    A row sandwiched between two neighbouring aggressors (``victim = agg±1``)
    accumulates "activations" on every round. Once the per-victim counter
    crosses :data:`HAMMER_THRESHOLD_ACTIVATIONS`, a single bit flip is
    applied to the row — unless :meth:`refresh` has been called since, which
    resets all activation counters.

    Notes on the modelling choices:

    * Only double-sided hammering flips bits here. Hammering non-adjacent
      rows does nothing, matching the "two aggressors sandwich a victim"
      geometry of real double-sided attacks.
    * The flipped bit is deterministic — it lives at a configurable
      (byte, bit) position inside the victim row. In reality the flip
      location depends on the specific DRAM device and data pattern, but
      templates of known-flipping locations are published for consumer
      parts, so "pick the PTE offset such that bit 1 flips" is a realistic
      (if heavily pre-profiled) attacker capability.
    """

    rows: list[bytearray] = field(default_factory=list)
    victim_state: dict[int, _VictimState] = field(default_factory=dict)
    flip_location: tuple[int, int, int] = (0, 0, 1)
    """(victim_row, byte_offset, bit_position) — the single bit the hammering
    primitive is pre-profiled to flip."""

    @classmethod
    def zeros(cls, num_rows: int = ROWS_PER_BANK) -> "DRAM":
        """Allocate a zero-initialised bank."""
        return cls(rows=[bytearray(ROW_SIZE_BYTES) for _ in range(num_rows)])

    # ── Normal reads / writes ────────────────────────────────────────────
    def read(self, row: int, offset: int = 0, length: int | None = None) -> bytes:
        """Read from a row. Returns a copy (so callers can't alias storage)."""
        if length is None:
            length = ROW_SIZE_BYTES - offset
        return bytes(self.rows[row][offset : offset + length])

    def write(self, row: int, offset: int, data: bytes) -> None:
        """Write ``data`` into a row at ``offset``. Resets hammer state for
        this row (writing the row is itself an ACTIVATE, which refreshes)."""
        end = offset + len(data)
        assert end <= ROW_SIZE_BYTES, f"write past row end: {end} > {ROW_SIZE_BYTES}"
        self.rows[row][offset:end] = data
        self.victim_state.pop(row, None)

    # ── Hammering primitive ──────────────────────────────────────────────
    def hammer_once(self, aggressor_a: int, aggressor_b: int) -> int:
        """One round of double-sided hammering.

        Returns the number of nanoseconds this round cost (2 × tRC).

        If ``|aggressor_a - aggressor_b| == 2``, the row between them is the
        victim and gets its activation counter incremented by 1 per
        aggressor (so +2 per round). Once the counter reaches the flip
        threshold the configured bit is cleared in the victim row.
        """
        if abs(aggressor_a - aggressor_b) == 2:
            victim = (aggressor_a + aggressor_b) // 2
            state = self.victim_state.setdefault(victim, _VictimState())
            if not state.flipped:
                state.activations += 2  # one per aggressor
                if state.activations >= 2 * HAMMER_THRESHOLD_ACTIVATIONS:
                    self._apply_flip(victim)
                    state.flipped = True
        return 2 * ACTIVATE_PRECHARGE_NS

    def refresh(self) -> None:
        """Simulate a whole-array refresh — wipes all accumulated leakage.

        Real controllers issue REFRESH every tREFI (≈1.9µs per row in
        GDDR6), so hammering has to reach the flip threshold within
        :data:`REFRESH_WINDOW_MS` or the accumulated charge leakage goes
        away.
        """
        self.victim_state.clear()

    def has_flipped(self, victim_row: int) -> bool:
        """True iff a RowHammer-induced flip has landed in ``victim_row``."""
        st = self.victim_state.get(victim_row)
        return bool(st and st.flipped)

    # ── Internal ─────────────────────────────────────────────────────────
    def _apply_flip(self, victim: int) -> None:
        """Toggle the configured bit in the victim row.

        Real RowHammer flips are directional: charge leaks in only one
        direction, and which logical value that maps to (0→1 or 1→0)
        depends on the cell's "true" vs "anti" encoding, which varies per
        device and per column. Attackers template flips ahead of time and
        pick bits that flip in the direction they need. Here we just toggle
        — the lab's initial PTE sets the aperture bit to 0 so the flip
        drives it to 1, which is what we want.
        """
        flip_row, byte_off, bit_pos = self.flip_location
        if victim != flip_row:
            # Row is adjacent-hammerable but we haven't pre-profiled a flip
            # for it. In this simulation only the configured row flips.
            return
        self.rows[victim][byte_off] ^= (1 << bit_pos)
