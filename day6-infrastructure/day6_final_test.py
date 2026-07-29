# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
from collections.abc import Callable
from pathlib import Path
from aisb_utils import report
from gpubreach_sim import *
from gpubreach_sim.dma import DriverPage



@report
def test_find_aggressors(solution: Callable[[int], tuple[int, int]]):
    for victim in [PTE_ROW, 100, 7777, 42]:
        a, b = solution(victim)
        assert abs(a - b) == 2, (
            f"aggressors for victim={victim} must differ by 2, got ({a}, {b})"
        )
        assert (a + b) // 2 == victim, (
            f"victim={victim} must lie between aggressors, midpoint was "
            f"{(a + b) // 2}"
        )
    print("  Aggressor geometry correct!")




@report
def test_hammer_until_flip(solution: Callable[..., dict]):
    fresh = make_environment()
    victim = fresh.dram.flip_location[0]
    a, b = victim - 1, victim + 1
    res = solution(fresh.dram, a, b, victim)
    assert res["flipped"], f"expected a flip to land, got {res}"
    assert res["rounds"] >= HAMMER_THRESHOLD_ACTIVATIONS, (
        f"expected at least {HAMMER_THRESHOLD_ACTIVATIONS:,} rounds, "
        f"got {res['rounds']:,}"
    )
    expected_ns = res["rounds"] * 2 * ACTIVATE_PRECHARGE_NS
    assert res["total_ns"] == expected_ns, (
        f"total_ns must accumulate hammer_once return; "
        f"expected {expected_ns}, got {res['total_ns']}"
    )
    # Using a non-double-sided pair must not flip anything.
    other = make_environment()
    no_flip = solution(other.dram, 0, 5, other.dram.flip_location[0], max_rounds=1000)
    assert not no_flip["flipped"], (
        "hammer_until_flip must stop at max_rounds if aggressors are wrong"
    )
    print("  Hammer loop and cycle accounting correct!")




@report
def test_trigger_pte_refresh(solution: Callable[[Environment], tuple[int, int]]):
    fresh = make_environment()
    # Hammer it directly (so we test *only* the refresh step in isolation).
    v = fresh.dram.flip_location[0]
    while not fresh.dram.has_flipped(v):
        fresh.dram.hammer_once(v - 1, v + 1)
    assert fresh.page_table.cached_pte.aperture == APERTURE_GPU_LOCAL, (
        "cached PTE must still be stale before sync_from_dram runs"
    )
    pair = solution(fresh)
    assert pair == (APERTURE_GPU_LOCAL, APERTURE_SYSTEM), (
        f"expected (0, 1), got {pair}"
    )
    assert fresh.page_table.cached_pte.aperture == APERTURE_SYSTEM, (
        "after refresh, cached_pte must reflect APERTURE_SYSTEM"
    )
    print("  PT resync propagated the flip!")




@report
def test_craft_overflow_payload(solution: Callable[..., bytes]):
    p = solution()
    assert isinstance(p, (bytes, bytearray)), (
        f"payload must be bytes, got {type(p).__name__}"
    )
    assert len(p) == DRIVER_BUFFER_SIZE + 4, (
        f"expected {DRIVER_BUFFER_SIZE + 4} bytes, got {len(p)}"
    )
    assert int.from_bytes(p[-4:], "little") == 0, (
        f"last 4 bytes should encode euid=0 little-endian, "
        f"got {p[-4:].hex()}"
    )
    # Parameterised euid should round-trip.
    p2 = solution(42)
    assert int.from_bytes(p2[-4:], "little") == 42, (
        f"parameterised new_euid should be respected, got {p2[-4:].hex()}"
    )
    print("  Payload layout correct!")




@report
def test_escalate_privileges(solution: Callable[..., bool]):
    # Build a fresh chain-completed env (DRAM flipped + PT resynced)
    # so the solution is exercised in isolation.
    fresh = make_environment()
    v = fresh.dram.flip_location[0]
    while not fresh.dram.has_flipped(v):
        fresh.dram.hammer_once(v - 1, v + 1)
    fresh.page_table.sync_from_dram(fresh.dram)
    # Build an overflow payload directly so the test is independent of
    # Exercise 6.3.8 still being defined in the caller's namespace.
    overflow_payload = b"A" * DRIVER_BUFFER_SIZE + (0).to_bytes(4, "little")
    assert not fresh.kernel_cred.is_root(), (
        "sanity: cred must start non-root"
    )
    ok = solution(fresh, overflow_payload)
    assert ok is True, "escalate_privileges should return True on success"
    assert fresh.kernel_cred.is_root(), (
        "kernel_cred.euid should be 0 after the OOB write"
    )
    # Test idempotency-ish: a payload that does NOT overflow must not root.
    fresh2 = make_environment()
    while not fresh2.dram.has_flipped(v):
        fresh2.dram.hammer_once(v - 1, v + 1)
    fresh2.page_table.sync_from_dram(fresh2.dram)
    tiny = b"A" * (DRIVER_BUFFER_SIZE - 1)  # no overflow
    ok2 = solution(fresh2, tiny)
    assert ok2 is False, (
        "a payload shorter than DRIVER_BUFFER_SIZE must NOT escalate"
    )
    print("  End-to-end escalation succeeded!")




@report
def test_decode_pte_manually(solution: Callable[[bytes], dict]):
    sample = bytes([0b0000_0011]) + (0xABCDEF).to_bytes(6, "little") + b"\x00"
    got = solution(sample)
    assert got["valid"] is True, f"'valid' should be True, got {got['valid']}"
    assert got["aperture"] == 1, f"'aperture' should be 1, got {got['aperture']}"
    assert got["physical_frame"] == 0xABCDEF, (
        f"'physical_frame' should be 0xABCDEF, got {got['physical_frame']:#x}"
    )
    # aperture=0, valid=1 case
    other = bytes([0b0000_0001]) + (1).to_bytes(6, "little") + b"\x00"
    got2 = solution(other)
    assert got2 == {"valid": True, "aperture": 0, "physical_frame": 1}
    # invalid
    inv = bytes(8)
    got3 = solution(inv)
    assert got3["valid"] is False, "all-zero PTE must be invalid"
    print("  Hand-decoded PTE matches the simulator's decode!")




@report
def test_find_flipped_bits(solution: Callable[[bytes, bytes], set]):
    fresh = make_environment()
    pre = fresh.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
    while not fresh.dram.has_flipped(PTE_ROW):
        fresh.dram.hammer_once(PTE_ROW - 1, PTE_ROW + 1)
    post = fresh.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
    flips = solution(pre, post)
    assert flips == {(0, APERTURE_BIT_POS)}, (
        f"expected exactly the aperture bit to flip, got {flips}"
    )
    # Sanity: identical byte strings → no flips.
    assert solution(b"\x00\x01\x02", b"\x00\x01\x02") == set()
    # Sanity: two byte differences → two (or more) flips.
    assert len(solution(b"\x00\x00", b"\xff\xff")) == 16
    print("  Exactly one flip, at the aperture bit, as templated!")




@report
def test_hammer_budget(solution: Callable[..., dict]):
    res = solution()
    assert res["rounds"] == HAMMER_THRESHOLD_ACTIVATIONS, (
        f"expected {HAMMER_THRESHOLD_ACTIVATIONS} rounds, got {res['rounds']}"
    )
    expected_ns = HAMMER_THRESHOLD_ACTIVATIONS * 2 * ACTIVATE_PRECHARGE_NS
    assert res["total_ns"] == expected_ns, (
        f"expected total_ns={expected_ns}, got {res['total_ns']}"
    )
    assert abs(res["total_ms"] - expected_ns / 1_000_000) < 1e-9, (
        f"total_ms should be total_ns / 1e6, got {res['total_ms']}"
    )
    assert res["fits_refresh_window"] is True, (
        f"with these parameters the budget should fit {REFRESH_WINDOW_MS} ms, "
        f"got fits_refresh_window={res['fits_refresh_window']}"
    )
    # Sanity: with a 1-ms refresh window it must NOT fit.
    tight = solution(refresh_ms=1)
    assert tight["fits_refresh_window"] is False, (
        "fits_refresh_window must respect the refresh_ms parameter"
    )
    print("  Budget arithmetic correct!")




@report
def test_max_rounds_in_window(solution: Callable[..., int]):
    got = solution()
    expected = (REFRESH_WINDOW_MS * 1_000_000) // (2 * ACTIVATE_PRECHARGE_NS)
    assert got == expected, f"expected {expected}, got {got}"
    # With a 1ms window the budget must shrink by ~64×.
    short = solution(refresh_ms=1)
    assert short == 1_000_000 // (2 * ACTIVATE_PRECHARGE_NS), (
        f"parameterised window ignored, got {short}"
    )
    # Headroom must be > 1× or the lab wouldn't work.
    assert got > HAMMER_THRESHOLD_ACTIVATIONS, (
        f"budget {got} must exceed threshold {HAMMER_THRESHOLD_ACTIVATIONS}"
    )
    print("  Budget arithmetic and parameterisation both correct!")




@report
def test_probe_iommu(solution: Callable[[Environment], dict]):
    # Use a fresh environment so the test does not depend on a module-level
    # ``env`` that the answer file may or may not still have around.
    fresh = make_environment()
    got = solution(fresh)
    assert got["intra_page_ok"] is True, (
        "IOMMU should allow a full-page write to the mapped driver page"
    )
    assert got["overflow_page"] is False, (
        "IOMMU must reject writes that cross the page boundary"
    )
    assert got["other_page"] is False, (
        "IOMMU must reject writes to a page it hasn't mapped for this device"
    )
    print("  IOMMU enforces what it promises, and no more!")




@report
def test_overflow_bytes(solution: Callable[[int], int]):
    assert solution(0) == 0
    assert solution(DRIVER_BUFFER_SIZE) == 0
    assert solution(DRIVER_BUFFER_SIZE + 4) == 4
    assert solution(DRIVER_BUFFER_SIZE + 100) == 100
    assert solution(DRIVER_BUFFER_SIZE - 1) == 0  # underflow clamps to 0
    print("  Overflow arithmetic correct!")




@report
def test_craft_precise_payload(solution: Callable[[int, int], bytes]):
    # Using the simulator's real CRED_OFFSET must escalate.
    fresh = make_environment()
    v = fresh.dram.flip_location[0]
    while not fresh.dram.has_flipped(v):
        fresh.dram.hammer_once(v - 1, v + 1)
    fresh.page_table.sync_from_dram(fresh.dram)
    p = solution(CRED_OFFSET, 0)
    assert len(p) == CRED_OFFSET + 4, (
        f"expected {CRED_OFFSET + 4} bytes, got {len(p)}"
    )
    perform_gpu_dma(
        p,
        VICTIM_GPU_VADDR,
        fresh.page_table,
        fresh.iommu,
        fresh.dram,
        fresh.driver_page,
    )
    assert fresh.kernel_cred.is_root(), (
        "payload should have landed euid=0 at CRED_OFFSET"
    )
    # And it should respect non-zero euids, too.
    p2 = solution(CRED_OFFSET, 1337)
    assert int.from_bytes(p2[-4:], "little") == 1337
    print("  Precise payload placement correct!")
