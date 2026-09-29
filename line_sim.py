"""Deterministic, stateless simulation of the inspection line.

The whole line is a pure function of a small control state (mode, accumulated simulation
seconds, start timestamp, a random seed and the cycles forced to fail) and the current time.
The state lives in the visitor's signed session cookie, so every server instance, including
separate serverless instances handling concurrent requests, computes the same counters, log
and verdicts. Unit outcomes come from a hash of (seed, cycle), so they never change once shown.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from functools import lru_cache

CYCLE_SECONDS = 4.0
COMPLETE_AT = 0.9          # a unit is counted when its cycle passes 90%
FAIL_RATE = 0.15
DEFECT_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
PRICE, UNIT_COST, PERFORMANCE = 45.0, 25.0, 0.96
MAX_FORCED = 16
RECENT_LOGS = 20
_KEYS = {"v", "seed", "mode", "acc", "started", "session_start", "forced"}


def new_state(now: float) -> dict:
    return {"v": 1, "seed": secrets.token_hex(4), "mode": "PAUSED", "acc": 0.0, "started": None,
            "session_start": now, "forced": []}


def is_valid(state) -> bool:
    return isinstance(state, dict) and state.get("v") == 1 and _KEYS <= state.keys()


def sim_time(state: dict, now: float) -> float:
    running = state["mode"] == "RUNNING" and state["started"] is not None
    return state["acc"] + (max(0.0, now - state["started"]) if running else 0.0)


def _uniform(seed: str, cycle: int, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{cycle}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def _fails(seed: str, forced, cycle: int) -> bool:
    return cycle in forced or _uniform(seed, cycle, "fail") < FAIL_RATE


def unit(state: dict, cycle: int) -> tuple[str, str | None]:
    """(status, defect class) of the unit produced in `cycle`."""
    if _fails(state["seed"], state["forced"], cycle):
        return "FAIL", DEFECT_CLASSES[int(_uniform(state["seed"], cycle, "defect") * len(DEFECT_CLASSES))]
    return "OK", None


BLOCK = 512  # cycles per cached block of OK counts


@lru_cache(maxsize=4096)
def _block_ok(seed: str, forced: tuple, block: int) -> int:
    start = block * BLOCK
    return sum(not _fails(seed, forced, c) for c in range(start, start + BLOCK))


def ok_before(state: dict, n: int) -> int:
    """Number of OK units among cycles [0, n).

    Still a pure function of (seed, forced cycles, n): full blocks are memoised per process, so a
    poll costs a handful of cache hits plus at most BLOCK - 1 hashes instead of one hash per unit
    the line has ever produced.
    """
    seed, forced = state["seed"], tuple(sorted(state["forced"]))
    full = n // BLOCK
    ok = sum(_block_ok(seed, forced, b) for b in range(full))
    return ok + sum(not _fails(seed, forced, c) for c in range(full * BLOCK, n))


def completed_units(t: float) -> int:
    cycle, progress = int(t // CYCLE_SECONDS), (t % CYCLE_SECONDS) / CYCLE_SECONDS
    return cycle + (1 if progress > COMPLETE_AT else 0)


def apply_command(state: dict, command: str | None, now: float) -> dict:
    state = {**state, "forced": list(state["forced"])}
    t = sim_time(state, now)
    if command == "START" and state["mode"] != "RUNNING":
        state.update(mode="RUNNING", acc=t, started=now)
    elif command == "PAUSE" and state["mode"] == "RUNNING":
        state.update(mode="PAUSED", acc=t, started=None)
    elif command == "ESTOP":
        state.update(mode="ESTOP", acc=t, started=None)
    elif command == "RESET":
        return new_state(now)
    elif command == "SIMULATE_FAIL":
        # Once the line has run, the current cycle's unit is already decided: force the next one.
        target = int(t // CYCLE_SECONDS) + (1 if t > 0 else 0)
        if target not in state["forced"]:
            state["forced"] = (state["forced"] + [target])[-MAX_FORCED:]
    return state


def _oee(state: dict, t: float, now: float, ok: int, total: int) -> tuple[float, float, float]:
    session_time = now - state["session_start"]
    availability = min(t / session_time, 1.0) if session_time > 1 else 0.0
    quality = ok / total if total else 1.0
    return availability, quality, availability * PERFORMANCE * quality


def history(state: dict, now: float, limit: int | None = None) -> list[dict]:
    """Completed units, oldest first, with the wall-clock time each one finished."""
    t = sim_time(state, now)
    done = completed_units(t)
    first = 0 if limit is None else max(0, done - limit)
    rows, ok = [], ok_before(state, first)
    for cycle in range(first, done):
        status, defect = unit(state, cycle)
        ok += status == "OK"
        finished = now - (t - (cycle + COMPLETE_AT) * CYCLE_SECONDS)
        _, _, oee = _oee(state, t, now, ok, cycle + 1)
        rows.append({"timestamp": datetime.fromtimestamp(finished), "unit_id": f"U_{cycle:04}",
                     "status": status, "defect": defect, "oee_score": round(oee, 4)})
    return rows


def snapshot(state: dict, now: float) -> dict:
    """Everything the HMI and the 3D twin read from /api/data."""
    t = sim_time(state, now)
    cycle = int(t // CYCLE_SECONDS)
    done = completed_units(t)
    ok = ok_before(state, done)
    nok = done - ok
    started = t > 0 or state["mode"] == "RUNNING"
    status, defect = unit(state, cycle) if started else ("PENDING", None)
    availability, quality, oee = _oee(state, t, now, ok, done)
    logs = [{"time": r["timestamp"].strftime("%H:%M:%S"), "id": r["unit_id"], "status": r["status"], "defect": r["defect"]}
            for r in reversed(history(state, now, RECENT_LOGS))]
    return {
        "system_mode": state["mode"], "total_units": done, "ok_units": ok, "nok_units": nok,
        "revenue": ok * PRICE, "cost": done * UNIT_COST, "net_profit": ok * PRICE - done * UNIT_COST,
        "availability": availability, "performance": PERFORMANCE, "quality": quality, "oee": oee,
        "recent_logs": logs, "sim_time": t, "cycle_seconds": CYCLE_SECONDS,
        "current_unit_id": f"U_{cycle:04}", "current_unit_status": status, "current_defect": defect,
        "status_cycle": cycle if started else -1,
    }
