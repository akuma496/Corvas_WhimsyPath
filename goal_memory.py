"""
Layer-2 state: Goal Memory.

A deterministic, persistent store of the commander's active engineering-material
goals, plus the per-goal notify latch (moved here from the plugin's in-memory
bool so it survives restarts). NO LLM in this file — converting natural-language
goal statements ("I need 3x Exceptional Scrambled Emission Data for grade 5 FSD")
into MaterialGoal entries is the NEXT milestone and is the only place a small LLM
parse legitimately enters the design.

Persistence is a plain JSON file in the plugin data folder (stdlib json only), so
the whole store is importable and testable without the app's runtime deps.
Reads/writes are fail-safe: a missing or corrupt file yields an empty store
rather than raising, and malformed goal records are skipped, not fatal.

Depends only on material_need_detector (stdlib-only) -> safe to import offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields
from typing import Any, Iterable, Optional

from .material_need_detector import MaterialGoal, TEST_GOAL

SCHEMA_VERSION = 1

_GOAL_FIELDS = {f.name for f in fields(MaterialGoal)}


def _goal_key(goal: MaterialGoal) -> str:
    """Stable identity for a goal + its latch. Material internal name is unique
    enough for the current single-material-per-goal model."""
    return goal.material_internal_name.lower()


def _goal_from_dict(d: dict) -> Optional[MaterialGoal]:
    """Reconstruct a MaterialGoal from stored JSON, tolerating schema drift.
    Unknown keys are dropped; a record missing required fields is skipped."""
    if not isinstance(d, dict):
        return None
    filtered = {k: v for k, v in d.items() if k in _GOAL_FIELDS}
    try:
        return MaterialGoal(**filtered)
    except TypeError:
        return None


def default_goals() -> list[MaterialGoal]:
    """Seed goal(s) for a fresh store. The FSD-data test goal is a PLACEHOLDER
    until the LLM natural-language parse (next milestone) lets the user set real
    goals; it keeps the pipeline demonstrable in the meantime."""
    return [TEST_GOAL]


class GoalMemory:
    def __init__(
        self,
        goals: Optional[Iterable[MaterialGoal]] = None,
        notified: Optional[dict[str, bool]] = None,
    ):
        self._goals: dict[str, MaterialGoal] = {}
        for g in (goals or []):
            self._goals[_goal_key(g)] = g
        self._notified: dict[str, bool] = {str(k): bool(v) for k, v in (notified or {}).items()}

    # ------------------------------------------------------------------ #
    # Queries                                                            #
    # ------------------------------------------------------------------ #
    def list_goals(self) -> list[MaterialGoal]:
        return list(self._goals.values())

    def get(self, key: str) -> Optional[MaterialGoal]:
        return self._goals.get(key.lower())

    def is_empty(self) -> bool:
        return not self._goals

    def __len__(self) -> int:
        return len(self._goals)

    # ------------------------------------------------------------------ #
    # Mutations                                                          #
    # ------------------------------------------------------------------ #
    def add_goal(self, goal: MaterialGoal) -> None:
        key = _goal_key(goal)
        self._goals[key] = goal
        self._notified.setdefault(key, False)

    def remove_goal(self, key: str) -> None:
        key = key.lower()
        self._goals.pop(key, None)
        self._notified.pop(key, None)

    def clear(self) -> None:
        self._goals.clear()
        self._notified.clear()

    # ------------------------------------------------------------------ #
    # Per-goal notify latch (edge-trigger; dispatch once per unmet->met) #
    # ------------------------------------------------------------------ #
    def is_notified(self, key: str) -> bool:
        return self._notified.get(key.lower(), False)

    def mark_notified(self, key: str) -> None:
        self._notified[key.lower()] = True

    def reset_notified(self, key: str) -> None:
        self._notified[key.lower()] = False

    # ------------------------------------------------------------------ #
    # Persistence (fail-safe JSON)                                       #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "goals": [asdict(g) for g in self._goals.values()],
            "notified": dict(self._notified),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoalMemory":
        raw_goals = d.get("goals") if isinstance(d, dict) else None
        goals: list[MaterialGoal] = []
        for item in (raw_goals or []):
            g = _goal_from_dict(item)
            if g is not None:
                goals.append(g)
        notified = d.get("notified") if isinstance(d, dict) else None
        return cls(goals=goals, notified=notified if isinstance(notified, dict) else None)

    def save(self, path: str) -> None:
        """Atomic write (tmp + replace) so a crash mid-write can't corrupt the store."""
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "GoalMemory":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except Exception:
            # Corrupt/unreadable store -> start empty rather than crash the plugin.
            return cls()

    @classmethod
    def load_or_seed(cls, path: str, defaults: Optional[list[MaterialGoal]] = None) -> "GoalMemory":
        """Load the store; if empty (fresh install or corrupt), seed defaults and persist."""
        gm = cls.load(path)
        if gm.is_empty():
            for g in (defaults if defaults is not None else default_goals()):
                gm.add_goal(g)
            gm.save(path)
        return gm
