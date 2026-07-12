"""
Layer-1 deterministic detector: Materials / Engineering need-scorer.

Pure arithmetic + dictionary lookups. NO LLM, NO network, NO game APIs.
Reads the shapes produced by the core projections:
  - Materials        (src/lib/projections/materials.py:32  -> Raw/Manufactured/Encoded: list[MaterialEntry{Name,Count,Name_Localised}])
  - EngineerProgress (src/lib/projections/engineer_progress.py:14 -> Engineers: list[EngineerState{Engineer,EngineerID,Progress,Rank,RankProgress}])

Both are passed to sideeffects/actions inside `projected_states`, keyed by the
projection CLASS NAME (EventManager.py:327-329), so the keys are literally
"Materials" and "EngineerProgress". Values are Pydantic models at runtime; this
module accepts either a Pydantic model (via .model_dump()) or a plain dict, so it
is testable offline without importing the projection classes (which pull in the
openai-dependent Config module).

This module is intentionally free of any `lib.*` import so it stays importable in
environments where the app's runtime deps (openai, pythonjsonlogger) are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# get_state_dict-equivalent (mirrors src/lib/projections/common.py:8-28,       #
# minus the LatestEventState branch which is irrelevant here).                 #
# --------------------------------------------------------------------------- #
def _as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


# --------------------------------------------------------------------------- #
# Hardcoded TEST goal for milestone 1.                                         #
# NOTE: the required_count and engineer mapping are TEST CONSTANTS. Real       #
# blueprint -> material-requirement resolution (via blueprint_finder /         #
# engineer_finder in actions_web.py) is explicitly OUT OF SCOPE this session.  #
# The material itself is real: internal "scrambledemissiondata" ->             #
# "Exceptional Scrambled Emission Data" (Encoded), src/assets/materials.json:625.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaterialGoal:
    material_internal_name: str
    material_display_name: str
    category: str                 # "Raw" | "Manufactured" | "Encoded"
    required_count: int
    engineer_name: str
    required_engineer_rank: int
    purpose_short: str


TEST_GOAL = MaterialGoal(
    material_internal_name="scrambledemissiondata",
    material_display_name="Exceptional Scrambled Emission Data",
    category="Encoded",
    required_count=3,
    engineer_name="Felicity Farseer",
    required_engineer_rank=5,
    purpose_short="grade 5 FSD (Increased Range)",
)


@dataclass
class MaterialNeed:
    """Deterministic scoring result. Every field is arithmetic or a lookup."""
    material_display_name: str
    purpose_short: str
    have: int
    required: int
    deficit: int
    need_score: float             # deficit / required, clamped to [0, 1]
    engineer_name: str
    engineer_unlocked: bool
    engineer_rank: Optional[int]
    engineer_ready: bool
    should_notify: bool           # fail-closed: only true on a real, quantified deficit

    def to_payload(self) -> dict:
        """Flat, JSON-safe payload carried in PluginEvent.plugin_event_content."""
        return {
            "material_display_name": self.material_display_name,
            "purpose_short": self.purpose_short,
            "have": self.have,
            "required": self.required,
            "deficit": self.deficit,
            "need_score": self.need_score,
            "engineer_name": self.engineer_name,
            "engineer_unlocked": self.engineer_unlocked,
            "engineer_rank": self.engineer_rank,
            "engineer_ready": self.engineer_ready,
            "should_notify": self.should_notify,
        }


def get_material_count(
    materials_state: Any,
    internal_name: str,
    display_name: str | None = None,
    category: str | None = None,
) -> int:
    """Current owned count of a material. Matches on internal Name OR Name_Localised,
    case-insensitively. Searches the given category bucket, or all three if None."""
    state = _as_dict(materials_state)
    buckets = [category] if category else ["Raw", "Manufactured", "Encoded"]
    target_internal = internal_name.lower()
    target_display = (display_name or "").lower()
    for bucket in buckets:
        for entry in state.get(bucket, []) or []:
            e = _as_dict(entry)
            name = str(e.get("Name", "")).lower()
            localised = str(e.get("Name_Localised") or "").lower()
            if name == target_internal or (target_display and localised == target_display):
                try:
                    return int(e.get("Count", 0) or 0)
                except (TypeError, ValueError):
                    return 0
    return 0


def get_engineer(engineer_state: Any, engineer_name: str) -> dict:
    """Return the EngineerState dict for a named engineer, or {} if not present."""
    state = _as_dict(engineer_state)
    target = engineer_name.lower()
    for entry in state.get("Engineers", []) or []:
        e = _as_dict(entry)
        if str(e.get("Engineer", "")).lower() == target:
            return e
    return {}


def score_material_need(
    goal: MaterialGoal,
    materials_state: Any,
    engineer_state: Any,
) -> MaterialNeed:
    """Deterministic need-score. No judgement, no LLM."""
    have = get_material_count(
        materials_state, goal.material_internal_name, goal.material_display_name, goal.category
    )
    required = max(0, int(goal.required_count))
    deficit = max(0, required - have)
    need_score = (deficit / required) if required > 0 else 0.0

    eng = get_engineer(engineer_state, goal.engineer_name)
    unlocked = str(eng.get("Progress", "")) == "Unlocked"
    rank = eng.get("Rank")
    rank_int = int(rank) if isinstance(rank, int) else None
    engineer_ready = unlocked and (rank_int or 0) >= goal.required_engineer_rank

    return MaterialNeed(
        material_display_name=goal.material_display_name,
        purpose_short=goal.purpose_short,
        have=have,
        required=required,
        deficit=deficit,
        need_score=round(need_score, 4),
        engineer_name=goal.engineer_name,
        engineer_unlocked=unlocked,
        engineer_rank=rank_int,
        engineer_ready=engineer_ready,
        # Fail-closed: only flag when there is a real, quantified shortfall.
        should_notify=deficit > 0,
    )
