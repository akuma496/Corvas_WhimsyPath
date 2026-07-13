"""
Specialist Agents — exploration specialist plugin for COVAS:NEXT.

Layers implemented so far:
  - Layer 1 (deterministic, no LLM): material_need_detector.py — need-scorer.
  - Layer 2 (state, no LLM):         goal_memory.py — persistent goal store +
                                     per-goal notify latch.

Pipeline wired here:
  1. sideeffect `_on_event` fires on material/engineer game events
     (the events that mutate the Materials projection — materials.py process()).
  2. for each goal in Goal Memory, runs the deterministic need-scorer.
  3. logs the result.
  4. on a real deficit (edge-triggered per goal), dispatches a PluginEvent.
  5. register_event wires should_reply_check + prompt_generator so the event
     reaches the assistant's should_reply gate and is eligible to voice.

NOT yet implemented (deferred, in dependency order): natural-language goal parse
(the one legitimate small-LLM entry point), additional detectors, and the
Layer-3 Reactive/Planning agents + orchestrator.

Import discipline (so the module stays importable without the app's runtime deps):
  - `lib.PluginBase` and `lib.Event` import only stdlib -> safe at module scope.
  - `lib.PluginHelper` (pulls the openai chain) is TYPE_CHECKING-only.
  - `lib.Logger` (pulls pythonjsonlogger) is imported lazily via _safe_log.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from lib.PluginBase import PluginBase, PluginManifest
from lib.Event import PluginEvent, GameEvent

from .material_need_detector import score_material_need, MaterialNeed
from .goal_memory import GoalMemory

if TYPE_CHECKING:
    from lib.PluginHelper import PluginHelper


def _safe_log(level: str, message: str) -> None:
    """Log through the app logger at runtime; fall back to stderr when the app's
    logging deps aren't installed (e.g. offline verification harness)."""
    try:
        from lib.Logger import log
        log(level, message)
    except Exception:
        print(f"[SpecialistAgents:{level}] {message}", file=sys.stderr, flush=True)


class SpecialistAgentsPlugin(PluginBase):
    """Deterministic detector + Goal Memory + voiced round-trip (no LLM yet)."""

    # Namespaced event name (projections/events keyed by name globally — avoid collisions).
    EVENT_NAME = "SpecialistMaterialOpportunity"

    GOAL_STORE_FILENAME = "goal_memory.json"

    # Game events that mutate the Materials / EngineerProgress projections.
    # Source: src/lib/projections/materials.py process() handles Materials,
    # MaterialCollected, MaterialDiscarded, MaterialTrade, EngineerCraft, Synthesis;
    # EngineerProgress updates the engineer projection.
    TRIGGER_EVENTS = frozenset({
        "Materials",
        "MaterialCollected",
        "MaterialDiscarded",
        "MaterialTrade",
        "EngineerCraft",
        "Synthesis",
        "EngineerProgress",
    })

    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)
        self.helper: "PluginHelper | None" = None
        self.goals: GoalMemory = GoalMemory()
        # Path to the persisted goal store; None disables persistence (used in tests).
        self._goal_path: "str | None" = None

    def on_chat_start(self, helper: "PluginHelper"):
        self.helper = helper
        # Load (or seed) the persistent goal store from the plugin data folder.
        self._goal_path = os.path.join(
            helper.get_plugin_data_path(self.plugin_manifest), self.GOAL_STORE_FILENAME
        )
        self.goals = GoalMemory.load_or_seed(self._goal_path)
        # Wire the voice path: registers a should_reply handler on the Assistant
        # and a prompt handler on the PromptGenerator (PluginHelper.py:116-146).
        helper.register_event(
            self.EVENT_NAME,
            should_reply_check=self._should_reply,
            prompt_generator=self._render_prompt,
        )
        # Wire the detector to the event stream (PluginHelper.py:91-101).
        helper.register_sideeffect(self._on_event)
        _safe_log(
            "info",
            f"[SpecialistAgents] registered; {len(self.goals)} goal(s) loaded from Goal Memory.",
        )

    def on_chat_stop(self, helper: "PluginHelper"):
        _safe_log("info", "[SpecialistAgents] stopped.")

    # ----------------------------------------------------------------------- #
    # Layer-1 detector over Layer-2 goals, driven by the event bus.           #
    # ----------------------------------------------------------------------- #
    def _on_event(self, event, projected_states) -> None:
        # Ignore anything that isn't a relevant material/engineer game event.
        # (Also prevents recursion: our own PluginEvent is not a GameEvent.)
        if not isinstance(event, GameEvent):
            return
        if event.content.get("event") not in self.TRIGGER_EVENTS:
            return

        materials_state = projected_states.get("Materials")
        engineer_state = projected_states.get("EngineerProgress")

        latch_changed = False
        for goal in self.goals.list_goals():
            key = goal.material_internal_name
            need = score_material_need(goal, materials_state, engineer_state)
            _safe_log("info", f"[SpecialistAgents] need-scorer -> {need.to_payload()}")

            if need.should_notify and not self.goals.is_notified(key):
                self.goals.mark_notified(key)
                latch_changed = True
                self._dispatch_need(need)
            elif not need.should_notify and self.goals.is_notified(key):
                # deficit cleared -> re-arm the latch for next time.
                self.goals.reset_notified(key)
                latch_changed = True

        if latch_changed and self._goal_path:
            self.goals.save(self._goal_path)

    def _dispatch_need(self, need: MaterialNeed) -> None:
        event = PluginEvent(
            plugin_event_name=self.EVENT_NAME,
            plugin_event_content=need.to_payload(),
        )
        _safe_log("info", f"[SpecialistAgents] dispatching {self.EVENT_NAME} (deficit={need.deficit}).")
        assert self.helper is not None, "dispatch before on_chat_start"
        self.helper.dispatch_event(event)

    # ----------------------------------------------------------------------- #
    # Voice gate. register_event has already filtered by isinstance PluginEvent#
    # and plugin_event_name before these are called (PluginHelper.py:123-146).#
    # ----------------------------------------------------------------------- #
    def _should_reply(self, event) -> bool:
        """Fail closed: only eligible to voice on a real, quantified deficit."""
        payload = getattr(event, "plugin_event_content", None) or {}
        try:
            return bool(payload.get("should_notify")) and int(payload.get("deficit", 0)) > 0
        except (TypeError, ValueError):
            return False

    def _render_prompt(self, event) -> str:
        """STRICTLY grounded: every value comes from the deterministic detector
        payload. When a reasoning agent replaces this later, it MUST be instructed
        to use only these fields and to say nothing if they're absent."""
        p = getattr(event, "plugin_event_content", None) or {}
        rank = p.get("engineer_rank")
        rank_str = f" (rank {rank})" if rank is not None else ""
        engineer_status = "unlocked" if p.get("engineer_unlocked") else "not unlocked"
        return (
            "[Materials opportunity — deterministic] "
            f"You still need {p.get('deficit')} more {p.get('material_display_name')} "
            f"(have {p.get('have')} of {p.get('required')} for {p.get('purpose_short')}). "
            f"Engineer {p.get('engineer_name')}: {engineer_status}{rank_str}. "
            "Only mention this if it is genuinely useful right now."
        )
