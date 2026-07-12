"""
Specialist Agents — multi-agent scaffold plugin for COVAS:NEXT.

Milestone 1 (this session): prove data flows game -> deterministic detector ->
voice, with ZERO reasoning in between. No LLM, no agents, no orchestrator.

Pipeline wired here:
  1. sideeffect `_on_event` fires on material/engineer game events
     (the events that mutate the Materials projection — materials.py process()).
  2. runs the Layer-1 deterministic need-scorer (material_need_detector.py).
  3. logs the result.
  4. on a real deficit, dispatches a PluginEvent via helper.dispatch_event.
  5. register_event wires should_reply_check + prompt_generator so that event
     reaches the main assistant's should_reply gate and is eligible to voice.

Import discipline (so the module stays importable without the app's runtime deps):
  - `lib.PluginBase` and `lib.Event` import only stdlib -> safe at module scope.
  - `lib.PluginHelper` (pulls the openai chain) is TYPE_CHECKING-only.
  - `lib.Logger` (pulls pythonjsonlogger) is imported lazily via _safe_log.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from lib.PluginBase import PluginBase, PluginManifest
from lib.Event import PluginEvent, GameEvent

from .material_need_detector import TEST_GOAL, score_material_need, MaterialNeed

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
    """Milestone-1 deterministic detector + voiced round-trip."""

    # Namespaced event name (projections/events keyed by name globally — avoid collisions).
    EVENT_NAME = "SpecialistMaterialOpportunity"

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
        # Edge-trigger latch: dispatch once per unmet->met transition, not every event.
        self._notified = False

    def on_chat_start(self, helper: "PluginHelper"):
        self.helper = helper
        self._notified = False
        # Wire the voice path: this registers a should_reply handler on the Assistant
        # and a prompt handler on the PromptGenerator (PluginHelper.py:116-146).
        helper.register_event(
            self.EVENT_NAME,
            should_reply_check=self._should_reply,
            prompt_generator=self._render_prompt,
        )
        # Wire the detector to the event stream (PluginHelper.py:91-101).
        helper.register_sideeffect(self._on_event)
        _safe_log("info", "[SpecialistAgents] milestone-1 detector + voiced round-trip registered.")

    def on_chat_stop(self, helper: "PluginHelper"):
        _safe_log("info", "[SpecialistAgents] stopped.")

    # ----------------------------------------------------------------------- #
    # Layer-1 detector, driven by the event bus.                              #
    # ----------------------------------------------------------------------- #
    def _on_event(self, event, projected_states) -> None:
        # Ignore anything that isn't a relevant material/engineer game event.
        # (Also prevents recursion: our own PluginEvent is not a GameEvent.)
        if not isinstance(event, GameEvent):
            return
        if event.content.get("event") not in self.TRIGGER_EVENTS:
            return

        need = score_material_need(
            TEST_GOAL,
            projected_states.get("Materials"),
            projected_states.get("EngineerProgress"),
        )
        _safe_log("info", f"[SpecialistAgents] need-scorer -> {need.to_payload()}")

        if need.should_notify and not self._notified:
            self._notified = True
            self._dispatch_need(need)
        elif not need.should_notify:
            # deficit cleared -> re-arm the latch for next time.
            self._notified = False

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
