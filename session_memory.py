"""
Session memory for the WebAgent — stores structured per-step records
and provides formatted context for prompt injection.
"""

from __future__ import annotations


class SessionMemory:
    """Lightweight in-memory store for agent step records."""

    def __init__(self):
        self._entries: list[dict] = []

    def add_entry(
        self,
        step: int,
        action: str,
        predicted_state: str | None = None,
        actual_summary: str | None = None,
        was_safe: bool = True,
        reflexion_note: str | None = None,
    ) -> None:
        self._entries.append({
            "step": step,
            "action": action,
            "predicted_state": predicted_state,
            "actual_summary": actual_summary,
            "was_safe": was_safe,
            "reflexion_note": reflexion_note,
        })

    def update_last(self, **kwargs) -> None:
        """Update fields on the most recent entry (e.g., after reflexion runs)."""
        if self._entries:
            self._entries[-1].update(kwargs)

    def get_recent(self, n: int = 5) -> list[dict]:
        return self._entries[-n:]

    def get_corrective_rules(self) -> list[str]:
        """Extract all non-None reflexion notes from history."""
        return [
            e["reflexion_note"]
            for e in self._entries
            if e.get("reflexion_note")
        ]

    def format_for_prompt(self, max_entries: int = 5) -> str:
        """Format recent memory as a string block for prompt injection.

        Returns empty string if no noteworthy entries exist.
        """
        recent = self.get_recent(max_entries)
        if not recent:
            return ""

        # Only include entries that have reflexion notes or safety warnings
        noteworthy = [
            e for e in recent
            if e.get("reflexion_note") or not e.get("was_safe", True)
        ]
        if not noteworthy and not self.get_corrective_rules():
            return ""

        lines = ["Session notes:"]
        for e in noteworthy:
            parts = [f"  Step {e['step']}: {e['action']}"]
            if e.get("reflexion_note"):
                parts.append(f"    LESSON: {e['reflexion_note']}")
            if not e.get("was_safe", True):
                parts.append(f"    WARNING: This action was flagged as unsafe")
            lines.extend(parts)

        rules = self.get_corrective_rules()
        if rules:
            lines.append("\nLearned rules (avoid repeating these mistakes):")
            for i, rule in enumerate(rules, 1):
                lines.append(f"  {i}. {rule}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)
