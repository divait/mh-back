"""Shared in-memory state across routes (hackathon scope — no Redis needed)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.npcs import Quest

# session_id → active Quest object
active_quests: dict[str, "Quest"] = {}
