"""
Quest endpoints:
  GET  /quest/models              → list available generation models
  GET  /quest                     → default quest (QUEST_0), no session needed
  POST /quest/generate            → generate a fresh quest for a session
  GET  /quest/session/{session_id}→ get active quest for a session
  POST /quest/solve               → submit a solution attempt
"""

from fastapi import APIRouter
from pydantic import BaseModel

import state as app_state
from agents.game_master import (
    MODEL_CREATIVE,
    MODEL_DEFAULT,
    MODEL_FINETUNED,
    MODEL_QUEST_0,
    generate_quest as gm_generate,
)
from agents.npcs import QUEST_0, Quest

router = APIRouter(prefix="/quest", tags=["quest"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    session_id: str
    model: str | None = None  # None → MODEL_DEFAULT; "quest_0" → static QUEST_0


class SolveRequest(BaseModel):
    session_id: str
    suspect: str
    motive: str
    method: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _public_view(quest: Quest) -> dict:
    """Return quest data safe for the frontend — no NPC secrets exposed."""
    return {
        "quest_id": quest.quest_id,
        "title": quest.title,
        "description": quest.description,
        "solution": quest.solution,
        "red_herrings": quest.red_herrings,
        "clues": sorted(
            [
                {
                    "npc_id": c.npc_id,
                    "secret": c.secret,
                    "hint": c.hint,
                    "sequence": c.sequence,
                    "leads_to": c.leads_to,
                }
                for c in quest.clues
            ],
            key=lambda c: c["sequence"],
        ),
    }


def _norm(s: str) -> str:
    return s.strip().lower()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models() -> dict:
    """
    Return the available quest-generation model options.
    The frontend uses this to populate a model picker.
    """
    models = [
        {
            "id": MODEL_QUEST_0,
            "label": "Default Quest (no generation)",
            "description": "Returns the built-in static quest instantly. No API call.",
            "available": True,
        },
        {
            "id": MODEL_DEFAULT,
            "label": "Mistral Medium (default)",
            "description": "Best quality creative generation.",
            "available": True,
        },
        {
            "id": MODEL_CREATIVE,
            "label": "Mistral Small Creative (Labs)",
            "description": "Experimental Mistral Labs model — fast and imaginative.",
            "available": True,
        },
        {
            "id": MODEL_FINETUNED or "finetuned_not_set",
            "label": "Fine-tuned Model",
            "description": "Your custom fine-tuned quest-generation model.",
            "available": bool(MODEL_FINETUNED),
        },
    ]
    return {"models": models, "default": MODEL_DEFAULT}


@router.get("/")
async def default_quest() -> dict:
    """Always returns QUEST_0 — useful for the tutorial / demo flow."""
    return _public_view(QUEST_0)


@router.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """
    Ask the Game Master to generate a fresh quest for this session.

    Pass `model` in the request body to choose the generation model:
      "quest_0"                    → static default quest, no API call
      "mistral-medium-latest"      → default (also used when model is omitted)
      "labs-mistral-small-creative"→ experimental Mistral Labs model
      "<finetuned-model-id>"       → your fine-tuned model

    Falls back to QUEST_0 if generation fails. The quest is stored server-side
    so /quest/solve and dialogue injection use the same quest for this session.
    """
    quest = gm_generate(model=req.model)
    app_state.active_quests[req.session_id] = quest
    return {**_public_view(quest), "model_used": req.model or "mistral-medium-latest"}


@router.get("/session/{session_id}")
async def get_session_quest(session_id: str) -> dict:
    """Return the active quest for a session, or QUEST_0 if none set."""
    quest = app_state.active_quests.get(session_id, QUEST_0)
    return _public_view(quest)


@router.post("/solve")
async def solve(req: SolveRequest) -> dict:
    """
    Validate the player's solution against the active quest.

    Matching is flexible: the player's answer just needs to be *contained in*
    the correct solution string (case-insensitive).

    Returns:
      correct       — True if suspect correct + at least motive OR method correct
      suspect_match — whether suspect is correct
      motive_match  — whether motive is correct
      method_match  — whether method is correct
      solution      — full solution revealed only on win
    """
    quest = app_state.active_quests.get(req.session_id, QUEST_0)
    sol = quest.solution

    suspect_match = _norm(req.suspect) in _norm(sol.get("suspect", ""))
    motive_match  = _norm(req.motive)  in _norm(sol.get("motive",  ""))
    method_match  = _norm(req.method)  in _norm(sol.get("method",  ""))

    correct = suspect_match and (motive_match or method_match)

    return {
        "correct":        correct,
        "suspect_match":  suspect_match,
        "motive_match":   motive_match,
        "method_match":   method_match,
        "solution":       sol if correct else None,
    }
