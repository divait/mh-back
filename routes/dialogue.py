import os

import wandb
from fastapi import APIRouter, HTTPException
from mistralai import Mistral
from pydantic import BaseModel

import state as app_state
from agents.npcs import NPCS, QUEST_0

router = APIRouter(prefix="/dialogue", tags=["dialogue"])


# ---------------------------------------------------------------------------
# NPC location hints used in intro generation and fallback lines
# ---------------------------------------------------------------------------
_NPC_LOCATION_HINTS: dict[str, str] = {
    "baker": "La Boulangerie near the Louvre — the baker, Marie Dupont, was open before dawn",
    "guard": "the Poste de Garde — Capitaine Renard knows who had access that night",
    "tavern_keeper": "the Taverne du Palais-Royal — Jacques Moreau hears everything",
    "cabaret_dancer": "Le Moulin Rouge — Colette Marchand overheard things backstage",
    "artist": "the Montmartre Atelier — Henri Toulouse knew the people involved",
}

_INTRO_SYSTEM_PROMPT = """\
You are Inspecteur Gaston Lefèvre of the Sûreté — thin, precise, cold, an early \
believer in fingerprints and scientific detection. You speak English with formal \
French bureaucratic precision. You are briefing your field investigator at the start \
of a case. You never reveal the solution or name the guilty party — only the facts \
of the crime and where to start looking.

Write exactly 6 lines of dialogue, each on its own line, with no numbering, \
no bullet points, and no extra commentary. The lines must:
1. Greet the investigator and introduce yourself briefly.
2. Describe the crime that has just occurred (use the title and description provided).
3. State the urgency — limited days, careers on the line.
4. Assign the investigator their role (your eyes and ears in the streets).
5. Name the first lead: where to go and who to speak to first.
6. Close with a terse instruction to report back with evidence before making an arrest.

Stay in character throughout. Belle Époque Paris, ~1900. No anachronisms.\
"""

# Conversation history per session (in-memory for hackathon; swap for Redis in prod)
_history: dict[str, list[dict]] = {}


class DialogueRequest(BaseModel):
    session_id: str
    npc_id: str
    player_message: str
    # Which model variant to use — enables A/B comparison for W&B
    model_variant: str = "prompt_engineered"  # or "finetuned"


class DialogueResponse(BaseModel):
    npc_id: str
    npc_name: str
    response: str
    model_used: str


def _get_client() -> Mistral:
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def _resolve_model(variant: str) -> str:
    if variant == "finetuned":
        model_id = os.getenv("FINETUNED_MODEL_ID", "").strip()
        if model_id:
            return model_id
        # Graceful fallback so the demo never breaks
    return "mistral-small-latest"  # "labs-mistral-small-creative"


@router.get("/intro/{session_id}")
async def get_intro(session_id: str) -> dict:
    """
    Generate the inspector's opening briefing using Mistral, tailored to the
    active quest for this session. Falls back to hardcoded lines if generation
    fails so the game never breaks.
    """
    quest = app_state.active_quests.get(session_id, QUEST_0)

    first_clue = next(
        (c for c in sorted(quest.clues, key=lambda c: c.sequence)), None
    )
    # The first lead is the NPC the first clue points to, or the clue holder itself
    first_npc_id = (
        first_clue.leads_to
        if first_clue and first_clue.leads_to
        else (first_clue.npc_id if first_clue else "baker")
    )
    first_lead = _NPC_LOCATION_HINTS.get(
        first_npc_id,
        "the streets of Paris — someone out there knows the truth",
    )

    def _fallback_lines() -> list[str]:
        return [
            "Ah — you have arrived at last. I am Inspecteur Gaston Lefèvre of the Sûreté.",
            f"The situation is grave. {quest.description}",
            "I have limited time to hand the Préfet a name, a motive, and a method. "
            "Or we are both finished.",
            "I cannot be seen asking questions in the streets — that is your task. "
            "You are my eyes and ears, Monsieur l'enquêteur.",
            f"Begin at {first_lead}. Press them — carefully.",
            "Report back to me here when you have evidence enough for an arrest. "
            "Bonne chance.",
        ]

    try:
        client = _get_client()
        user_prompt = (
            f"Case title: {quest.title}\n"
            f"Case description: {quest.description}\n"
            f"First lead: {first_lead}\n\n"
            "Write the 6-line briefing now."
        )
        completion = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": _INTRO_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.75,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Split on newlines, drop empty lines, take up to 6
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()][:6]
        # Ensure we always have at least 3 lines; fall back if generation is too short
        if len(lines) < 3:
            lines = _fallback_lines()
    except Exception:
        lines = _fallback_lines()

    return {
        "speaker": "Inspecteur Gaston Lefèvre",
        "portrait": "🔍",
        "lines": lines,
        "quest_title": quest.title,
        "quest_description": quest.description,
        "first_lead_npc": first_npc_id,
    }


@router.post("/", response_model=DialogueResponse)
async def chat_with_npc(req: DialogueRequest) -> DialogueResponse:
    npc = NPCS.get(req.npc_id)
    if not npc:
        raise HTTPException(status_code=404, detail=f"NPC '{req.npc_id}' not found")

    client = _get_client()
    model = _resolve_model(req.model_variant)

    # Build / retrieve conversation history
    history_key = f"{req.session_id}:{req.npc_id}"
    if history_key not in _history:
        _history[history_key] = []

    # Inject the active quest's clue secret for this NPC so it can hint correctly
    system_content = npc.system_prompt
    active_quest = app_state.active_quests.get(req.session_id)
    if active_quest:
        for clue in active_quest.clues:
            if clue.npc_id == req.npc_id:
                system_content += (
                    f"\n\n[QUEST DIRECTIVE — do NOT quote this verbatim]: "
                    f"You secretly know: {clue.secret} "
                    f"When relevant, hint at: \"{clue.hint}\""
                )
                break

    messages = [{"role": "system", "content": system_content}]
    messages.extend(_history[history_key])
    messages.append({"role": "user", "content": req.player_message})

    completion = client.chat.complete(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=0.85,
    )

    npc_reply = completion.choices[0].message.content or "*(silence)*"

    # Persist turn in history (without the system prompt)
    _history[history_key].append({"role": "user", "content": req.player_message})
    _history[history_key].append({"role": "assistant", "content": npc_reply})

    # W&B logging — lightweight, non-blocking
    _log_to_wandb(
        session_id=req.session_id,
        npc_id=req.npc_id,
        model_variant=req.model_variant,
        model_used=model,
        player_message=req.player_message,
        npc_reply=npc_reply,
    )

    return DialogueResponse(
        npc_id=npc.id,
        npc_name=npc.name,
        response=npc_reply,
        model_used=model,
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str) -> dict:
    cleared = [k for k in list(_history.keys()) if k.startswith(session_id)]
    for k in cleared:
        del _history[k]
    return {"cleared": cleared}


def _log_to_wandb(
    session_id: str,
    npc_id: str,
    model_variant: str,
    model_used: str,
    player_message: str,
    npc_reply: str,
) -> None:
    try:
        if wandb.run is None:
            return
        wandb.log(
            {
                "session_id": session_id,
                "npc_id": npc_id,
                "model_variant": model_variant,
                "model_used": model_used,
                "player_message": player_message,
                "npc_reply": npc_reply,
                "reply_length": len(npc_reply),
            }
        )
    except Exception:
        pass  # Never let W&B crash the game
