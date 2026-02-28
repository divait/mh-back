import os

import wandb
from fastapi import APIRouter, HTTPException
from mistralai import Mistral
from pydantic import BaseModel

import state as app_state
from agents.npcs import NPCS

router = APIRouter(prefix="/dialogue", tags=["dialogue"])

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
