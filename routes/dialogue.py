import os
import time
from typing import AsyncGenerator

import httpx
import wandb
import boto3
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import state as app_state
from agents.npcs import NPCS, QUEST_0

router = APIRouter(prefix="/dialogue", tags=["dialogue"])

# ---------------------------------------------------------------------------
# ElevenLabs TTS Mapping
# ---------------------------------------------------------------------------
_VOICE_IDS = {
    # Default selection of voices from ElevenLabs pre-mades
    "inspector": "JBFqnCBsd6RMkjVDRZzb",      # George
    "baker": "hpp4J3VqNfWAUOO0d1Us",          # Bella
    "guard": "SOYHLrjzK2X1ezoPC6cr",          # Harry
    "tavern_keeper": "cjVigY5qzO86Huf0OWal",  # Eric
    "cabaret_dancer": "cgSgspJ2msm6clMCkdW9", # Jessica
    "artist": "TX3LPaxmHKxFdv7VOQHJ",         # Liam
    "default": "N2lVS1w4EtoT3dr4eOWO",        # Callum
}

@router.get("/tts")
async def get_tts(text: str, npc_id: str):
    """
    Fetch TTS audio from ElevenLabs for the given text and NPC.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")

    voice_id = _VOICE_IDS.get(npc_id, _VOICE_IDS["default"])
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?optimize_streaming_latency=3"
    
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    
    data = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            if response.status_code != 200:
                print(f"[TTS] API Error {response.status_code}: {response.text}")
                raise HTTPException(status_code=response.status_code, detail="TTS generation failed")
                
            return Response(content=response.content, media_type="audio/mpeg")
        except Exception as e:
            print(f"[TTS] Error for NPC '{npc_id}': {e}")
            raise HTTPException(status_code=500, detail="TTS request failed")

# ---------------------------------------------------------------------------
# W&B dialogue table — accumulates rows across turns, flushed periodically
# ---------------------------------------------------------------------------
_dialogue_table: wandb.Table | None = None


def _get_dialogue_table() -> wandb.Table:
    global _dialogue_table
    if _dialogue_table is None:
        _dialogue_table = wandb.Table(
            columns=[
                "session_id", "npc_id", "model_variant", "turn",
                "player_message", "npc_reply", "latency_ms", "reply_length",
                "anachronism_score", "period_score",
            ]
        )
    return _dialogue_table


# Modern words that should NOT appear in 1900 Paris — simple heuristic
_MODERN_WORDS = {
    "computer", "internet", "phone", "smartphone", "email", "website",
    "laptop", "television", "radio", "airplane", "nuclear", "plastic",
    "okay", "ok", "gonna", "wanna", "awesome", "totally", "literally",
    "selfie", "video", "app", "online", "digital", "robot", "laser",
}

_PERIOD_WORDS = {
    "monsieur", "madame", "mademoiselle", "sacré", "mon dieu", "voilà",
    "boulangerie", "gendarmerie", "sûreté", "belle époque", "exposition",
    "moulin", "montmartre", "louvre", "paris", "n'est-ce pas", "gaslamp",
    "carriage", "telegram", "telegraph", "affair", "dreyfus", "absinthe",
}


def _score_reply(reply: str) -> tuple[int, int]:
    """Return (anachronism_count, period_word_count) for the reply."""
    lower = reply.lower()
    anachronisms = sum(1 for w in _MODERN_WORDS if w in lower)
    period_hits = sum(1 for w in _PERIOD_WORDS if w in lower)
    return anachronisms, period_hits


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


def _get_client():
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region)


def _resolve_model(variant: str) -> str:
    if variant == "finetuned":
        model_id = os.getenv("FINETUNED_MODEL_ID", "").strip()
        if model_id:
            return model_id
        # Graceful fallback so the demo never breaks
    return "mistral.ministral-3-8b-instruct"


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
        completion = client.converse(
            modelId="mistral.ministral-3-8b-instruct",
            messages=[
                {"role": "user", "content": [{"text": user_prompt}]}
            ],
            system=[{"text": _INTRO_SYSTEM_PROMPT}],
            inferenceConfig={"maxTokens": 400, "temperature": 0.75},
        )
        raw = completion["output"]["message"]["content"][0]["text"].strip()
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

    bedrock_messages = []
    for msg in _history[history_key]:
        bedrock_messages.append({
            "role": "assistant" if msg["role"] == "assistant" else "user",
            "content": [{"text": msg["content"]}]
        })
    bedrock_messages.append({"role": "user", "content": [{"text": req.player_message}]})

    t0 = time.monotonic()
    try:
        response = client.converse(
            modelId=model,
            messages=bedrock_messages,
            system=[{"text": system_content}],
            inferenceConfig={"maxTokens": 200, "temperature": 0.85},
        )
        npc_reply = response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        err_str = str(e)
        print(f"[Dialogue] Bedrock error for model={model}: {err_str}")
        if "Operation not allowed" in err_str or "not authorized" in err_str.lower() or "AccessDeniedException" in err_str:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Bedrock model '{model}' is not accessible. "
                    "Please enable model access in the AWS Bedrock console "
                    "(Bedrock → Model access → Enable Mistral models)."
                ),
            )
        raise HTTPException(status_code=500, detail=f"Bedrock API error: {err_str}")
    latency_ms = int((time.monotonic() - t0) * 1000)

    # Persist turn in history (without the system prompt)
    _history[history_key].append({"role": "user", "content": req.player_message})
    _history[history_key].append({"role": "assistant", "content": npc_reply})
    turn_number = len(_history[history_key]) // 2  # each turn = 1 user + 1 assistant

    # W&B logging — lightweight, non-blocking
    _log_to_wandb(
        session_id=req.session_id,
        npc_id=req.npc_id,
        model_variant=req.model_variant,
        model_used=model,
        player_message=req.player_message,
        npc_reply=npc_reply,
        latency_ms=latency_ms,
        turn_number=turn_number,
    )

    return DialogueResponse(
        npc_id=npc.id,
        npc_name=npc.name,
        response=npc_reply,
        model_used=model,
    )


class SummarizeRequest(BaseModel):
    session_id: str
    npc_id: str
    npc_name: str
    # Full conversation so far: list of {role: "player"|"npc", content: str}
    conversation: list[dict]


class SummarizeResponse(BaseModel):
    summary: str


_SUMMARIZE_SYSTEM = """\
You are a sharp detective's assistant. Read the conversation between an investigator \
and a witness in Belle Époque Paris (1900). Extract ONLY the factual investigative \
clues the witness revealed — names, locations, times, motives, or suspicious details. \
Ignore pleasantries, flirting, jokes, or anything that does not help solve the case.

Write a single concise paragraph (1–2 sentences, max 80 words) in the style of an \
investigator's field note. Begin with the witness's name. Be specific. \
If the witness revealed NOTHING useful, reply with exactly: NO_CLUE\
"""


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_clue(req: SummarizeRequest) -> SummarizeResponse:
    """
    Use Mistral to distil the NPC conversation into a crisp investigator's note.
    Returns {summary} — a 1-2 sentence detective's journal entry.
    Falls back to a generic note if the AI fails.
    """
    if not req.conversation:
        return SummarizeResponse(summary="")

    # Format conversation for the prompt
    transcript_lines = []
    for turn in req.conversation:
        role_label = "Investigator" if turn.get("role") == "player" else req.npc_name
        transcript_lines.append(f"{role_label}: {turn.get('content', '')}")
    transcript = "\n".join(transcript_lines)

    try:
        client = _get_client()
        completion = client.converse(
            modelId="mistral.ministral-3-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": f"Witness: {req.npc_name}\n\nConversation:\n{transcript}\n\nWrite the field note now."}],
                }
            ],
            system=[{"text": _SUMMARIZE_SYSTEM}],
            inferenceConfig={"maxTokens": 120, "temperature": 0.3},
        )
        raw = completion["output"]["message"]["content"][0]["text"].strip()
        if raw == "NO_CLUE" or not raw:
            return SummarizeResponse(summary="")
        return SummarizeResponse(summary=raw)
    except Exception:
        return SummarizeResponse(summary=f"{req.npc_name} shared information that may be relevant to the investigation.")


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
    latency_ms: int = 0,
    turn_number: int = 1,
) -> None:
    try:
        if wandb.run is None:
            return

        anachronism_count, period_score = _score_reply(npc_reply)

        # --- Numeric metrics for W&B charts ---
        wandb.log(
            {
                # Core performance metrics (these become chart lines)
                "reply_length": len(npc_reply),
                "latency_ms": latency_ms,
                "anachronism_count": anachronism_count,
                "period_authenticity_score": period_score,
                "turn_number": turn_number,
                # Model variant as 0/1 so it can be plotted
                "is_finetuned": int(model_variant == "finetuned"),
            }
        )

        # --- Dialogue table (browseable in W&B) ---
        table = _get_dialogue_table()
        table.add_data(
            session_id,
            npc_id,
            model_variant,
            turn_number,
            player_message[:500],   # truncate for W&B column limits
            npc_reply[:500],
            latency_ms,
            len(npc_reply),
            anachronism_count,
            period_score,
        )
        # Log updated table snapshot so W&B UI stays fresh
        wandb.log({"dialogue_log": table})

    except Exception:
        pass  # Never let W&B crash the game
