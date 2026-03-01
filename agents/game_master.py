"""
Game Master agent — generates dynamic Quest objects using the Mistral API.

Supported modes (passed as `model` to generate_quest):
  "quest_0"                    — skip API, return the static default quest instantly
  "mistral-medium-latest"      — default: best quality for creative generation
  "labs-mistral-small-creative"— experimental Mistral Labs model, fast + creative
  "<finetuned-model-id>"       — your fine-tuned model (FINETUNED_MODEL_ID in .env)

Falls back to QUEST_0 on any API failure, always returns a valid Quest.
"""

import json
import os
import random
import time
import wandb
import boto3

from agents.npcs import NPCS, QUEST_0, Quest, QuestClue

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

MODEL_QUEST_0   = "quest_0"                          # no API call — instant static quest
MODEL_DEFAULT   = "mistral.mistral-large-3-675b-instruct"  # best quality
MODEL_CREATIVE  = "mistral.ministral-3-8b-instruct"        # fast, lightweight
MODEL_FINETUNED = os.getenv("FINETUNED_MODEL_ID", "").strip()  # set after fine-tuning

# NPCs available as clue-holders in the current Phaser scene.
# "inspector" is intentionally excluded — Inspecteur Lefèvre is the case-resolution NPC
# the player speaks to in order to submit a solution. He must never hold a clue or be the suspect.
ANCHOR_NPC_IDS = ["baker", "guard", "tavern_keeper", "cabaret_dancer", "artist"]

# Diverse premises so each generated quest feels different
_PREMISES = [
    "A renowned jewel thief has vanished along with the Countess de Méribel's sapphire brooch, last seen at the Opéra Garnier during a masked ball.",
    "The body of a forger has been found in the Seine, his pockets stuffed with counterfeit Exposition Universelle tickets.",
    "A blackmail letter threatens to expose a senator's affair unless a ransom is paid at the base of the Eiffel Tower at midnight.",
    "The head chef at the Grand Hôtel has been poisoned, and every member of the kitchen staff refuses to speak.",
    "A celebrated journalist has disappeared after claiming to have proof of corruption inside the Préfecture de Police.",
    "An art dealer is found dead in his gallery on the Rue Lafayette, and a rare Degas sketch has gone missing.",
    "A diplomat's coded dispatch has been stolen from the Foreign Ministry, threatening a delicate Franco-Russian treaty.",
    "A laundryman's body is discovered near the Gare du Nord — in his pocket, a list of names that no one will explain.",
    "A wealthy banker claims his safe was opened without a scratch; he suspects someone from his own household.",
    "The star soprano of the Opéra Comique has received death threats and a severed rose every morning for a week.",
    "A priceless medieval manuscript has been stolen from the Bibliothèque Nationale during a gala reception.",
    "A prominent physician is found strangled in his consulting room, his patient ledger torn to pieces.",
]

SYSTEM_PROMPT = """\
You are a mystery quest designer for "Les Mystères de Paris", a detective RPG set in
Belle Époque Paris (~1889–1911). Generate a complete mystery quest in JSON format
that is DIFFERENT from the default Mona Lisa theft.

## Rules

### Character selection
Select exactly 4 to 6 NPCs from the provided list. Choose NPCs whose roles and
knowledge make them PLAUSIBLE witnesses for this specific crime.

### Inspector — RESERVED ROLE (critical)
The `inspector` (Inspecteur Lefèvre) is the player's case-resolution NPC — the
character they speak to in order to submit their solution. He is NOT in the
available list, but even if he were:
  ✗ NEVER assign `inspector` as a clue holder in the clues array.
  ✗ NEVER name Inspecteur Lefèvre as the suspect in the solution.

### Clue chain — MOST IMPORTANT RULE
The clues MUST form a coherent investigative chain (relay race structure):
- Clue 1 (sequence=1): Entry point. The hint reveals something strange but does NOT
  solve the crime. It MUST implicitly point toward the NPC who holds clue 2.
- Each subsequent clue deepens the mystery and points toward the next NPC.
- Final clue (leads_to=null): Confirms or strongly implies the suspect, motive, method.

Each hint must:
  ✓ Be plausible for the NPC's role and personality
  ✓ Contain enough information to motivate visiting the NEXT NPC
  ✓ NOT directly name the culprit (except possibly the final clue)
  ✓ Feel like natural Belle Époque dialogue, not a list of facts

Chain rule: clue[i].leads_to == clue[i+1].npc_id. Last clue has leads_to = null.

### Secret vs hint
- `secret`: Full internal truth this NPC knows. 1–2 sentences. Specific.
- `hint`: What the NPC says in dialogue. Vaguer, atmospheric, semantically implies
  the next NPC (mention their location, profession, or a name fragment).

### Solution
Must be derivable by following the clue chain from start to finish.
- `suspect`: Full name and role (e.g. "Armand Chevalier, the locksmith")
- `motive`: Why they did it (specific, personal, period-appropriate)
- `method`: How they did it (specific, consistent with the clues)

### Red herrings
Include exactly 2 plausible false leads consistent with Belle Époque Paris.
They must NOT accidentally point to the real solution.

### Period accuracy
Paris ~1889–1911. Electric lights, Eiffel Tower, the Métro (from 1900), Dreyfus
Affair, Moulin Rouge. No anachronisms whatsoever.

## Output format — return ONLY valid JSON
{
  "quest_id": "quest_<slug>",
  "title": "French title, evocative, 3–7 words",
  "description": "2–3 sentence English description of the mystery as the player first hears it",
  "clues": [
    {
      "npc_id": "<one of the selected npc ids>",
      "secret": "<full internal truth, 1-2 sentences>",
      "hint": "<what the NPC says — atmospheric, implies next NPC>",
      "sequence": <integer starting at 1>,
      "leads_to": "<npc_id of next NPC, or null for the last clue>"
    }
  ],
  "solution": {
    "suspect": "<full name and role>",
    "motive": "<specific personal motive>",
    "method": "<specific method consistent with the clues>"
  },
  "red_herrings": ["<plausible false lead 1>", "<plausible false lead 2>"]
}

The clues array must be ordered by sequence (1, 2, 3...).
leads_to values must form a valid chain: clue[i].leads_to == clue[i+1].npc_id.
The last clue must have leads_to = null.\
"""


def _build_user_prompt(premise: str, npcs: list[dict]) -> str:
    npc_block = "\n".join(
        f"- id: {n['id']} | name: {n['name']} | role: {n['role']} | secret knowledge: {n['secret']}"
        for n in npcs
    )
    return f"PREMISE: {premise}\n\nAVAILABLE NPCs ({len(npcs)}):\n{npc_block}"


def _parse_quest(data: dict) -> Quest:
    """Validate and convert raw JSON dict into a Quest dataclass. Raises ValueError on bad data."""
    clues_raw = data.get("clues", [])
    if not (4 <= len(clues_raw) <= 6):
        raise ValueError(f"Expected 4–6 clues, got {len(clues_raw)}")

    valid_npc_ids = set(ANCHOR_NPC_IDS)
    clues: list[QuestClue] = []
    for c in clues_raw:
        npc_id = c["npc_id"]
        if npc_id not in valid_npc_ids:
            raise ValueError(f"Unknown npc_id in clue: {npc_id!r}")
        clues.append(
            QuestClue(
                npc_id=npc_id,
                secret=c["secret"],
                hint=c["hint"],
                sequence=int(c["sequence"]),
                leads_to=c.get("leads_to"),
            )
        )

    sorted_clues = sorted(clues, key=lambda c: c.sequence)

    for i, clue in enumerate(sorted_clues[:-1]):
        expected_next = sorted_clues[i + 1].npc_id
        if clue.leads_to != expected_next:
            raise ValueError(
                f"Chain broken at sequence {clue.sequence}: "
                f"leads_to={clue.leads_to!r} but next npc_id={expected_next!r}"
            )
    if sorted_clues[-1].leads_to is not None:
        raise ValueError(f"Last clue must have leads_to=null, got {sorted_clues[-1].leads_to!r}")

    solution = data.get("solution", {})
    if not all(solution.get(k) for k in ("suspect", "motive", "method")):
        raise ValueError("Incomplete solution — missing suspect, motive, or method")

    return Quest(
        quest_id=data.get("quest_id", "quest_generated"),
        title=data.get("title", "Mystère sans titre"),
        description=data.get("description", ""),
        clues=sorted_clues,
        solution=solution,
        red_herrings=data.get("red_herrings", []),
    )


def generate_quest(model: str | None = None) -> Quest:
    """
    Generate a fresh quest.

    model options:
      None / "mistral-medium-latest" — default, best quality
      "quest_0"                      — return static QUEST_0 instantly (no API)
      "labs-mistral-small-creative"  — experimental Mistral Labs model
      "<any model id>"               — use that Mistral model directly

    Always returns a valid Quest — falls back to QUEST_0 on any failure.
    """
    resolved = model or MODEL_DEFAULT
    if resolved == "finetuned":
        resolved = MODEL_FINETUNED

    start_time = time.perf_counter()
    retry_count = 0
    fallback_used = False
    generated_quest = None

    # Static fallback — no API call
    if resolved == MODEL_QUEST_0:
        print("[GameMaster] mode=quest_0 — returning QUEST_0")
        generated_quest = QUEST_0
        fallback_used = True
    else:
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not api_key:
            print("[GameMaster] No MISTRAL_API_KEY — returning QUEST_0")
            generated_quest = QUEST_0
            fallback_used = True
        else:
            print(f"[GameMaster] Generating quest with model={resolved}")
            region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            client = boto3.client("bedrock-runtime", region_name=region)
            premise = random.choice(_PREMISES)

            anchor_npcs = [
                {"id": npc.id, "name": npc.name, "role": npc.role, "secret": npc.secret}
                for npc_id, npc in NPCS.items()
                if npc_id in ANCHOR_NPC_IDS
            ]
            user_prompt = _build_user_prompt(premise, anchor_npcs)

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    retry_count = attempt - 1
                    response = client.converse(
                        modelId=resolved,
                        messages=[
                            {"role": "user", "content": [{"text": user_prompt}]}
                        ],
                        system=[{"text": SYSTEM_PROMPT}],
                        inferenceConfig={"temperature": 0.85, "maxTokens": 1500},
                    )
                    raw = response["output"]["message"]["content"][0]["text"]
                    data = json.loads(raw)
                    generated_quest = _parse_quest(data)
                    print(f"[GameMaster] Quest generated: {generated_quest.quest_id} — {generated_quest.title}")
                    break
                except Exception as e:
                    print(f"[GameMaster] Attempt {attempt}/{MAX_RETRIES} failed ({resolved}): {e}")

            if generated_quest is None:
                print(f"[GameMaster] All attempts failed ({resolved}) — returning QUEST_0 fallback")
                generated_quest = QUEST_0
                fallback_used = True

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # W&B Logging
    wandb_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_key:
        try:
            # We don't want to block the request on WandB initialization
            if not wandb.run:
                wandb.init(
                    project=os.getenv("WANDB_PROJECT", "les-mysteres-de-paris"),
                    name="quest-generation",
                    job_type="generation",
                    reinit=True
                )
            
            wandb.log({
                "quest_id": generated_quest.quest_id,
                "latency_ms": latency_ms,
                "retry_count": retry_count,
                "fallback_used": fallback_used,
                "model_used": resolved
            })
        except Exception as e:
            print(f"[GameMaster] W&B logging failed: {e}")

    return generated_quest
