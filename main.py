import os
from contextlib import asynccontextmanager

import wandb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.npcs import NPC_COLORS, NPCS, PERSONS, QUEST_0
from routes.dialogue import router as dialogue_router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init W&B run for this server session (tracks all live dialogue calls)
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "les-mysteres-de-paris"),
        name="live-dialogue-session",
        tags=["dialogue", "live"],
        config={
            "baseline_model": "mistral-small-latest",  # "labs-mistral-small-creative",
            "finetuned_model": os.getenv("FINETUNED_MODEL_ID", "pending"),
        },
        reinit=True,
    )
    yield
    wandb.finish()


app = FastAPI(
    title="Les Mystères de Paris — AI Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dialogue_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "npcs": list(NPCS.keys())}


@app.get("/npcs")
async def list_npcs() -> dict:
    """Agent NPCs and Person NPCs with full metadata for frontend zone building."""
    agents = [
        {
            "id": n.id,
            "name": n.name,
            "role": n.role,
            "location": n.location,
            "category": n.category,
        }
        for n in NPCS.values()
    ]
    persons = [
        {"id": p.id, "name": p.name, "greeting": p.greeting} for p in PERSONS.values()
    ]
    return {
        "agents": agents,
        "persons": persons,
        "colors": NPC_COLORS,
    }


@app.get("/quest")
async def get_quest() -> dict:
    """Active quest — MVP always returns QUEST_0."""
    return {
        "quest_id": QUEST_0.quest_id,
        "title": QUEST_0.title,
        "description": QUEST_0.description,
        "solution": QUEST_0.solution,
        "red_herrings": QUEST_0.red_herrings,
        # Expose only hints to the frontend, never secrets
        "clues": [{"npc_id": c.npc_id, "hint": c.hint} for c in QUEST_0.clues],
    }
