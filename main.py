import os
from contextlib import asynccontextmanager

import wandb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    return {"status": "ok", "npcs": ["baker", "guard", "tavern_keeper"]}
