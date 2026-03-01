# Les Mystères de Paris — Backend

The API server for **Les Mystères de Paris: AI Chronicles** — a mystery-RPG set in Belle Époque Paris (1889–1911), powered by fine-tuned Mistral AI NPCs.

> Hackathon project (24h) — Built with FastAPI, Mistral AI, and Weights & Biases.

## Related Repositories

| Project | Repo |
|---------|------|
| Frontend (React + Phaser 3) | [github.com/divait/mh-front](https://github.com/divait/mh-front) |
| Fine-tuning (W&B + Mistral) | [github.com/divait/mh-fine-tuning](https://github.com/divait/mh-fine-tuning) |

## Overview

- **FastAPI** — REST API consumed by the Phaser game client
- **Mistral AI** — powers NPC dialogue (agent-style, with secrets and knowledge graphs) and quest generation (structured JSON output)
- **Weights & Biases** — tracks every live dialogue call for experiment comparison (baseline vs prompted vs fine-tuned)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/npcs` | Returns all NPCs (agents + persons) and their UI colour categories |
| `POST` | `/quest/generate` | Generates a unique mystery quest as structured JSON |
| `POST` | `/dialogue/{npc_id}` | Sends a player message to an NPC agent and returns its reply |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | FastAPI 0.115 |
| Runtime | Python 3.13+ via `uv` |
| AI | Mistral API (`mistral-small-latest` + optional fine-tuned model) |
| Experiment tracking | Weights & Biases |
| HTTP client | httpx |
| Env management | python-dotenv |

## Project Structure

```
backend/
  main.py           # FastAPI app, CORS, W&B lifespan init
  state.py          # In-memory game state (active quest, clue progress)
  agents/
    npcs.py         # NPC definitions — secrets, knowledge, system prompts
    game_master.py  # Quest generation logic (Mistral structured output)
  routes/
    dialogue.py     # POST /dialogue/{npc_id} — NPC chat handler
    quest.py        # POST /quest/generate — quest generation handler
```

## Quick Start

### Prerequisites

- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) — `pip install uv`
- A [Mistral AI API key](https://console.mistral.ai/)
- A [Weights & Biases account](https://wandb.ai/) (optional but recommended)

### Install & Run

```bash
cp .env.example .env   # then fill in your keys (see below)
uv sync
uv run uvicorn main:app --reload
```

API available at [http://localhost:8000](http://localhost:8000).
Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

## Environment Variables

Create a `.env` file at the root of this directory:

```env
MISTRAL_API_KEY=...
WANDB_API_KEY=...
WANDB_PROJECT=les-mysteres-de-paris

# Set this after a fine-tuning job completes (see mh-fine-tuning)
FINETUNED_MODEL_ID=
```

## NPC Agents

Six Mistral-powered NPCs — each has a `system_prompt`, a `secret` they are hiding, and `knowledge` of other characters. The dialogue route injects the current quest context so every NPC's answers are consistent with the generated mystery.

| ID | Name | Role |
|----|------|------|
| `baker` | Marie Dupont | Boulangère near the Louvre |
| `guard` | Capitaine Renard | Préfecture de Police inspector |
| `tavern_keeper` | Jacques Moreau | Bistro owner near Montmartre |
| `cabaret_dancer` | Colette Marchand | Moulin Rouge dancer |
| `inspector` | Inspecteur Gaston Lefèvre | Early forensic scientist, Sûreté |
| `artist` | Henri Toulouse | Starving painter in Montmartre |

Generic `Person` NPCs have no `system_prompt` — the backend returns their canned greeting directly without calling Mistral.

## Using a Fine-Tuned Model

Once a fine-tuning job completes (see [mh-fine-tuning](https://github.com/divait/mh-fine-tuning)), add the returned model ID to `.env`:

```env
FINETUNED_MODEL_ID=ft:open-mistral-7b:your-org:...
```

Restart the server — quest generation will automatically use the fine-tuned model.

## Running Tests

```bash
uv run pytest
```
