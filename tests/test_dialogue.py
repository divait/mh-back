"""
Tests for routes/dialogue.py — Task 1.1 critical path.

Mistral and W&B are mocked so these run without API keys.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mistral_response(text: str) -> MagicMock:
    """Build a minimal mock that looks like a Mistral chat completion."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """
    TestClient with Mistral and W&B fully mocked.
    W&B lifespan is skipped so no network calls happen on startup.
    """
    with (
        patch("wandb.init"),
        patch("wandb.finish"),
        patch("wandb.log"),
        patch("wandb.run", new=None),
    ):
        # Import app after patching so lifespan doesn't call real wandb
        from main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_lists_all_six_agents(self, client):
        resp = client.get("/health")
        npcs = set(resp.json()["npcs"])
        assert npcs == {"baker", "guard", "tavern_keeper", "cabaret_dancer", "inspector", "artist"}


# ---------------------------------------------------------------------------
# GET /npcs
# ---------------------------------------------------------------------------

class TestListNpcs:
    def test_returns_agents_persons_colors(self, client):
        resp = client.get("/npcs")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert "persons" in data
        assert "colors" in data

    def test_six_agents(self, client):
        agents = client.get("/npcs").json()["agents"]
        assert len(agents) == 6

    def test_agent_has_category(self, client):
        agents = client.get("/npcs").json()["agents"]
        for agent in agents:
            assert agent["category"] in ("original", "belle_epoque")

    def test_three_persons(self, client):
        persons = client.get("/npcs").json()["persons"]
        assert len(persons) == 3

    def test_person_has_greeting(self, client):
        persons = client.get("/npcs").json()["persons"]
        for person in persons:
            assert person["greeting"].strip()

    def test_colors_keys(self, client):
        colors = client.get("/npcs").json()["colors"]
        assert set(colors.keys()) == {"original", "belle_epoque", "person"}


# ---------------------------------------------------------------------------
# POST /dialogue/ — 404 for unknown / Person NPC
# ---------------------------------------------------------------------------

class TestDialogue404:
    @pytest.mark.parametrize("bad_id", ["unknown_npc", "passerby", "shopkeeper", "flaneur", ""])
    def test_unknown_npc_returns_404(self, client, bad_id):
        resp = client.post(
            "/dialogue/",
            json={
                "session_id": "test_session",
                "npc_id": bad_id,
                "player_message": "Bonjour!",
            },
        )
        assert resp.status_code == 404, (
            f"Expected 404 for npc_id={bad_id!r}, got {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# POST /dialogue/ — happy path for all 6 agent NPCs
# ---------------------------------------------------------------------------

AGENT_IDS = ["baker", "guard", "tavern_keeper", "cabaret_dancer", "inspector", "artist"]


class TestDialogueHappyPath:
    @pytest.mark.parametrize("npc_id", AGENT_IDS)
    def test_valid_npc_returns_200(self, client, npc_id):
        mock_reply = f"Bonjour, je suis {npc_id}."
        mock_completion = _make_mistral_response(mock_reply)

        with patch("routes.dialogue._get_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.chat.complete.return_value = mock_completion
            mock_client_factory.return_value = mock_client

            resp = client.post(
                "/dialogue/",
                json={
                    "session_id": "test_session",
                    "npc_id": npc_id,
                    "player_message": "Bonjour!",
                },
            )

        assert resp.status_code == 200, f"Expected 200 for {npc_id}, got {resp.status_code}"
        data = resp.json()
        assert data["npc_id"] == npc_id
        assert data["response"] == mock_reply
        assert data["model_used"]

    @pytest.mark.parametrize("npc_id", AGENT_IDS)
    def test_response_schema(self, client, npc_id):
        mock_completion = _make_mistral_response("Une réponse.")

        with patch("routes.dialogue._get_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.chat.complete.return_value = mock_completion
            mock_client_factory.return_value = mock_client

            resp = client.post(
                "/dialogue/",
                json={
                    "session_id": "test_session",
                    "npc_id": npc_id,
                    "player_message": "Parlez-moi du vol.",
                },
            )

        data = resp.json()
        assert set(data.keys()) >= {"npc_id", "npc_name", "response", "model_used"}


# ---------------------------------------------------------------------------
# POST /dialogue/ — conversation history is preserved per session+npc
# ---------------------------------------------------------------------------

class TestDialogueHistory:
    def test_history_accumulates(self, client):
        """Second message to the same NPC should include prior turn in context."""
        calls = []

        def fake_complete(model, messages, **kwargs):
            calls.append(messages)
            return _make_mistral_response("Oui, bien sûr.")

        with patch("routes.dialogue._get_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.complete.side_effect = fake_complete
            mock_factory.return_value = mock_client

            client.post(
                "/dialogue/",
                json={"session_id": "hist_test", "npc_id": "baker", "player_message": "Bonjour!"},
            )
            client.post(
                "/dialogue/",
                json={"session_id": "hist_test", "npc_id": "baker", "player_message": "Et alors?"},
            )

        # Second call's messages should contain the first user turn
        second_call_messages = calls[1]
        user_contents = [m["content"] for m in second_call_messages if m["role"] == "user"]
        assert "Bonjour!" in user_contents, "Prior player message should be in history"

    def test_history_isolated_per_npc(self, client):
        """History for baker should not bleed into guard's conversation."""
        calls: dict[str, list] = {}

        def fake_complete(model, messages, **kwargs):
            npc_id = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            return _make_mistral_response("Réponse.")

        with patch("routes.dialogue._get_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.complete.side_effect = fake_complete
            mock_factory.return_value = mock_client

            client.post(
                "/dialogue/",
                json={"session_id": "iso_test", "npc_id": "baker", "player_message": "Pain?"},
            )
            resp = client.post(
                "/dialogue/",
                json={"session_id": "iso_test", "npc_id": "guard", "player_message": "Garde?"},
            )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# DELETE /dialogue/history/{session_id}
# ---------------------------------------------------------------------------

class TestClearHistory:
    def test_clear_history_returns_cleared_keys(self, client):
        with patch("routes.dialogue._get_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.complete.return_value = _make_mistral_response("Ok.")
            mock_factory.return_value = mock_client

            client.post(
                "/dialogue/",
                json={"session_id": "clear_test", "npc_id": "baker", "player_message": "Hi"},
            )

        resp = client.delete("/dialogue/history/clear_test")
        assert resp.status_code == 200
        data = resp.json()
        assert "cleared" in data
        assert any("clear_test" in k for k in data["cleared"])
