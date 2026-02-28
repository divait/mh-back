"""
Tests for the quest system — clue delivery, quest generation, and solve endpoint.

Covers:
  - _parse_quest: clue chain validation, schema validation, inspector restrictions
  - generate_quest: model selection, quest_0 mode, fallback on failure
  - GET  /quest/models             → available model list
  - GET  /quest/                   → default QUEST_0 public view
  - POST /quest/generate           → model param, stores quest, no secrets exposed
  - GET  /quest/session/{id}       → returns active quest or QUEST_0
  - POST /quest/solve              → correct / partial / wrong solutions
  - Clue injection into dialogue   → NPC system prompt receives secret + hint for its clue
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agents.game_master import (
    ANCHOR_NPC_IDS,
    MODEL_CREATIVE,
    MODEL_DEFAULT,
    MODEL_QUEST_0,
    _parse_quest,
)
from agents.npcs import QUEST_0, QuestClue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mistral_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _valid_quest_dict(num_clues: int = 4) -> dict:
    """
    Build a minimal valid quest dict with a correct clue chain.
    Uses only ANCHOR_NPC_IDS — inspector is intentionally excluded.
    """
    # Pick from the 5 available anchor NPCs (inspector excluded)
    available = [n for n in ANCHOR_NPC_IDS if n != "inspector"]
    if num_clues > len(available):
        # Repeat from the beginning for chain tests that need > 5 clues
        selected = (available * 3)[:num_clues]
    else:
        selected = available[:num_clues]

    clues = []
    for i, npc_id in enumerate(selected):
        leads_to = selected[i + 1] if i < len(selected) - 1 else None
        clues.append(
            {
                "npc_id": npc_id,
                "secret": f"Secret known by {npc_id}.",
                "hint": f"Hint from {npc_id}.",
                "sequence": i + 1,
                "leads_to": leads_to,
            }
        )
    return {
        "quest_id": "quest_test",
        "title": "Le Mystère du Test",
        "description": "A test mystery.",
        "clues": clues,
        "solution": {
            "suspect": "Jean Dupont, the locksmith",
            "motive": "Revenge for a stolen inheritance",
            "method": "Picked the lock with a custom tool",
        },
        "red_herrings": ["A suspicious stranger was seen nearby.", "A forged letter was found."],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    with (
        patch("wandb.init"),
        patch("wandb.finish"),
        patch("wandb.log"),
        patch("wandb.run", new=None),
    ):
        from main import app

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# _parse_quest — unit tests
# ---------------------------------------------------------------------------


class TestParseQuest:
    def test_valid_4_clue_chain(self):
        quest = _parse_quest(_valid_quest_dict(4))
        assert quest.quest_id == "quest_test"
        assert len(quest.clues) == 4

    def test_valid_5_clue_chain(self):
        quest = _parse_quest(_valid_quest_dict(5))
        assert len(quest.clues) == 5

    def test_clues_sorted_by_sequence(self):
        data = _valid_quest_dict(4)
        data["clues"] = list(reversed(data["clues"]))
        quest = _parse_quest(data)
        sequences = [c.sequence for c in quest.clues]
        assert sequences == sorted(sequences)

    def test_last_clue_leads_to_null(self):
        quest = _parse_quest(_valid_quest_dict(4))
        assert quest.clues[-1].leads_to is None

    def test_chain_integrity(self):
        quest = _parse_quest(_valid_quest_dict(4))
        for i, clue in enumerate(quest.clues[:-1]):
            assert clue.leads_to == quest.clues[i + 1].npc_id

    def test_too_few_clues_raises(self):
        data = _valid_quest_dict(4)
        data["clues"] = data["clues"][:2]
        with pytest.raises(ValueError, match="Expected 4"):
            _parse_quest(data)

    def test_too_many_clues_raises(self):
        data = _valid_quest_dict(4)
        extra = {**data["clues"][0], "sequence": 7, "leads_to": None}
        data["clues"] = data["clues"] + [extra] * 4
        with pytest.raises(ValueError):
            _parse_quest(data)

    def test_unknown_npc_id_raises(self):
        data = _valid_quest_dict(4)
        data["clues"][0]["npc_id"] = "ghost_npc"
        with pytest.raises(ValueError, match="Unknown npc_id"):
            _parse_quest(data)

    def test_broken_chain_raises(self):
        data = _valid_quest_dict(4)
        data["clues"][0]["leads_to"] = "artist"  # should be "guard"
        with pytest.raises(ValueError, match="Chain broken"):
            _parse_quest(data)

    def test_last_clue_not_null_raises(self):
        data = _valid_quest_dict(4)
        data["clues"][-1]["leads_to"] = "baker"  # should be null
        with pytest.raises(ValueError, match="leads_to=null"):
            _parse_quest(data)

    def test_missing_solution_raises(self):
        data = _valid_quest_dict(4)
        data["solution"] = {"suspect": "Someone", "motive": "", "method": ""}
        with pytest.raises(ValueError, match="Incomplete solution"):
            _parse_quest(data)

    def test_solution_fields_preserved(self):
        quest = _parse_quest(_valid_quest_dict(4))
        assert "suspect" in quest.solution
        assert "motive" in quest.solution
        assert "method" in quest.solution

    def test_red_herrings_preserved(self):
        quest = _parse_quest(_valid_quest_dict(4))
        assert len(quest.red_herrings) == 2

    # --- Inspector restrictions ---

    def test_inspector_as_clue_holder_raises(self):
        """Inspector must never be assigned a clue."""
        data = _valid_quest_dict(4)
        data["clues"][0]["npc_id"] = "inspector"
        with pytest.raises(ValueError, match="Unknown npc_id"):
            _parse_quest(data)

    def test_inspector_not_in_anchor_npc_ids(self):
        """Inspector must be absent from ANCHOR_NPC_IDS at module level."""
        assert "inspector" not in ANCHOR_NPC_IDS


# ---------------------------------------------------------------------------
# generate_quest — model selection and fallback
# ---------------------------------------------------------------------------


class TestGenerateQuestModels:
    def test_quest_0_mode_returns_quest_0_instantly(self):
        """quest_0 mode must return QUEST_0 without any API call."""
        from agents.game_master import generate_quest

        with patch("agents.game_master.Mistral") as MockMistral:
            quest = generate_quest(model=MODEL_QUEST_0)
            MockMistral.assert_not_called()

        assert quest.quest_id == QUEST_0.quest_id

    def test_none_model_uses_default(self):
        """None model should call the API with MODEL_DEFAULT."""
        from agents.game_master import generate_quest

        good_response = _make_mistral_response(json.dumps(_valid_quest_dict(4)))
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = good_response
                MockMistral.return_value = mock_client
                generate_quest(model=None)
                call_kwargs = mock_client.chat.complete.call_args
                assert call_kwargs.kwargs["model"] == MODEL_DEFAULT

    def test_creative_model_is_passed_to_api(self):
        """MODEL_CREATIVE should be forwarded to the Mistral client."""
        from agents.game_master import generate_quest

        good_response = _make_mistral_response(json.dumps(_valid_quest_dict(4)))
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = good_response
                MockMistral.return_value = mock_client
                generate_quest(model=MODEL_CREATIVE)
                call_kwargs = mock_client.chat.complete.call_args
                assert call_kwargs.kwargs["model"] == MODEL_CREATIVE

    def test_custom_model_id_is_forwarded(self):
        """Any arbitrary model string should be forwarded as-is (e.g. fine-tuned ID)."""
        from agents.game_master import generate_quest

        custom_model = "ft:open-mistral-7b:abc123:paris-quest"
        good_response = _make_mistral_response(json.dumps(_valid_quest_dict(4)))
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = good_response
                MockMistral.return_value = mock_client
                generate_quest(model=custom_model)
                call_kwargs = mock_client.chat.complete.call_args
                assert call_kwargs.kwargs["model"] == custom_model


class TestGenerateQuestFallback:
    def test_falls_back_when_no_api_key(self):
        from agents.game_master import generate_quest

        with patch.dict("os.environ", {"MISTRAL_API_KEY": ""}):
            quest = generate_quest()
        assert quest.quest_id == QUEST_0.quest_id

    def test_falls_back_on_mistral_exception(self):
        from agents.game_master import generate_quest

        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.side_effect = RuntimeError("API down")
                MockMistral.return_value = mock_client
                quest = generate_quest()

        assert quest.quest_id == QUEST_0.quest_id

    def test_falls_back_on_invalid_json(self):
        from agents.game_master import generate_quest

        bad_response = _make_mistral_response("not valid json at all {{{")
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = bad_response
                MockMistral.return_value = mock_client
                quest = generate_quest()

        assert quest.quest_id == QUEST_0.quest_id

    def test_falls_back_on_invalid_chain(self):
        from agents.game_master import generate_quest

        bad_quest = _valid_quest_dict(4)
        bad_quest["clues"][0]["leads_to"] = "artist"  # broken chain
        bad_response = _make_mistral_response(json.dumps(bad_quest))
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = bad_response
                MockMistral.return_value = mock_client
                quest = generate_quest()

        assert quest.quest_id == QUEST_0.quest_id

    def test_falls_back_when_inspector_in_clue(self):
        """A quest where inspector is a clue-holder must be rejected and fall back."""
        from agents.game_master import generate_quest

        # Manually craft a quest that sneaks inspector into the clues
        bad_quest = _valid_quest_dict(4)
        bad_quest["clues"][0]["npc_id"] = "inspector"
        bad_response = _make_mistral_response(json.dumps(bad_quest))
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = bad_response
                MockMistral.return_value = mock_client
                quest = generate_quest()

        assert quest.quest_id == QUEST_0.quest_id

    def test_returns_valid_quest_on_success(self):
        from agents.game_master import generate_quest

        good_quest = _valid_quest_dict(4)
        good_response = _make_mistral_response(json.dumps(good_quest))
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = good_response
                MockMistral.return_value = mock_client
                quest = generate_quest()

        assert quest.quest_id == "quest_test"
        assert len(quest.clues) == 4


# ---------------------------------------------------------------------------
# GET /quest/models
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/quest/models")
        assert resp.status_code == 200

    def test_has_models_and_default(self, client):
        data = client.get("/quest/models").json()
        assert "models" in data
        assert "default" in data

    def test_contains_quest_0_option(self, client):
        data = client.get("/quest/models").json()
        ids = [m["id"] for m in data["models"]]
        assert MODEL_QUEST_0 in ids

    def test_contains_default_model(self, client):
        data = client.get("/quest/models").json()
        ids = [m["id"] for m in data["models"]]
        assert MODEL_DEFAULT in ids

    def test_contains_creative_model(self, client):
        data = client.get("/quest/models").json()
        ids = [m["id"] for m in data["models"]]
        assert MODEL_CREATIVE in ids

    def test_default_field_matches_constant(self, client):
        data = client.get("/quest/models").json()
        assert data["default"] == MODEL_DEFAULT

    def test_each_model_has_required_fields(self, client):
        data = client.get("/quest/models").json()
        for model in data["models"]:
            assert "id" in model
            assert "label" in model
            assert "available" in model


# ---------------------------------------------------------------------------
# GET /quest/ — default quest
# ---------------------------------------------------------------------------


class TestDefaultQuest:
    def test_returns_200(self, client):
        resp = client.get("/quest/")
        assert resp.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/quest/").json()
        assert set(data.keys()) >= {"quest_id", "title", "description", "clues", "red_herrings"}

    def test_no_secrets_in_public_view(self, client):
        data = client.get("/quest/").json()
        for clue in data["clues"]:
            assert "secret" not in clue, "Clue secrets must not be exposed to the frontend"

    def test_clues_have_hint_sequence_leads_to(self, client):
        data = client.get("/quest/").json()
        for clue in data["clues"]:
            assert "hint" in clue
            assert "sequence" in clue
            assert "leads_to" in clue

    def test_clues_ordered_by_sequence(self, client):
        data = client.get("/quest/").json()
        sequences = [c["sequence"] for c in data["clues"]]
        assert sequences == sorted(sequences)

    def test_last_clue_leads_to_null(self, client):
        data = client.get("/quest/").json()
        assert data["clues"][-1]["leads_to"] is None

    def test_inspector_not_in_default_quest_clues(self, client):
        data = client.get("/quest/").json()
        clue_npcs = [c["npc_id"] for c in data["clues"]]
        assert "inspector" not in clue_npcs


# ---------------------------------------------------------------------------
# POST /quest/generate
# ---------------------------------------------------------------------------


class TestGenerateEndpoint:
    def _post_generate(self, client, session_id: str = "gen_session", model: str | None = None) -> dict:
        good_response = _make_mistral_response(json.dumps(_valid_quest_dict(4)))
        payload = {"session_id": session_id}
        if model is not None:
            payload["model"] = model
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = good_response
                MockMistral.return_value = mock_client
                resp = client.post("/quest/generate", json=payload)
        return resp

    def test_returns_200(self, client):
        assert self._post_generate(client).status_code == 200

    def test_no_secrets_in_response(self, client):
        data = self._post_generate(client).json()
        for clue in data["clues"]:
            assert "secret" not in clue

    def test_response_includes_model_used(self, client):
        data = self._post_generate(client).json()
        assert "model_used" in data

    def test_model_used_reflects_requested_model(self, client):
        data = self._post_generate(client, model=MODEL_CREATIVE).json()
        assert data["model_used"] == MODEL_CREATIVE

    def test_quest_stored_for_session(self, client):
        import state as app_state

        session_id = "store_test_unique_1"
        self._post_generate(client, session_id)
        assert session_id in app_state.active_quests

    def test_quest_0_mode_returns_instantly(self, client):
        """quest_0 mode must not call Mistral."""
        with patch("agents.game_master.Mistral") as MockMistral:
            resp = client.post("/quest/generate", json={"session_id": "q0_test", "model": MODEL_QUEST_0})
            MockMistral.assert_not_called()
        assert resp.status_code == 200
        assert resp.json()["quest_id"] == QUEST_0.quest_id

    def test_fallback_quest_returned_on_failure(self, client):
        with patch.dict("os.environ", {"MISTRAL_API_KEY": ""}):
            resp = client.post("/quest/generate", json={"session_id": "fallback_session"})
        assert resp.status_code == 200
        assert resp.json()["quest_id"] == QUEST_0.quest_id

    def test_inspector_never_in_generated_clues(self, client):
        data = self._post_generate(client).json()
        clue_npcs = [c["npc_id"] for c in data["clues"]]
        assert "inspector" not in clue_npcs


# ---------------------------------------------------------------------------
# GET /quest/session/{session_id}
# ---------------------------------------------------------------------------


class TestSessionQuest:
    def test_returns_quest_0_when_no_session(self, client):
        resp = client.get("/quest/session/nonexistent_session_xyz")
        assert resp.status_code == 200
        assert resp.json()["quest_id"] == QUEST_0.quest_id

    def test_returns_stored_quest_after_generate(self, client):
        good_response = _make_mistral_response(json.dumps(_valid_quest_dict(4)))
        session_id = "session_quest_test_unique"

        with patch.dict("os.environ", {"MISTRAL_API_KEY": "fake-key"}):
            with patch("agents.game_master.Mistral") as MockMistral:
                mock_client = MagicMock()
                mock_client.chat.complete.return_value = good_response
                MockMistral.return_value = mock_client
                client.post("/quest/generate", json={"session_id": session_id})

        resp = client.get(f"/quest/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["quest_id"] == "quest_test"


# ---------------------------------------------------------------------------
# POST /quest/solve — solution matching
# ---------------------------------------------------------------------------


class TestSolveEndpoint:
    def _solve(self, client, session_id: str, suspect: str, motive: str, method: str) -> dict:
        return client.post(
            "/quest/solve",
            json={
                "session_id": session_id,
                "suspect": suspect,
                "motive": motive,
                "method": method,
            },
        ).json()

    def test_correct_solution_wins(self, client):
        sol = QUEST_0.solution
        result = self._solve(client, "solve_correct", sol["suspect"], sol["motive"], sol["method"])
        assert result["correct"] is True
        assert result["suspect_match"] is True
        assert result["motive_match"] is True
        assert result["method_match"] is True

    def test_solution_revealed_on_win(self, client):
        sol = QUEST_0.solution
        result = self._solve(client, "solve_reveal", sol["suspect"], sol["motive"], sol["method"])
        assert result["solution"] is not None
        assert "suspect" in result["solution"]

    def test_wrong_suspect_loses(self, client):
        sol = QUEST_0.solution
        result = self._solve(client, "solve_wrong_suspect", "Nobody McWrongname", sol["motive"], sol["method"])
        assert result["correct"] is False
        assert result["suspect_match"] is False

    def test_wrong_motive_and_method_loses(self, client):
        sol = QUEST_0.solution
        result = self._solve(client, "solve_wrong_motive", sol["suspect"], "completely wrong", "completely wrong")
        assert result["correct"] is False

    def test_partial_match_suspect_plus_motive_wins(self, client):
        sol = QUEST_0.solution
        result = self._solve(client, "solve_partial_m", sol["suspect"], sol["motive"], "wrong method")
        assert result["correct"] is True

    def test_partial_match_suspect_plus_method_wins(self, client):
        sol = QUEST_0.solution
        result = self._solve(client, "solve_partial_t", sol["suspect"], "wrong motive", sol["method"])
        assert result["correct"] is True

    def test_solution_hidden_on_loss(self, client):
        result = self._solve(client, "solve_hidden", "Wrong Person", "wrong", "wrong")
        assert result["solution"] is None

    def test_case_insensitive_matching(self, client):
        sol = QUEST_0.solution
        result = self._solve(client, "solve_case", sol["suspect"].upper(), sol["motive"].upper(), sol["method"])
        assert result["correct"] is True

    def test_partial_text_matching(self, client):
        """Player only needs to provide a substring of the correct answer."""
        sol = QUEST_0.solution
        first_word = sol["suspect"].split()[0]
        result = self._solve(client, "solve_partial_text", first_word, sol["motive"], sol["method"])
        assert result["correct"] is True

    def test_no_session_falls_back_to_quest_0(self, client):
        sol = QUEST_0.solution
        result = self._solve(client, "completely_unknown_session_abc123", sol["suspect"], sol["motive"], sol["method"])
        assert result["correct"] is True


# ---------------------------------------------------------------------------
# Clue injection into dialogue — NPC receives its secret + hint
# ---------------------------------------------------------------------------


class TestClueInjectionInDialogue:
    def _setup_quest_for_session(self, session_id: str):
        import state as app_state
        from agents.npcs import Quest

        clues = [
            QuestClue(npc_id="baker",        secret="The baker saw the thief enter through the side door at dawn.", hint="Ask the guard about unusual visitors at dawn.", sequence=1, leads_to="guard"),
            QuestClue(npc_id="guard",        secret="The guard let a man through without checking his papers.",    hint="The tavern keeper knows who that man met later.",             sequence=2, leads_to="tavern_keeper"),
            QuestClue(npc_id="tavern_keeper",secret="The man met an artist in the back room and exchanged a package.", hint="The artist on Montmartre knows what was in the package.", sequence=3, leads_to="artist"),
            QuestClue(npc_id="artist",       secret="The artist was paid to forge a copy of the painting.",        hint=None,                                                         sequence=4, leads_to=None),
        ]
        app_state.active_quests[session_id] = Quest(
            quest_id="quest_injection_test",
            title="Test Quest",
            description="A test.",
            clues=clues,
            solution={"suspect": "Jean Forger", "motive": "Greed", "method": "Forgery and substitution"},
            red_herrings=["Red herring 1", "Red herring 2"],
        )

    def _capture_system_prompt(self, client, session_id: str, npc_id: str) -> str:
        captured = []

        def fake_complete(model, messages, **kwargs):
            captured.extend(messages)
            return _make_mistral_response("...")

        with patch("routes.dialogue._get_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.complete.side_effect = fake_complete
            mock_factory.return_value = mock_client
            client.post("/dialogue/", json={"session_id": session_id, "npc_id": npc_id, "player_message": "Tell me."})

        return next(m["content"] for m in captured if m["role"] == "system")

    def test_baker_receives_its_clue_secret(self, client):
        session_id = "inject_baker"
        self._setup_quest_for_session(session_id)
        system = self._capture_system_prompt(client, session_id, "baker")
        assert "saw the thief enter through the side door" in system

    def test_baker_receives_its_clue_hint(self, client):
        session_id = "inject_baker_hint"
        self._setup_quest_for_session(session_id)
        system = self._capture_system_prompt(client, session_id, "baker")
        assert "Ask the guard about unusual visitors" in system

    def test_guard_receives_its_own_clue_not_bakers(self, client):
        session_id = "inject_guard"
        self._setup_quest_for_session(session_id)
        system = self._capture_system_prompt(client, session_id, "guard")
        assert "let a man through without checking his papers" in system
        assert "saw the thief enter through the side door" not in system

    def test_npc_without_clue_gets_no_injection(self, client):
        session_id = "inject_no_clue"
        self._setup_quest_for_session(session_id)
        system = self._capture_system_prompt(client, session_id, "cabaret_dancer")
        assert "QUEST DIRECTIVE" not in system

    def test_no_active_quest_no_injection(self, client):
        session_id = "no_quest_unique_xyz_789"
        system = self._capture_system_prompt(client, session_id, "baker")
        assert "QUEST DIRECTIVE" not in system

    def test_all_clue_npcs_receive_injection(self, client):
        for npc_id in ["baker", "guard", "tavern_keeper", "artist"]:
            session_id = f"inject_all_{npc_id}"
            self._setup_quest_for_session(session_id)
            system = self._capture_system_prompt(client, session_id, npc_id)
            assert "QUEST DIRECTIVE" in system, f"NPC {npc_id!r} should receive quest injection"

    def test_inspector_dialogue_gets_no_clue_injection(self, client):
        """Inspector is the solve NPC — he must never receive a clue injection."""
        session_id = "inject_inspector"
        self._setup_quest_for_session(session_id)
        system = self._capture_system_prompt(client, session_id, "inspector")
        assert "QUEST DIRECTIVE" not in system
