"""
Tests for agents/npcs.py — data integrity for Task 1.1.

These are pure unit tests (no network, no Mistral, no W&B).
They verify that the NPC definitions, QUEST_0, Person NPCs, and
colour constants are internally consistent and complete.
"""

import pytest
from agents.npcs import (
    NPC,
    NPC_COLORS,
    NPCS,
    PERSONS,
    QUEST_0,
    Person,
    Quest,
    QuestClue,
)

EXPECTED_AGENT_IDS = {"baker", "guard", "tavern_keeper", "cabaret_dancer", "inspector", "artist"}
EXPECTED_ORIGINAL_IDS = {"baker", "guard", "tavern_keeper"}
EXPECTED_BELLE_EPOQUE_IDS = {"cabaret_dancer", "inspector", "artist"}
EXPECTED_PERSON_IDS = {"passerby", "shopkeeper", "flaneur"}


# ---------------------------------------------------------------------------
# NPC_COLORS
# ---------------------------------------------------------------------------

class TestNpcColors:
    def test_all_keys_present(self):
        assert set(NPC_COLORS.keys()) == {"original", "belle_epoque", "person"}

    def test_values_are_ints(self):
        for key, value in NPC_COLORS.items():
            assert isinstance(value, int), f"NPC_COLORS[{key!r}] should be int"

    def test_specific_values(self):
        assert NPC_COLORS["original"] == 0x4A6FA5
        assert NPC_COLORS["belle_epoque"] == 0xD4AF37
        assert NPC_COLORS["person"] == 0x888888


# ---------------------------------------------------------------------------
# NPCS dict — agent NPCs
# ---------------------------------------------------------------------------

class TestNpcs:
    def test_exactly_six_agents(self):
        assert set(NPCS.keys()) == EXPECTED_AGENT_IDS

    def test_all_are_npc_instances(self):
        for npc_id, npc in NPCS.items():
            assert isinstance(npc, NPC), f"NPCS[{npc_id!r}] should be NPC"

    def test_id_matches_dict_key(self):
        for key, npc in NPCS.items():
            assert npc.id == key, f"NPC id mismatch: key={key!r}, npc.id={npc.id!r}"

    def test_original_category(self):
        for npc_id in EXPECTED_ORIGINAL_IDS:
            assert NPCS[npc_id].category == "original", f"{npc_id} should be 'original'"

    def test_belle_epoque_category(self):
        for npc_id in EXPECTED_BELLE_EPOQUE_IDS:
            assert NPCS[npc_id].category == "belle_epoque", f"{npc_id} should be 'belle_epoque'"

    @pytest.mark.parametrize("npc_id", list(EXPECTED_AGENT_IDS))
    def test_required_fields_non_empty(self, npc_id):
        npc = NPCS[npc_id]
        assert npc.name.strip(), f"{npc_id}.name is empty"
        assert npc.role.strip(), f"{npc_id}.role is empty"
        assert npc.location.strip(), f"{npc_id}.location is empty"
        assert npc.personality.strip(), f"{npc_id}.personality is empty"
        assert npc.secret.strip(), f"{npc_id}.secret is empty"
        assert npc.knowledge.strip(), f"{npc_id}.knowledge is empty"
        assert npc.system_prompt.strip(), f"{npc_id}.system_prompt is empty"

    @pytest.mark.parametrize("npc_id", list(EXPECTED_AGENT_IDS))
    def test_system_prompt_no_1789_references(self, npc_id):
        prompt = NPCS[npc_id].system_prompt.lower()
        forbidden = ["1789", "liste des traîtres", "pamphlet", "café de foy", "révolution"]
        for word in forbidden:
            assert word not in prompt, (
                f"{npc_id}.system_prompt contains forbidden 1789 reference: {word!r}"
            )

    @pytest.mark.parametrize("npc_id", list(EXPECTED_AGENT_IDS))
    def test_system_prompt_contains_belle_epoque_context(self, npc_id):
        prompt = NPCS[npc_id].system_prompt.lower()
        assert "louvre" in prompt or "mona lisa" in prompt or "belle époque" in prompt, (
            f"{npc_id}.system_prompt should reference the Louvre, Mona Lisa, or Belle Époque"
        )

    def test_no_person_id_in_npcs(self):
        """Person IDs must never appear in NPCS — they must not call Mistral."""
        for person_id in EXPECTED_PERSON_IDS:
            assert person_id not in NPCS, f"Person id {person_id!r} must not be in NPCS"


# ---------------------------------------------------------------------------
# PERSONS dict — non-agentic background NPCs
# ---------------------------------------------------------------------------

class TestPersons:
    def test_exactly_three_persons(self):
        assert set(PERSONS.keys()) == EXPECTED_PERSON_IDS

    def test_all_are_person_instances(self):
        for pid, person in PERSONS.items():
            assert isinstance(person, Person), f"PERSONS[{pid!r}] should be Person"

    def test_id_matches_dict_key(self):
        for key, person in PERSONS.items():
            assert person.id == key

    def test_person_has_no_system_prompt(self):
        for pid, person in PERSONS.items():
            assert not hasattr(person, "system_prompt"), (
                f"Person {pid!r} must not have system_prompt"
            )
            assert not hasattr(person, "secret"), f"Person {pid!r} must not have secret"
            assert not hasattr(person, "knowledge"), f"Person {pid!r} must not have knowledge"

    @pytest.mark.parametrize("pid", list(EXPECTED_PERSON_IDS))
    def test_greeting_non_empty(self, pid):
        assert PERSONS[pid].greeting.strip(), f"Person {pid!r} greeting is empty"

    @pytest.mark.parametrize("pid", list(EXPECTED_PERSON_IDS))
    def test_color_is_person_grey(self, pid):
        assert PERSONS[pid].color == NPC_COLORS["person"], (
            f"Person {pid!r} color should be NPC_COLORS['person']"
        )


# ---------------------------------------------------------------------------
# QUEST_0 — hardcoded "Quest 0" schema
# ---------------------------------------------------------------------------

class TestQuest0:
    def test_is_quest_instance(self):
        assert isinstance(QUEST_0, Quest)

    def test_required_fields_non_empty(self):
        assert QUEST_0.quest_id == "quest_0"
        assert QUEST_0.title.strip()
        assert QUEST_0.description.strip()

    def test_solution_has_required_keys(self):
        assert "suspect" in QUEST_0.solution
        assert "motive" in QUEST_0.solution
        assert "method" in QUEST_0.solution

    def test_clues_cover_all_anchor_agents(self):
        clue_npc_ids = {c.npc_id for c in QUEST_0.clues}
        expected = EXPECTED_AGENT_IDS - {"inspector"}
        assert clue_npc_ids == expected, (
            f"QUEST_0 clues should cover all anchor agents. Missing: {expected - clue_npc_ids}"
        )

    def test_each_clue_is_quest_clue_instance(self):
        for clue in QUEST_0.clues:
            assert isinstance(clue, QuestClue)

    @pytest.mark.parametrize("clue", QUEST_0.clues)
    def test_clue_fields_non_empty(self, clue):
        assert clue.npc_id.strip(), "clue.npc_id is empty"
        assert clue.secret.strip(), f"clue for {clue.npc_id!r} has empty secret"
        assert clue.hint.strip(), f"clue for {clue.npc_id!r} has empty hint"

    def test_clue_npc_ids_exist_in_npcs(self):
        for clue in QUEST_0.clues:
            assert clue.npc_id in NPCS, (
                f"QUEST_0 clue references unknown NPC: {clue.npc_id!r}"
            )

    def test_red_herrings_are_list(self):
        assert isinstance(QUEST_0.red_herrings, list)
