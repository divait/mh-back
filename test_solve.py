import asyncio
from unittest.mock import MagicMock
from routes.quest import _llm_validate_solution, SolveRequest

true_sol = {
    "suspect": "Marie Dupont, the baker",
    "motive": "She wanted to pay off her gambling debts by selling the jewels",
    "method": "She copied the vault key when delivering morning bread"
}

print("Testing exact match...")
exact_req = SolveRequest(session_id="test", suspect="Marie Dupont, the baker", motive="She wanted to pay off her gambling debts by selling the jewels", method="She copied the vault key when delivering morning bread")
print("exact:", _llm_validate_solution(MagicMock(), exact_req, true_sol))

print("Testing vague match...")
vague_req = SolveRequest(session_id="test", suspect="The bakery lady", motive="She owed a lot of money", method="She duplicated the keys during her delivery")
print("vague:", _llm_validate_solution(MagicMock(), vague_req, true_sol))

print("Testing incorrect match...")
bad_req = SolveRequest(session_id="test", suspect="The guard", motive="He hated the victim", method="He snuck in the window")
print("bad:", _llm_validate_solution(MagicMock(), bad_req, true_sol))
