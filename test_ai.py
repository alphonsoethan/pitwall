# test_ai.py
from race_data import load_session, build_recap_data
from ai_recap import generate_recap, generate_strategy_verdict, answer_question

session = load_session(2024, "São Paulo", "R")
data = build_recap_data(session, "VER", 10)

print("=== RECAP ===")
print(generate_recap(data))

print("\n=== VERDICT ===")
print(generate_strategy_verdict(data))

print("\n=== Q&A ===")
print(answer_question(data, "What happened with the red flag?"))