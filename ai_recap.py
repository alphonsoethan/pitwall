# ai_recap.py
import os
import streamlit as st
from google import genai
from google.genai import types

# --- Gemini client initialization ---
# Priority: read from Streamlit secrets; if running locally, read from environment variable
def _get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = _get_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .streamlit/secrets.toml or environment variable.")
        _client = genai.Client(api_key=api_key)
    return _client


# --- Prompt templates ---

RECAP_SYSTEM_PROMPT = """You are a knowledgeable F1 race companion for a fan who follows ONE driver.
You will be given a structured summary of what happened after a specific lap, generated from real race timing data.

Rules:
1. Write a fan-friendly recap of 150-250 words. Focus on the chosen driver.
2. Do NOT invent events not present in the data. If something isn't mentioned, it didn't happen.
3. If the data mentions a RED FLAG, explain it as race context (tyre changes allowed, race neutralized) — NOT as the driver having an incident.
4. Mention strategy: tyre life, position changes, key overtakes/being overtaken.
5. Tone: like a friend who watched the race explaining it to you. Not robotic, not a data dump.
6. End with one sentence on what to watch in the next phase of the race (if applicable).
7. Avoid superlatives like "greatest ever" or "historic". Let the facts (position changes, lap numbers) carry the excitement."""

VERDICT_SYSTEM_PROMPT = """You are an F1 strategy analyst. Given the structured race summary, write a 60-100 word verdict on whether the driver's strategy worked after the cutoff lap.

Rules:
- Be specific: cite position changes and tyre usage from the data.
- If the red flag affected strategy, mention it.
- Do NOT invent pit stops or incidents not in the data.
- Tone: analytical but accessible."""

CHAT_SYSTEM_PROMPT = """You are an F1 race companion. Answer the user's question using ONLY the structured race summary provided.

Rules:
- If the answer is not in the summary, say: "The race data I have doesn't cover that. Try asking about position changes, tyre strategy, or key moments after the cutoff lap."
- Do NOT invent lap times, pit stops, or incidents.
- Keep answers concise (under 120 words) unless the user asks for detail.
- Reference specific lap numbers from the data when possible."""


# --- LLM functions ---

def generate_recap(recap_data):
    """Generate the main fan-friendly narrative."""
    client = get_client()
    summary = recap_data["summary"]

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"Here is the race data:\n\n{summary}",
        config=types.GenerateContentConfig(
                system_instruction=RECAP_SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=2000,      # Increase to 2000
                thinking_config=types.ThinkingConfig(
                thinking_budget=0         # Disable thinking (or set to 128)
            ),
        ),
    )
    return response.text


def generate_strategy_verdict(recap_data):
    """Generate a short strategy verdict paragraph."""
    client = get_client()
    summary = recap_data["summary"]

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"Here is the race data:\n\n{summary}",
        config=types.GenerateContentConfig(
            system_instruction=VERDICT_SYSTEM_PROMPT,
            temperature=0.5,
            max_output_tokens=2000,
        ),
    )
    return response.text


def answer_question(recap_data, question, chat_history=None):
    """
    Answer a user question grounded in the race summary.
    chat_history: optional list of {"role": "user"/"model", "parts": [text]}
    """
    client = get_client()
    summary = recap_data["summary"]

    # Build contents: summary as context + optional history + current question
    contents = f"Race summary (use this as your only factual source):\n\n{summary}\n\n"
    if chat_history:
        for turn in chat_history:
            contents += f"{turn['role']}: {turn['parts'][0]}\n"
    contents += f"\nUser question: {question}"

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=CHAT_SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=2000,
        ),
    )
    return response.text    