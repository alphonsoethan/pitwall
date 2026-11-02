# gemini_client.py
import os
import time
import random
import streamlit as st
from google import genai
from google.genai import types


def _get_client():
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _call_gemini(prompt):
    client = _get_client()
    if client is None:
        return "Gemini API key not configured. Add GEMINI_API_KEY to .streamlit/secrets.toml."

    models_to_try = ["gemini-3.6-flash", "gemini-3.8-flash"]
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=4000,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return _clean_response(response.text)
            except Exception as e:
                err = str(e)
                if ("503" in err or "429" in err) and attempt < 2:
                    time.sleep(min(2 ** attempt + random.uniform(0, 1), 10))
                    continue
                break
    return "Gemini is currently under high demand. Please try again in a few seconds."


def _clean_response(text):
    """Strip any leaked token markers / stray fragments."""
    if not isinstance(text, str):
        return text
    import re
    # Remove patterns like "(72)" at sentence boundaries, or empty leading commas
    cleaned = re.sub(r"\(\d+\)", "", text)
    cleaned = re.sub(r"^\s*[,;]\s*", "", cleaned)          # leading comma
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def get_strategy(weather_data, leaderboard_text):
    prompt = f"""You are an F1 race strategist. Based on the following live race conditions, suggest the best tyre strategy and expected pit stops.

Weather: Air {weather_data['air']}°C, Track {weather_data['track']}°C, Humidity {weather_data['humidity']}%, Rain: {weather_data['rain']}

Current leaderboard (top 10):
{leaderboard_text}

Give a concise recommendation (2-3 sentences) covering:
1. Which tyre compound is best suited right now and why
2. Expected number of pit stops for the remaining race
3. Any risk factors (weather, degradation)"""
    return _call_gemini(prompt)


def get_recap(summary):
    prompt = f"""You are a knowledgeable F1 race companion for a fan who follows ONE driver.
Here is a structured summary of what happened after a specific lap, from real race timing data.

{summary}

Write a fan-friendly recap of 150-250 words focused on the chosen driver.
Rules:
- Do NOT invent events not in the data.
- If a RED FLAG is mentioned, explain it as race context, not a driver incident.
- Mention strategy, tyre life, position changes.
- Tone: like a friend explaining the race. Avoid superlatives like "greatest ever".
- End with one sentence on what to watch next."""
    return _call_gemini(prompt)

def get_race_recap(summary):
    prompt = f"""You are an F1 race companion writing a short recap for a fan who missed the race.

Here is the structured race summary:
{summary}

Write ONE paragraph (120-160 words), no bullet points, no headers, no token counts.
Cover only: the winner, the final podium, and the one strategic moment (red flag / safety car / pit timing) that shaped the result.
Do NOT invent anything not in the summary. Do NOT mention championship implications unless they appear in the summary.
Tone: like a friend explaining the race. Plain sentences."""
    return _call_gemini(prompt)


def get_race_verdict(summary):
    prompt = f"""You are an F1 analyst. Using ONLY this race summary:

{summary}

Write ONE short paragraph (50-80 words), no headers, no numbers in parentheses.
State the single strategic call that decided the race and why it worked.
If a red flag or safety car shaped the result, name it. Otherwise say the race was decided on pace.
Do NOT invent events."""
    return _call_gemini(prompt)

def get_race_verdict(summary):
    prompt = f"""You are an F1 analyst. Given this race summary, write a 60-100 word verdict on the race's defining strategic moment.

{summary}

Rules:
- Cite the winner and the key strategic call (or red flag) that decided it.
- Do NOT invent events.
- Analytical but accessible tone."""
    return _call_gemini(prompt)

def get_verdict(summary):
    prompt = f"""You are an F1 strategy analyst. Given this race summary, write a 60-100 word verdict on whether the driver's strategy worked after the cutoff lap.

{summary}

Rules:
- Cite position changes and tyre usage from the data.
- If the red flag affected strategy, mention it.
- Do NOT invent pit stops or incidents.
- Analytical but accessible tone."""
    return _call_gemini(prompt)