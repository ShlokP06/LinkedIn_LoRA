"""
Quick Groq prompt-cleaning test — calls Groq directly, no backend needed.
Run with: python test_prompts.py
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq
from deploy.prompts import _system_prompt  # no side effects — safe to import directly

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

PROMPTS = [
    "make me look like a CEO",
    "casual outdoor vibe",
    "tech startup founder",
    "cool and techy",
    "serious academic",
    "creative director at an agency",
    "finance bro",
    "i want something with a bookshelf background",
    "make me look approachable but smart",
    "outdoor, sunny, relaxed",
]


def clean(client: Groq, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.9,  # keep in sync with deploy/api.py — variety needs high temp
    )
    elapsed = time.perf_counter() - t0
    return (resp.choices[0].message.content or "").strip(), elapsed


def main():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    print(f"Model: llama-3.3-70b-versatile\n{'-' * 70}")

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i:02d}] INPUT : {prompt}")
        try:
            cleaned, elapsed = clean(client, prompt)
            starts_ok = cleaned.lower().startswith("linkedin")
            has_shlok = "shl0k" in cleaned.lower()
            has_glasses = "glasses" in cleaned.lower() or "frames" in cleaned.lower()
            has_shave = "clean-shaven" in cleaned.lower()
            word_count = len(cleaned.split())
            length_ok = 70 <= word_count <= 130
            flags = []
            if not starts_ok: flags.append("MISSING LinkedIn start")
            if not has_shlok: flags.append("MISSING Shl0k trigger")
            if not has_glasses: flags.append("MISSING glasses")
            if not has_shave: flags.append("MISSING clean-shaven")
            if not length_ok: flags.append(f"LENGTH {word_count} words (want 70-130)")
            status = "OK" if not flags else " | ".join(flags)
            print(f"     OUTPUT: {cleaned}")
            print(f"     TIME  : {elapsed:.2f}s  |  {status}")
        except Exception as e:
            print(f"     ERROR : {e}")

    print(f"\n{'-' * 70}\nDone.")


if __name__ == "__main__":
    main()
