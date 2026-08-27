from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


@dataclass(frozen=True)
class QueryIntent:
    people: list[str]
    genres: list[str]
    moods: list[str]
    keywords: list[str]
    query_rewrite: str | None = None


def _clean_list(values: object, limit: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []

    cleaned = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _parse_json_payload(text: str) -> dict | None:
    """Parse model JSON even if it is wrapped in a Markdown code fence."""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort recovery if the model adds a short preamble around the JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None

    return payload if isinstance(payload, dict) else None


def parse_query_intent(query: str) -> QueryIntent | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_QUERY_INTENT_MODEL", "gpt-4.1-mini"),
            temperature=0,
            max_output_tokens=350,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Translate movie discovery requests into compact JSON search intent. "
                        "Do not choose specific movies. Return exactly these keys: people, genres, moods, keywords, query_rewrite. "
                        "Infer the user's desired viewing experience, not just literal plot words. "
                        "For example, 'a rainy night sitting with your lover' should imply a romantic/cozy/intimate viewing mood and romance when appropriate; "
                        "rain, night, sitting, and lover should not automatically become literal plot requirements. "
                        "Use people only for actual named directors, actors, writers, or creators. "
                        "Genres should contain standard movie genres when they are explicit or strongly implied. "
                        "Moods should describe tone or viewing vibe. Keywords should be meaningful story/theme constraints only, not filler words or viewing context. "
                        "query_rewrite should be a concise natural-language description of the movies the user actually wants. "
                        "Return JSON only, with no Markdown fence or commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        payload = _parse_json_payload(response.output_text)
        if payload is None:
            return None

        rewrite = payload.get("query_rewrite")
        return QueryIntent(
            people=_clean_list(payload.get("people")),
            genres=_clean_list(payload.get("genres")),
            moods=_clean_list(payload.get("moods")),
            keywords=_clean_list(payload.get("keywords"), limit=12),
            query_rewrite=rewrite.strip() if isinstance(rewrite, str) and rewrite.strip() else None,
        )
    except Exception:
        return None
