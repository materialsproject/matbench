"""Thin public client for the private Scientia verifier service.

The service URL and a revocable benchmark key are supplied at runtime. No
knowledge graph, extracted evidence, model weights, or verifier logic is
distributed with this client.
"""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


def verify_materials(endpoint: str, fold: int, candidates: list[dict]) -> list[dict]:
    base_url = os.environ["SCIENTIA_VERIFIER_API_URL"].rstrip("/")
    api_key = os.environ["SCIENTIA_VERIFIER_API_KEY"]
    payload = json.dumps({
        "endpoint": endpoint,
        "fold": fold,
        "candidates": candidates,
    }).encode("utf-8")
    request = Request(
        f"{base_url}/v1/verify/materials",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        body = json.load(response)
    predictions = body.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(candidates):
        raise RuntimeError("verifier API returned a malformed prediction batch")
    expected = [str(row["id"]) for row in candidates]
    received = [str(row["id"]) for row in predictions]
    if received != expected:
        raise RuntimeError("verifier API changed candidate order or identity")
    return predictions
