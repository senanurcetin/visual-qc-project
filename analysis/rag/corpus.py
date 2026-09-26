"""Knowledge-base definition and Wikipedia fetcher for the steel QC assistant.

The corpus is a fixed list of English Wikipedia articles (CC BY-SA 4.0) grouped into six
topics that a steel-strip quality team actually touches. Each fetch records the revision id
so the published artifacts can be traced back to the exact text that was embedded.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "visual-qc-project/1.0 (portfolio RAG case study; https://github.com/senanurcetin/visual-qc-project)"
LICENSE = "CC BY-SA 4.0"

TOPICS: dict[str, list[str]] = {
    "Surface defects": [
        "Mill scale", "Non-metallic inclusions", "Crazing", "Surface roughness",
        "Surface finish", "Fatigue (material)",
    ],
    "Rolling & steelmaking": [
        "Rolling (metalworking)", "Hot working", "Cold working", "Steelmaking",
        "Continuous casting", "Pickling (metal)", "Annealing (materials science)", "Shot peening",
    ],
    "Corrosion & protection": [
        "Corrosion", "Pitting corrosion", "Rust", "Galvanization", "Hot-dip galvanization",
        "Passivation (chemistry)", "Stainless steel",
    ],
    "Quality control & SPC": [
        "Statistical process control", "Control chart", "Pareto chart", "Pareto principle",
        "Acceptance sampling", "Six Sigma", "Root-cause analysis",
        "Failure mode and effects analysis", "Process capability index",
    ],
    "Inspection & NDT": [
        "Nondestructive testing", "Ultrasonic testing", "Eddy-current testing",
        "Magnetic particle inspection", "Dye penetrant inspection", "Visual inspection",
        "Weld quality assurance", "Automated optical inspection",
    ],
    "Machine vision & ML": [
        "Machine vision", "Histogram of oriented gradients", "Gabor filter", "Random forest",
        "Anomaly detection",
    ],
}


def article_topics() -> list[tuple[str, str]]:
    return [(title, topic) for topic, titles in TOPICS.items() for title in titles]


def _get(params: dict) -> dict:
    query = urllib.parse.urlencode({"format": "json", "formatversion": 2, **params})
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_article(title: str, cache_dir: Path) -> dict:
    """Return {'title', 'pageid', 'revid', 'url', 'extract'}; cached on disk after first fetch."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (urllib.parse.quote(title, safe="") + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get({
        "action": "query", "prop": "extracts|revisions|info", "explaintext": 1,
        "exsectionformat": "wiki", "rvprop": "ids", "inprop": "url", "redirects": 1, "titles": title,
    })
    page = data["query"]["pages"][0]
    if page.get("missing"):
        raise LookupError(f"Wikipedia article not found: {title}")
    article = {
        "title": page["title"],
        "requested_title": title,
        "pageid": page["pageid"],
        "revid": page["revisions"][0]["revid"],
        "url": page["fullurl"],
        "extract": page["extract"],
    }
    path.write_text(json.dumps(article, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.2)  # stay polite to the public API
    return article
