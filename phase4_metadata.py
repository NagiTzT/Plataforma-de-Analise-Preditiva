"""Outcome-independent competition metadata and append-only repairs."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
from contextlib import closing

import pycountry
from phase2_observations import append

METADATA_VERSION = "competition-metadata-v2"


def _normal(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


@lru_cache(maxsize=1)
def _aliases():
    aliases = {}
    for country in pycountry.countries:
        for field in ("name", "official_name", "common_name", "alpha_2", "alpha_3"):
            name = getattr(country, field, None)
            if name:
                aliases[_normal(name)] = country.name
    aliases.update({
        "england": "England", "scotland": "Scotland", "wales": "Wales",
        "northern ireland": "Northern Ireland", "czech republic": "Czechia",
        "south korea": "Korea, Republic of", "korea south": "Korea, Republic of",
        "north korea": "Korea, Democratic People's Republic of",
        "russia": "Russian Federation", "iran": "Iran, Islamic Republic of",
        "bolivia": "Bolivia, Plurinational State of",
        "venezuela": "Venezuela, Bolivarian Republic of",
        "vietnam": "Viet Nam", "turkey": "Türkiye", "tanzania": "Tanzania, United Republic of",
        "kosovo": "Kosovo", "palestine": "Palestine, State of",
        "bosnia herzegovina": "Bosnia and Herzegovina", "uae": "United Arab Emirates",
    })
    return aliases


def country_from_league(league):
    label = _normal(league)
    # Token boundaries prevent e.g. US from matching a team/word prefix.
    for alias in sorted(_aliases(), key=len, reverse=True):
        if label == alias or label.startswith(alias + " "):
            return _aliases()[alias]
    return ""


def competition_metadata(*sources, league=""):
    """Use provider fields, then a recognized country prefix of the league."""
    for source_index, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        event = source.get("event") if isinstance(source.get("event"), dict) else source
        containers = [("event", event)]
        for key in ("championship", "league", "tournament", "competition"):
            node = event.get(key)
            if isinstance(node, dict):
                containers.append((key, node))
                unique = node.get("uniqueTournament")
                if isinstance(unique, dict):
                    containers.append((key + ".uniqueTournament", unique))
        for path, node in list(containers):
            category = node.get("category")
            if isinstance(category, dict):
                containers.append((path + ".category", category))
        for path, node in containers:
            for key in ("country", "country_name"):
                value = node.get(key)
                if isinstance(value, dict):
                    value = value.get("name") or value.get("title") or value.get("alpha2")
                canonical = _aliases().get(_normal(value))
                if canonical:
                    return {"country": canonical, "source_index": source_index,
                            "source_path": path + "." + key,
                            "method": "provider_field", "version": METADATA_VERSION}
            if path.endswith("category"):
                canonical = _aliases().get(_normal(node.get("name")))
                if canonical:
                    return {"country": canonical, "source_index": source_index,
                            "source_path": path + ".name",
                            "method": "provider_field", "version": METADATA_VERSION}
    country = country_from_league(league)
    return {"country": country or None, "source_index": None,
            "source_path": "league", "method": "frozen_league_prefix" if country else "missing",
            "version": METADATA_VERSION}


def repair_frozen_metadata(db_path):
    """Recover only static country labels from the original frozen league.

    The original record/hash is untouched. No current API, result or table is
    used to reconstruct old feature values or old predictions.
    """
    sidecar = Path(db_path).resolve().with_name("phase2_observations.db")
    counts = {"missing": 0, "repaired": 0, "unresolved": 0, "conflicts": 0}
    with closing(sqlite3.connect(sidecar, timeout=10)) as conn:
        rows = conn.execute("SELECT run_id,match_id,observed_at,sha256,payload FROM observations WHERE stage='phase4_prefilter_v1'").fetchall()
    for run_id, match_id, observed_at, digest, raw in rows:
        if hashlib.sha256(raw.encode()).hexdigest() != digest:
            counts["conflicts"] += 1
            continue
        data = json.loads(raw)
        if data.get("country"):
            continue
        counts["missing"] += 1
        metadata = competition_metadata(league=data.get("league"))
        if not metadata["country"]:
            counts["unresolved"] += 1
            continue
        payload = {**metadata, "parent_stage": "phase4_prefilter_v1", "parent_sha256": digest,
                   "source_observed_at": observed_at, "league": data.get("league"),
                   "static_metadata_only": True, "affects_production": False}
        if append(db_path, run_id, match_id, "phase4_metadata_repair_v1", payload):
            counts["repaired"] += 1
        else:
            counts["conflicts"] += 1
    return counts
