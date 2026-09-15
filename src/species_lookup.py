"""
Looks up real organism/species names for reference accession IDs via
NCBI's public Entrez esummary API (nuccore database) -- MitoFish itself
ships no species names, only bare accession numbers (established in
Part 3 of Explanation.md).

Results are cached to a local CSV (SPECIES_CACHE_PATH) so repeat lookups
for the same accession don't re-hit the network. Network access is only
needed the first time a given accession is looked up; a network failure
just leaves that accession's name as "" rather than raising, so callers
in an offline environment (e.g. tests) never hang or crash on this.
"""
import csv
import json
import os
import urllib.parse
import urllib.request
from typing import Dict, List

SPECIES_CACHE_PATH = "data/species_name_cache.csv"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_BATCH_SIZE = 150  # stay comfortably under NCBI's URL-length / rate limits


def _load_cache(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row["Reference_ID"]: row["Species"] for row in csv.DictReader(f)}


def _save_cache(cache: Dict[str, str], path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Reference_ID", "Species"])
        for rid, species in sorted(cache.items()):
            w.writerow([rid, species])


def _query_ncbi(accessions: List[str]) -> Dict[str, str]:
    ids = ",".join(accessions)
    url = f"{_ESUMMARY_URL}?db=nuccore&id={urllib.parse.quote(ids)}&retmode=json"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.load(resp)
    result = data.get("result", {})
    names = {}
    for uid in result.get("uids", []):
        row = result.get(uid, {})
        caption, organism = row.get("caption"), row.get("organism")
        if caption and organism:
            names[caption] = organism
    return names


def fetch_organism_names(accessions: List[str], cache_path: str = SPECIES_CACHE_PATH) -> Dict[str, str]:
    """Returns {accession: organism_name} for the given accessions.
    Cached accessions are returned without any network call; only
    accessions not yet in the cache trigger an NCBI lookup (batched,
    _BATCH_SIZE at a time). An accession that can't be resolved (network
    failure or unknown ID) maps to ""."""
    accessions = [a for a in accessions if a]
    cache = _load_cache(cache_path)
    missing = [a for a in accessions if a not in cache]

    for i in range(0, len(missing), _BATCH_SIZE):
        batch = missing[i:i + _BATCH_SIZE]
        try:
            fetched = _query_ncbi(batch)
        except Exception:
            fetched = {}
        for acc in batch:
            cache[acc] = fetched.get(acc, "")

    if missing:
        _save_cache(cache, cache_path)

    return {a: cache.get(a, "") for a in accessions}
