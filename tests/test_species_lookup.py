from unittest.mock import patch

from src.species_lookup import fetch_organism_names


def test_fetch_organism_names_uses_cache_without_network_call(tmp_path):
    cache_path = tmp_path / "species_cache.csv"
    cache_path.write_text("Reference_ID,Species\nAB123456,Thunnus albacares\n")

    with patch("src.species_lookup._query_ncbi") as mock_query:
        result = fetch_organism_names(["AB123456"], cache_path=str(cache_path))

    mock_query.assert_not_called()
    assert result == {"AB123456": "Thunnus albacares"}


def test_fetch_organism_names_queries_and_caches_missing_accessions(tmp_path):
    cache_path = tmp_path / "species_cache.csv"

    with patch("src.species_lookup._query_ncbi", return_value={"XY999999": "Katsuwonus pelamis"}) as mock_query:
        result = fetch_organism_names(["XY999999"], cache_path=str(cache_path))

    mock_query.assert_called_once_with(["XY999999"])
    assert result == {"XY999999": "Katsuwonus pelamis"}
    assert cache_path.exists()
    assert "Katsuwonus pelamis" in cache_path.read_text()


def test_fetch_organism_names_falls_back_to_empty_string_on_network_failure(tmp_path):
    cache_path = tmp_path / "species_cache.csv"

    with patch("src.species_lookup._query_ncbi", side_effect=OSError("no network")):
        result = fetch_organism_names(["ZZ000000"], cache_path=str(cache_path))

    assert result == {"ZZ000000": ""}


def test_fetch_organism_names_ignores_blank_accessions(tmp_path):
    cache_path = tmp_path / "species_cache.csv"

    with patch("src.species_lookup._query_ncbi", return_value={}) as mock_query:
        result = fetch_organism_names(["", "AB123456", ""], cache_path=str(cache_path))

    mock_query.assert_called_once_with(["AB123456"])
    assert result == {"AB123456": ""}
