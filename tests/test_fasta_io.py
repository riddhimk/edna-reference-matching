import os
import tempfile

from src.fasta_io import iter_fasta, parse_header, to_dna

REAL_HEADER_1 = ">AY846372.1.1779  Eukaryota;Archaeplastida;Chloroplastida;Chlorophyta;Chlorophyceae;Sphaeropleales;Ankistrodesmus;Ankistrodesmus sp. Mary 8/18 T-2w"
REAL_HEADER_2 = ">CS394307.13.1872  Eukaryota;Amorphea;Obazoa;Opisthokonta;Holozoa;Choanozoa;Metazoa;Animalia;BCP clade;Bilateria;Deuterostomia;Chordata;Vertebrata;Gnathostomata;Euteleostomi;Tetrapoda;Mammalia;Rattus norvegicus (Norway rat)"


def test_parse_header_record_1():
    rec = parse_header(REAL_HEADER_1)
    assert rec is not None
    assert rec["Reference_ID"] == "AY846372.1.1779"
    assert rec["Accession"] == "AY846372"
    assert rec["Start"] == 1
    assert rec["Stop"] == 1779
    assert rec["Ref_Length_bp"] == 1779
    assert rec["Domain"] == "Eukaryota"
    assert rec["Organism_Label"] == "Ankistrodesmus sp. Mary 8/18 T-2w"
    assert rec["Taxonomy_Depth"] == 8


def test_parse_header_record_2_variable_depth():
    rec = parse_header(REAL_HEADER_2)
    assert rec is not None
    assert rec["Reference_ID"] == "CS394307.13.1872"
    assert rec["Start"] == 13
    assert rec["Stop"] == 1872
    assert rec["Ref_Length_bp"] == 1860
    assert rec["Organism_Label"] == "Rattus norvegicus (Norway rat)"
    # This is the key finding from Section 3.2: depth is NOT fixed across records
    # (18 fields here vs. 8 in record 1 -- corrected from an earlier miscount of 17)
    assert rec["Taxonomy_Depth"] == 18


def test_taxonomy_depth_varies_between_real_records():
    r1 = parse_header(REAL_HEADER_1)
    r2 = parse_header(REAL_HEADER_2)
    assert r1["Taxonomy_Depth"] != r2["Taxonomy_Depth"]


def test_parse_header_returns_none_for_malformed_line():
    assert parse_header(">not_a_valid_silva_header no dots here") is None


def test_to_dna_converts_u_to_t_without_mutating_input():
    rna = "CUGCCAGUAGUCAUAUGC"
    dna = to_dna(rna)
    assert dna == "CTGCCAGTAGTCATATGC"
    assert rna == "CUGCCAGUAGUCAUAUGC"  # original untouched


def test_iter_fasta_multiline_sequence_concat():
    content = (
        f"{REAL_HEADER_1}\n"
        "CUGCCAGUAGUCAUAUGC\n"
        "UUGUCUCAAGAUUAAGCC\n"
        f"{REAL_HEADER_2}\n"
        "CCUGGUGGAUCAAGAUCU\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        records = list(iter_fasta(path))
        assert len(records) == 2
        header1, seq1 = records[0]
        assert header1 == REAL_HEADER_1
        assert seq1 == "CUGCCAGUAGUCAUAUGCUUGUCUCAAGAUUAAGCC"
        header2, seq2 = records[1]
        assert header2 == REAL_HEADER_2
        assert seq2 == "CCUGGUGGAUCAAGAUCU"
    finally:
        os.unlink(path)
