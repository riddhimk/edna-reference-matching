"""
Shared FASTA parsing for SILVA-format headers.

Header grammar confirmed from real inspected records (not assumed from memory):
    >{accession}.{start}.{stop}  {semicolon-delimited taxonomy}...{organism label}
e.g.
    >AY846372.1.1779  Eukaryota;Archaeplastida;...;Ankistrodesmus sp. Mary 8/18 T-2w
    >CS394307.13.1872  Eukaryota;Amorphea;...;Rattus norvegicus (Norway rat)

Taxonomy depth is VARIABLE (8 fields vs 17 fields seen in the two records above)
-- do not assume a fixed rank ladder. See README.md.
"""
import re
from typing import Iterator, Optional, Tuple, Dict

HEADER_RE = re.compile(r"^>(\S+)\.(\d+)\.(\d+)\s+(.*)$")


def iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header_line, sequence) for each record in a FASTA file.

    header_line includes the leading '>' and is NOT parsed here -- use
    parse_header() on it. Sequence has all newlines stripped and multi-line
    records concatenated.
    """
    header, seq_lines = None, []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header, seq_lines = line.strip(), []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            yield header, "".join(seq_lines)


def parse_header(header_line: str) -> Optional[Dict]:
    """Parse a single SILVA-format header line.

    Returns None (does NOT guess) if the line doesn't match the confirmed
    grammar -- callers must count and report these, not silently drop them.
    """
    m = HEADER_RE.match(header_line.strip())
    if not m:
        return None
    accession, start, stop, taxonomy = m.groups()
    ref_id = f"{accession}.{start}.{stop}"
    fields = [f.strip() for f in taxonomy.split(";") if f.strip()]
    return {
        "Reference_ID": ref_id,
        "Accession": accession,
        "Start": int(start),
        "Stop": int(stop),
        "Ref_Length_bp": int(stop) - int(start) + 1,
        "Taxonomy_Full": taxonomy,
        "Taxonomy_Depth": len(fields),
        "Domain": fields[0] if fields else "",
        "Organism_Label": fields[-1] if fields else "",
    }


def to_dna(seq: str) -> str:
    """Convert an RNA sequence (U alphabet, as SILVA stores it) to DNA (T).

    SILVA sequences use U, but DNABERT-S/DNABERT-2 tokenizers are trained on
    the DNA alphabet. This returns a DERIVED copy -- callers must not
    overwrite the original stored sequence with this.
    """
    return seq.upper().replace("U", "T")
