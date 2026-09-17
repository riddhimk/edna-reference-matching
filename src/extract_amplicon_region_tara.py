"""
Re-extracts SILVA's Eukaryota 18S region using the ACTUAL primers used by
a real, publicly available seawater eDNA dataset -- Tara Pacific's 18S V9
amplicon sequencing (ENA project PRJEB60216) -- instead of reusing Part 2's
1391F/EukBr extraction, which used a close-but-different primer pair
(Amaral-Zettler et al. 2009 EMP primers, not what this specific real
dataset used).

Real dataset's primers (confirmed via literature, not assumed):
    1389F: TTGTACACACCGCCC
    1510R: CCTTCYGCAGGTTCACCTAC
(Tara Oceans/Pacific 18S V9 protocol; see e.g. Nature Scientific Data
2017, "Viral to metazoan marine plankton nucleotide sequences from the
Tara Oceans expedition".)

Why a separate extraction instead of reusing Part 2's 18s_v9 output: this
project's own rule (established in Part 1/2) is that a close-but-different
primer pair is a real region mismatch, not something to assume is "close
enough" -- the same reasoning that made Part 1's full-length matching fail
and Part 2's primer-verified re-extraction succeed.

Reuses extract_amplicon_region.py's PRIMERS dict, IUPAC handling, and
extract_region() function UNCHANGED -- adds a new PRIMERS entry at import
time rather than editing that frozen module, so Part 2's original
1391F/EukBr "18s_v9" extraction is completely undisturbed.

Usage:
    python -m src.extract_amplicon_region_tara <in_ref_table.csv> <out_csv>
"""
import sys

from .extract_amplicon_region import PRIMERS, extract_region

PRIMERS["18s_v9_tara"] = {
    "fwd": ("1389F_TaraOceans", "TTGTACACACCGCCC"),
    "rev": ("1510R_TaraOceans", "CCTTCYGCAGGTTCACCTAC"),
    "domains": {"Eukaryota"},
    "mode": "both_primers",  # tried first; empirically confirmed below whether this file supports it
}


if __name__ == "__main__":
    in_csv, out_csv = sys.argv[1], sys.argv[2]
    extract_region(in_csv, out_csv, "18s_v9_tara")
