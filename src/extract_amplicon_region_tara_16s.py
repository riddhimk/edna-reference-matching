"""
Re-extracts SILVA's Bacteria/Archaea 16S region using the ACTUAL primers
used by a real, publicly available seawater eDNA dataset -- Tara Oceans'
prokaryote amplicon sequencing (ENA project PRJEB36282) -- instead of
reusing Part 2's 515F/806R (pure V4) extraction, which used a
close-but-different primer pair.

Real dataset's primers (confirmed via literature, not assumed):
    515F-Y: GTGYCAGCMGCCGCGGTAA
    926R:   CCGYCAATTYMTTTRAGTTT
(Parada et al. 2016 pair, as used by the Tara Oceans expedition; targets
V4+V5, not V4 alone -- see Explanation.md Part 10 for the primer-mismatch
reasoning, same discipline as Part 9's 18S V9 re-extraction.)

Note: 515F-Y is IDENTICAL to Part 2's existing 515F primer (both
GTGYCAGCMGCCGCGGTAA) -- only the reverse primer differs (806R -> 926R,
extending the amplicon from V4-only into V4+V5). This is a smaller,
more targeted mismatch than the 18S V9 case, but still a real one: 806R
and 926R bind different sites, so reusing Part 2's V4-only extraction
would still truncate real reads that extend into V5.

Reuses extract_amplicon_region.py's PRIMERS dict, IUPAC handling, and
extract_region() function UNCHANGED -- adds a new PRIMERS entry at
import time rather than editing that frozen module.

Usage:
    python -m src.extract_amplicon_region_tara_16s <in_ref_table.csv> <out_csv>
"""
import sys

from .extract_amplicon_region import PRIMERS, extract_region

PRIMERS["16s_v4v5_tara"] = {
    "fwd": ("515F-Y_Parada2016", "GTGYCAGCMGCCGCGGTAA"),
    "rev": ("926R_Parada2016", "CCGYCAATTYMTTTRAGTTT"),
    "domains": {"Bacteria", "Archaea"},
    "mode": "both_primers",
}


if __name__ == "__main__":
    in_csv, out_csv = sys.argv[1], sys.argv[2]
    extract_region(in_csv, out_csv, "16s_v4v5_tara")
