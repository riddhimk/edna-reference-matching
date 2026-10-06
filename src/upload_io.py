"""
Parses an uploaded eDNA read file into (read_id, sequence) pairs.

Accepts FASTQ or FASTA, plain or gzip-compressed -- detected from the bytes,
never trusted from the filename, since uploads are routinely misnamed
(".fq" vs ".fastq", a gzip saved without ".gz").

There is deliberately no size or read-count limit: a file is refused only if
it is empty or is not FASTQ/FASTA. The file is streamed from disk and hashed
while it is parsed, so a large upload is never held in memory as bytes.

Accepted variations of the formats, all of which real files contain:
  * multi-line FASTQ (sequence and quality wrapped over several lines) and
    multi-line FASTA;
  * lower-case bases, RNA (U, read as T), IUPAC ambiguity codes, and
    alignment gap characters ("-", "."), which are dropped;
  * Windows line endings, blank lines between or after records, records
    with an empty sequence (they simply match nothing).

A structurally broken file -- e.g. FASTQ whose quality string is shorter than
its sequence, or text that is not sequence data -- is rejected with the line
it failed on, rather than half-parsed into a run that quietly describes only
part of the data.
"""
import gzip
import hashlib
import io
import re
from dataclasses import dataclass
from typing import BinaryIO, Iterator, List, Optional, Tuple

GZIP_MAGIC = b"\x1f\x8b"
_GAPS = str.maketrans("", "", "-.")
_VALID_SEQ = re.compile(r"^[ACGTNRYKMSWBDHV]*$")


class UploadError(ValueError):
    """The upload is not a usable read file. The message is shown to the user."""


@dataclass
class ParsedUpload:
    file_format: str            # "fastq" | "fasta"
    compressed: bool
    raw_sha256: str             # of the bytes as uploaded
    content_sha256: str         # of the decompressed bytes
    read_ids: List[str]
    sequences: List[str]


class _HashingReader(io.RawIOBase):
    """Wraps a binary stream and hashes every byte read through it."""

    def __init__(self, stream: BinaryIO):
        self.stream, self.h = stream, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, b):
        data = self.stream.read(len(b))
        self.h.update(data)
        b[:len(data)] = data
        return len(data)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_gzip(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def _clean_seq(seq: str, line_no: int) -> str:
    seq = seq.strip().upper().replace("U", "T").translate(_GAPS)
    if not _VALID_SEQ.match(seq):
        bad = sorted(set(seq) - set("ACGTNRYKMSWBDHV"))
        raise UploadError(f"line {line_no}: sequence contains characters that are not nucleotide codes: {bad}")
    return seq


def _lines(text: io.TextIOBase) -> Iterator[Tuple[int, str]]:
    for n, line in enumerate(text, 1):
        yield n, line.rstrip("\r\n")


def _parse_fastq(lines: Iterator[Tuple[int, str]], first: Tuple[int, str]):
    ids, seqs = [], []
    pending: Optional[Tuple[int, str]] = first
    while True:
        # header
        if pending is None:
            pending = next(lines, None)
            if pending is None:
                break
        n, header = pending
        pending = None
        if not header.strip():          # blank lines between records are harmless
            continue
        if not header.startswith("@"):
            raise UploadError(f"line {n}: expected a FASTQ record header starting with '@'")
        rid = header[1:].split()[0] if header[1:].strip() else f"read_{len(ids) + 1}"

        # sequence: one or more lines up to the '+' separator
        seq_parts, seq_line = [], n + 1
        while True:
            item = next(lines, None)
            if item is None:
                raise UploadError(f"line {n}: FASTQ record '{rid}' ends before its '+' separator line")
            if item[1].startswith("+"):
                break
            seq_parts.append(item[1].strip())
        seq = _clean_seq("".join(seq_parts), seq_line)
        raw_len = len("".join(seq_parts))

        # quality: read lines until it covers the sequence (handles wrapped FASTQ)
        qual_len, plus_line = 0, item[0]
        if raw_len == 0:
            # An empty read still has its (empty) quality line; consume it,
            # unless the file goes straight on to the next record.
            q = next(lines, None)
            if q is not None and q[1].strip():
                pending = q
        while qual_len < raw_len:
            q = next(lines, None)
            if q is None:
                raise UploadError(f"line {plus_line}: FASTQ record '{rid}' ends before its quality string is complete")
            qual_len += len(q[1].strip())
        if qual_len != raw_len:
            raise UploadError(f"line {plus_line + 1}: quality string length ({qual_len}) does not match the "
                              f"sequence length ({raw_len}) for read '{rid}'")
        ids.append(rid)
        seqs.append(seq)
    return ids, seqs


def _parse_fasta(lines: Iterator[Tuple[int, str]], first: Tuple[int, str]):
    ids, seqs, cur_id, cur, start = [], [], None, [], 0

    def flush():
        ids.append(cur_id)
        seqs.append(_clean_seq("".join(cur), start))

    for n, line in _chain(first, lines):
        line = line.strip()
        if not line or line.startswith(";"):          # ';' = legacy FASTA comment line
            continue
        if line.startswith(">"):
            if cur_id is not None:
                flush()
            cur_id = line[1:].split()[0] if line[1:].strip() else f"read_{len(ids) + 1}"
            cur, start = [], n
        else:
            if cur_id is None:
                raise UploadError(f"line {n}: sequence data before the first '>' header")
            cur.append(line)
    if cur_id is not None:
        flush()
    return ids, seqs


def _chain(first, rest):
    yield first
    yield from rest


def parse_stream(raw: BinaryIO, compressed: bool) -> Tuple[str, List[str], List[str], str]:
    """Parses a binary stream. Returns (format, read_ids, sequences, content_sha256)."""
    hashing = _HashingReader(gzip.GzipFile(fileobj=raw) if compressed else raw)
    buffered = io.BufferedReader(hashing, buffer_size=1 << 20)
    text = io.TextIOWrapper(buffered, encoding="ascii", errors="strict", newline=None)
    lines = _lines(text)
    try:
        first = next(((n, l) for n, l in lines if l.strip()), None)
        if first is None:
            raise UploadError("file is empty")
        if first[1].startswith("@"):
            fmt, (ids, seqs) = "fastq", _parse_fastq(lines, first)
        elif first[1].startswith(">") or first[1].startswith(";"):
            fmt, (ids, seqs) = "fasta", _parse_fasta(lines, first)
        else:
            raise UploadError("not a FASTQ file (records start with '@') or FASTA file (records start with '>')")
        buffered.read()                                  # hash any trailing bytes too
    except UnicodeDecodeError:
        raise UploadError("file is not plain-text FASTQ/FASTA (it contains non-text bytes)")
    except (OSError, EOFError) as e:
        raise UploadError(f"file looks gzip-compressed but could not be decompressed: {e}")
    if not ids:
        raise UploadError("file contains no reads")
    return fmt, ids, seqs, hashing.h.hexdigest()


def parse_upload_file(path: str) -> ParsedUpload:
    compressed = is_gzip(path)
    with open(path, "rb") as f:
        fmt, ids, seqs, content_sha = parse_stream(f, compressed)
    return ParsedUpload(file_format=fmt, compressed=compressed, raw_sha256=sha256_file(path),
                        content_sha256=content_sha, read_ids=ids, sequences=seqs)


def parse_upload(raw: bytes) -> ParsedUpload:
    """In-memory convenience wrapper (tests, small inputs)."""
    if not raw:
        raise UploadError("file is empty")
    compressed = raw.startswith(GZIP_MAGIC)
    fmt, ids, seqs, content_sha = parse_stream(io.BytesIO(raw), compressed)
    return ParsedUpload(file_format=fmt, compressed=compressed, raw_sha256=hashlib.sha256(raw).hexdigest(),
                        content_sha256=content_sha, read_ids=ids, sequences=seqs)
