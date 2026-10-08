"""
Project-specific analysis chatbot.

What makes it specific to this project rather than a generic assistant:

  * Its system prompt carries the project's own write-up -- README.md and the
    full Explanation.md (~28k tokens) -- so methodology questions ("why BM25
    before DNABERT?", "what is hubness?") are answered from what this project
    actually found, not general knowledge. That block never changes between
    requests, so it is prompt-cached: paid once, then read at ~1/20 the price.
  * Numbers come from tools that read the same sources as the dashboard:
    uploaded runs and their per-read results, and the saved benchmark files.
    The prompt tells it to fetch rather than recall any figure.
  * A chat session can be tied to a run, so "which fish did we find?" means
    the run the user is looking at.

Two model providers, chosen per server by EDNA_CHAT_PROVIDER (default: Gemini
when a Gemini key is available, else Claude). Both get the same instructions,
documentation and tools; each conversation records which provider it was
started with, because each stores history in its own format and neither can
replay the other's.

Conversation history is stored append-only (src/run_store.py) with the full
content -- thinking blocks / thought signatures included, since both
providers require them to be passed back unchanged. Nothing is stored for a
turn that fails, so a failed request never leaves a half-written exchange
(e.g. a tool call with no result) behind.

Configuration (environment):
    EDNA_CHAT_PROVIDER  -- gemini | anthropic (default: auto, as above)
    Gemini:  GEMINI_API_KEY, or the "api key: ..." line of api-key.txt in the
             project root (gitignored; EDNA_GEMINI_KEY_FILE to move it)
             EDNA_GEMINI_MODEL -- default gemini-3.8-flash
             EDNA_GEMINI_FALLBACKS -- default gemini-3.5-flash,gemini-3.1-flash-lite
    Claude:  ANTHROPIC_API_KEY, or any credential the Anthropic SDK resolves
             EDNA_CHAT_MODEL   -- default claude-opus-5-5
             EDNA_CHAT_EFFORT  -- low | medium | high | xhigh | max (default medium)
"""
import json
import os
import threading
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import anthropic
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as gtypes
    HAS_GEMINI = True
except ImportError:
    genai = None
    genai_errors = None
    gtypes = None
    HAS_GEMINI = False

class _DummyGenAIError(Exception):
    code: int = 0
    message: str = ""

_GenAIAPIError = genai_errors.APIError if HAS_GEMINI else _DummyGenAIError

from . import run_analysis as ra
from .final_pipeline import BEST_GUESS_MIN_IDENTITY, BM25_K_MER, FROZEN_IDENTITY_THRESHOLD
from .app_routes import _get_run_or_404, analytics_compare, run_reads, state

router = APIRouter(prefix="/chat", tags=["chat"])

MODEL = os.environ.get("EDNA_CHAT_MODEL", "claude-opus-5-5")
GEMINI_MODEL = os.environ.get("EDNA_GEMINI_MODEL", "gemini-3.8-flash")
# Tried in order when a model is overloaded (503), out of quota (429 -- free-tier
# quotas are per model), or retired for this key (404, as gemini-2.5-flash is).
GEMINI_FALLBACKS = [m for m in os.environ.get("EDNA_GEMINI_FALLBACKS", "gemini-3.5-flash,gemini-3.1-flash-lite").split(",") if m]
_FALLBACK_CODES = (404, 429, 503)
EFFORT = os.environ.get("EDNA_CHAT_EFFORT", "medium")
MAX_TOKENS = 16000
MAX_TOOL_ROUNDS = 10
TOOL_RESULT_CHARS = 30000   # a single tool result larger than this is cut, and says so

_INSTRUCTIONS = """\
You are the analysis assistant for eDNA-TaxAI, a research project that identifies which species a short \
environmental-DNA read came from by matching it against reference databases (BM25 k-mer retrieval -> MMR \
diversification on DNABERT-S embeddings -> Smith-Waterman identity check -> tiered decision at a frozen {thr}% \
identity threshold). You talk to the researchers who run it, inside its results dashboard.

Answer questions about this project: files users uploaded and the runs analysing them, the project's \
benchmark results, its methodology, and how to interpret findings. The project's own documentation is \
below -- treat it as the authority on what was done, measured and concluded. If a question is outside this \
project (general coding help, unrelated science, chit-chat), say briefly that you only cover this \
project's analyses and offer what you can help with.

Get every number from a tool, not from memory or the documentation's prose, whenever a tool can supply it -- \
the documentation describes past experiments, while the tools read the current result files and the \
user's uploads. Name the run id or benchmark a number came from. If a tool cannot answer something, say so \
plainly rather than estimating.

Interpretation the project has established, to apply when relevant:
- Decisions: "assigned" (>={thr}% identity) is a confident call; "best_guess" ({low}-{thr}%) is a lead, not an \
identification; "best_guess_low_confidence" (<{low}%) is only the nearest known relative; "unclassified" means \
no reference shared a {k}-mer; "no_primer" reads were excluded before matching.
- A reference set only works for reads from the same gene region and primers (the region-mismatch \
finding). A low assignment rate often means the wrong marker or a sparse reference set, not an absent species.
- "Hubness": a few references absorbing matches from many unrelated reads. A high top-10 reference share \
among assigned reads is worth flagging.
- MitoFish references have no species names; fish names come from an NCBI accession lookup and can be \
missing.

Keep answers short and concrete; researchers read them beside the charts. Use a small table when comparing \
several numbers. Write plain Markdown -- no LaTeX or math notation (write ">= {thr}%", not a formula).""".format(thr=f"{FROZEN_IDENTITY_THRESHOLD:g}", low=f"{BEST_GUESS_MIN_IDENTITY:g}", k=BM25_K_MER)


@lru_cache(maxsize=1)
def system_blocks() -> List[dict]:
    """Static, so it caches. Never put per-request values (dates, run ids) in here."""
    docs = []
    for name in ("README.md", "Explanation.md"):
        path = os.path.join(ra.ROOT, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                docs.append(f"<document name=\"{name}\">\n{f.read()}\n</document>")
    return [{"type": "text", "text": _INSTRUCTIONS},
            {"type": "text", "text": "Project documentation:\n\n" + "\n\n".join(docs),
             "cache_control": {"type": "ephemeral"}}]


# ---------------------------------------------------------------- tools ---

def _t_list_runs(limit: int = 20, marker: Optional[str] = None) -> Any:
    runs, total = state.get_store().list_runs(marker=marker, limit=max(1, min(int(limit), 100)))
    return {"total_runs": total, "runs": [{
        "run_id": r["id"], "filename": r["filename"], "marker": r["marker"], "status": r["status"],
        "created_at": r["created_at"], "error": r["error"],
        "n_reads": (r["summary"] or {}).get("input", {}).get("n_reads"),
        "assignment_rate_pct": (r["summary"] or {}).get("assignment_rate_pct")} for r in runs]}


def _t_get_run_summary(run_id: str) -> Any:
    r = _get_run_or_404(run_id)
    return {"run_id": r["id"], "filename": r["filename"], "marker": r["marker"], "status": r["status"],
            "progress": f"{r['progress_done']}/{r['progress_total']}", "error": r["error"],
            "created_at": r["created_at"], "summary": r["summary"]}


def _t_query_run_reads(run_id: str, decision: Optional[str] = None, min_identity: Optional[float] = None,
                       taxon: Optional[str] = None, limit: int = 20) -> Any:
    return run_reads(run_id, decision=decision, min_identity=min_identity, taxon=taxon,
                     limit=max(1, min(int(limit), 50)), offset=0)


def _t_compare_runs(run_ids: List[str]) -> Any:
    out = analytics_compare(",".join(run_ids))
    out.pop("charts", None)
    return out


def _t_get_benchmark(kind: str, kingdom: Optional[str] = None) -> Any:
    from . import api
    if kind == "precision_recall_f1":
        return api.metrics_prf(kingdom=kingdom, n=None, arm=None)
    if kind == "frozen_vs_finetuned":
        return api.metrics_prf_comparison(5000)
    if kind == "architecture":
        return api.metrics_architecture(kingdom)
    if kind == "embedding_models":
        return api.models_comparison()
    if kind == "checkpoints":
        return api.models()
    if kind == "reference_real_read_runs":
        return api.read_datasets()
    raise ValueError(f"unknown kind '{kind}'")


def _t_get_reference_run(dataset: str) -> Any:
    from . import api
    return api.reads_summary(dataset, 15)


def _t_list_markers() -> Any:
    return ra.describe_markers()


TOOLS: List[dict] = [
    {"name": "list_runs",
     "description": "List the user's uploaded files and their analysis runs, newest first: run id, filename, "
                    "marker, status, read count, assignment rate.",
     "input_schema": {"type": "object", "properties": {
         "limit": {"type": "integer", "description": "max runs to return, 1-100 (default 20)"},
         "marker": {"type": "string", "description": "only runs for this marker key"}}}},
    {"name": "get_run_summary",
     "description": "Full findings for one run: input stats, primer detection, decision counts, identity "
                    "histogram and medians, top taxa, lineage composition, reference hubness share, runtime.",
     "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}},
    {"name": "query_run_reads",
     "description": "Individual read results from a finished run, filtered. Returns the total matching count and "
                    "up to `limit` rows (read id, nearest reference, organism, taxonomy, identity %, coverage %, decision).",
     "input_schema": {"type": "object", "properties": {
         "run_id": {"type": "string"},
         "decision": {"type": "string", "enum": ra.DECISIONS},
         "min_identity": {"type": "number", "description": "minimum identity %, 0-100"},
         "taxon": {"type": "string", "description": "case-insensitive substring of organism name or taxonomy"},
         "limit": {"type": "integer", "description": "rows to return, 1-50 (default 20)"}},
         "required": ["run_id"]}},
    {"name": "compare_runs",
     "description": "Compare 2-10 finished runs: per-run decision rates and medians, a taxon-by-run count matrix "
                    "of assigned reads, and pairwise taxon overlap (shared, unique, Jaccard).",
     "input_schema": {"type": "object", "properties": {
         "run_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 10}},
         "required": ["run_ids"]}},
    {"name": "get_benchmark",
     "description": "The project's saved benchmark results. kinds: precision_recall_f1 (POC-6 held-out P/R/F1 per "
                    "kingdom at n=1000/2000/5000, frozen and fine-tuned); frozen_vs_finetuned (side by side at "
                    "n=5000); architecture (7-arm retrieval comparison: recall@k, assignment, accuracy, time); "
                    "embedding_models (DNABERT-S vs Nucleotide Transformer, HyenaDNA, GENA-LM recall); "
                    "checkpoints (fine-tuning metadata per model); reference_real_read_runs (the saved real-read "
                    "runs from earlier project parts).",
     "input_schema": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["precision_recall_f1", "frozen_vs_finetuned", "architecture",
                                             "embedding_models", "checkpoints", "reference_real_read_runs"]},
         "kingdom": {"type": "string", "description": "for precision_recall_f1: bacteria | fish | eukaryota_large; "
                                                      "for architecture: bacteria | fish | eukaryota"}},
         "required": ["kind"]}},
    {"name": "get_reference_run",
     "description": "Summary of one saved real-read run from earlier project parts (decision counts, score "
                    "distribution, top organisms). Dataset slugs come from get_benchmark(kind=reference_real_read_runs).",
     "input_schema": {"type": "object", "properties": {"dataset": {"type": "string"}}, "required": ["dataset"]}},
    {"name": "list_markers",
     "description": "The marker presets uploads can be analysed against: reference database, primers, and the "
                    "real-data run each was validated on.",
     "input_schema": {"type": "object", "properties": {}}},
]

HANDLERS: Dict[str, Callable[..., Any]] = {
    "list_runs": _t_list_runs, "get_run_summary": _t_get_run_summary, "query_run_reads": _t_query_run_reads,
    "compare_runs": _t_compare_runs, "get_benchmark": _t_get_benchmark, "get_reference_run": _t_get_reference_run,
    "list_markers": _t_list_markers,
}


def run_tool(name: str, args: dict) -> (str, bool):
    """Returns (content, is_error). Errors go back to the model as tool errors
    so it can correct itself (e.g. a mistyped run id), not to the user."""
    fn = HANDLERS.get(name)
    if fn is None:
        return f"unknown tool '{name}'", True
    try:
        out = json.dumps(fn(**(args or {})), default=str)
    except HTTPException as e:
        return f"error: {e.detail}", True
    except (TypeError, ValueError) as e:
        return f"error: invalid arguments -- {e}", True
    if len(out) > TOOL_RESULT_CHARS:
        out = out[:TOOL_RESULT_CHARS] + f"... [truncated: result was {len(out):,} characters; narrow the query]"
    return out, False


# ------------------------------------------------------------- the loop ---

_client: Optional[anthropic.Anthropic] = None
_session_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _session_lock(sid: str) -> threading.Lock:
    with _locks_guard:
        return _session_locks.setdefault(sid, threading.Lock())


class ChatNotConfigured(Exception):
    """No credentials for the active provider. The message is shown to the user."""


def _create(messages: List[dict]):
    try:
        return _send(messages)
    except TypeError as e:
        # The SDK raises a bare TypeError, not an AnthropicError, when it finds
        # no credentials at all. Narrowed to that case so a genuine TypeError
        # from a bug still surfaces as one.
        if "authentication" in str(e).lower():
            raise ChatNotConfigured("set ANTHROPIC_API_KEY (or log in with `ant auth login`) for the API "
                                    "server, then restart it") from e
        raise


def _send(messages: List[dict]):
    return get_client().beta.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_blocks(),
        tools=TOOLS,
        messages=messages,
        output_config={"effort": EFFORT},
        cache_control={"type": "ephemeral"},          # also caches the growing conversation
        # If a safety classifier declines a turn, the API re-runs it on a
        # recommended fallback model instead of returning a bare refusal.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def converse_anthropic(history: List[dict], user_text: str) -> dict:
    """Runs one user turn to completion. Returns the new messages to append
    and the reply. Raises anthropic errors to the caller untouched."""
    new: List[dict] = [{"role": "user", "content": user_text}]
    tools_used: List[dict] = []
    rounds = 0
    while True:
        resp = _create(history + new)
        new.append({"role": "assistant", "content": [b.to_dict() for b in resp.content]})

        if resp.stop_reason != "tool_use":
            break
        rounds += 1
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            if rounds > MAX_TOOL_ROUNDS:
                content, err = "tool budget for this question is used up; answer with what you have", True
            else:
                content, err = run_tool(b.name, b.input)
            tools_used.append({"tool": b.name, "input": b.input, "error": err})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": err})
        new.append({"role": "user", "content": results})
        if rounds > MAX_TOOL_ROUNDS + 1:
            break

    text = "\n\n".join(b.text for b in resp.content if b.type == "text").strip()
    if resp.stop_reason == "refusal":
        text = text or "I can't help with that request."
    elif resp.stop_reason == "max_tokens":
        text += "\n\n[The answer was cut off at the length limit -- ask a narrower question.]"
    return {"new_messages": new, "reply": text, "tools_used": tools_used, "stop_reason": resp.stop_reason,
            "model": resp.model}



# -------------------------------------------------------------- Gemini ---

def gemini_api_key() -> Optional[str]:
    """GEMINI_API_KEY, else api-key.txt: either a bare key, or a labelled
    "api key: ..." line. Never logged or returned by any endpoint."""
    env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env:
        return env.strip()
    path = os.environ.get("EDNA_GEMINI_KEY_FILE", os.path.join(ra.ROOT, "api-key.txt"))
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        label, sep, value = line.partition(":")
        if sep and label.strip().lower() in ("api key", "api_key", "apikey", "key", "gemini api key"):
            return value.strip() or None
    return lines[0] if len(lines) == 1 and ":" not in lines[0] else None


def active_provider() -> str:
    p = os.environ.get("EDNA_CHAT_PROVIDER", "auto").lower()
    if p in ("gemini", "anthropic"):
        return p
    return "gemini" if (HAS_GEMINI and gemini_api_key()) else "anthropic"


def provider_model(provider: str) -> str:
    return GEMINI_MODEL if provider == "gemini" else MODEL


_gemini_client: Optional[Any] = None


def get_gemini_client() -> Any:
    global _gemini_client
    if not HAS_GEMINI:
        raise ChatNotConfigured("google-genai package is not installed; install google-genai or use anthropic")
    if _gemini_client is None:
        key = gemini_api_key()
        if not key:
            raise ChatNotConfigured("no Gemini API key: set GEMINI_API_KEY or add an \"api key: ...\" line to "
                                    "api-key.txt in the project root, then restart the server")
        _gemini_client = genai.Client(api_key=key, http_options=gtypes.HttpOptions(
            retry_options=gtypes.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=8.0,
                                                  http_status_codes=[429, 500, 503, 504])))
    return _gemini_client


@lru_cache(maxsize=1)
def gemini_config() -> Any:
    if not HAS_GEMINI:
        return None
    return gtypes.GenerateContentConfig(
        system_instruction="\n\n".join(b["text"] for b in system_blocks()),
        tools=[gtypes.Tool(function_declarations=[
            gtypes.FunctionDeclaration(name=t["name"], description=t["description"],
                                       parameters_json_schema=t["input_schema"]) for t in TOOLS])],
        # The loop below runs tools itself, so it can record them, cap them,
        # and store the exact turns Gemini produced.
        automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True),
    )


def _gemini_send(contents: list):
    if not HAS_GEMINI:
        raise ChatNotConfigured("google-genai package is not installed; install google-genai or use anthropic")
    # The client has already retried with backoff; a model still overloaded,
    # out of quota or retired is skipped for the next, rather than failing the turn.
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    for i, model in enumerate(models):
        try:
            return get_gemini_client().models.generate_content(model=model, contents=contents, config=gemini_config())
        except _GenAIAPIError as e:
            if e.code not in _FALLBACK_CODES or i == len(models) - 1:
                raise


def _dump(content: Any) -> dict:
    # mode="json" base64-encodes thought signatures, which model_validate restores.
    return content.model_dump(mode="json", exclude_none=True)


def converse_gemini(history: List[dict], user_text: str) -> dict:
    """Gemini twin of converse_anthropic. History entries hold a dumped
    gtypes.Content in "content"."""
    if not HAS_GEMINI:
        raise ChatNotConfigured("google-genai package is not installed; install google-genai or use anthropic")
    contents = [gtypes.Content.model_validate(m["content"]) for m in history]
    new = [gtypes.Content(role="user", parts=[gtypes.Part(text=user_text)])]
    tools_used: List[dict] = []
    rounds = 0
    while True:
        resp = _gemini_send(contents + new)
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None:
            reason = (resp.prompt_feedback.block_reason if resp.prompt_feedback else None) or \
                     (cand.finish_reason if cand else "no candidate")
            text = f"I can't answer that (the model returned no content: {reason})."
            return {"new_messages": [], "reply": text, "tools_used": tools_used, "stop_reason": str(reason),
                    "model": GEMINI_MODEL, "store": False}
        new.append(cand.content)

        calls = [p.function_call for p in (cand.content.parts or []) if p.function_call]
        if not calls:
            break
        rounds += 1
        parts = []
        for fc in calls:
            args = dict(fc.args or {})
            if rounds > MAX_TOOL_ROUNDS:
                content, err = "tool budget for this question is used up; answer with what you have", True
            else:
                content, err = run_tool(fc.name, args)
            tools_used.append({"tool": fc.name, "input": args, "error": err})
            parts.append(gtypes.Part(function_response=gtypes.FunctionResponse(
                id=fc.id, name=fc.name, response={"error": content} if err else {"result": content})))
        new.append(gtypes.Content(role="user", parts=parts))
        if rounds > MAX_TOOL_ROUNDS + 1:
            break

    text = "\n\n".join(p.text for p in (cand.content.parts or []) if p.text and not p.thought).strip()
    finish = str(cand.finish_reason.name if hasattr(cand.finish_reason, "name") else cand.finish_reason)
    if finish == "MAX_TOKENS":
        text += "\n\n[The answer was cut off at the length limit -- ask a narrower question.]"
    elif not text:
        text = f"(no answer -- the model stopped with {finish})"
    return {"new_messages": [{"role": c.role, "content": _dump(c)} for c in new], "reply": text,
            "tools_used": tools_used, "stop_reason": finish, "model": resp.model_version or GEMINI_MODEL}


# ------------------------------------------------------------- endpoints ---

class NewSession(BaseModel):
    run_id: Optional[str] = Field(None, description="tie the conversation to this run")


class UserMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


def _visible(messages: List[dict]) -> List[dict]:
    """The transcript as a person reads it: user questions and assistant
    text, without tool plumbing or thinking. Handles both providers' formats."""
    out = []
    for m in messages:
        c = m["content"]
        role = "assistant" if m["role"] in ("assistant", "model") else m["role"]
        if isinstance(c, str):                                   # Claude user turn
            text = c
        elif isinstance(c, dict):                                # Gemini Content
            text = "\n\n".join(p.get("text", "") for p in c.get("parts", [])
                               if p.get("text") and not p.get("thought")).strip()
        else:                                                    # Claude content blocks
            text = "\n\n".join(b.get("text", "") for b in c if b.get("type") == "text").strip()
        if role == "user":
            text = text.split("\n\n[Context:")[0]
        if text:
            out.append({"role": role, "text": text, "created_at": m["created_at"]})
    return out


@router.get("/status", summary="Which model answers, and whether it is configured")
def chat_status():
    p = active_provider()
    configured = bool(gemini_api_key()) if p == "gemini" else bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return {"provider": p, "model": provider_model(p), "configured": configured}


@router.post("/sessions", summary="Start a conversation, optionally about one run")
def create_session(body: NewSession):
    if body.run_id:
        _get_run_or_404(body.run_id)
    return state.get_store().create_chat_session(body.run_id, active_provider())


@router.get("/sessions/{session_id}", summary="A conversation's transcript")
def get_session(session_id: str):
    store = state.get_store()
    s = store.get_chat_session(session_id)
    if not s:
        raise HTTPException(404, f"no chat session '{session_id}'")
    return {**s, "messages": _visible(store.get_chat_messages(session_id))}


@router.post("/sessions/{session_id}/messages", summary="Ask a question; returns the assistant's answer")
def post_message(session_id: str, body: UserMessage):
    store = state.get_store()
    s = store.get_chat_session(session_id)
    if not s:
        raise HTTPException(404, f"no chat session '{session_id}'")

    lock = _session_lock(session_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "this conversation is still answering the previous question")
    try:
        history = [{"role": m["role"], "content": m["content"]} for m in store.get_chat_messages(session_id)]
        text = body.message.strip()
        if s["run_id"] and not history:
            run = store.get_run(s["run_id"])
            if run:
                text += (f"\n\n[Context: I am looking at run {run['id']} -- file '{run['filename']}', "
                         f"marker {run['marker']}, status {run['status']}.]")
        provider = s.get("provider") or "anthropic"
        if provider != active_provider():
            raise HTTPException(409, f"this conversation was started with {provider}, but the server now uses "
                                     f"{active_provider()}; start a new conversation")
        try:
            out = (converse_gemini if provider == "gemini" else converse_anthropic)(history, text)
        except ChatNotConfigured as e:
            raise HTTPException(503, f"the chatbot is not configured: {e}")
        # Gemini
        except _GenAIAPIError as e:
            if e.code in (401, 403) or (e.code == 400 and "api key" in str(e).lower()):
                raise HTTPException(503, "the Gemini API key was rejected; check api-key.txt / GEMINI_API_KEY")
            if e.code == 429:
                raise HTTPException(429, "Gemini quota or rate limit reached; try again in a minute")
            raise HTTPException(502, f"the Gemini API returned an error ({e.code}): {e.message}")
        except httpx.TransportError:
            raise HTTPException(502, "could not reach the Gemini API")
        # Claude
        except anthropic.AuthenticationError:
            raise HTTPException(503, "the chatbot's Anthropic API credentials were rejected")
        except anthropic.PermissionDeniedError as e:
            raise HTTPException(503, f"the chatbot's API key lacks access: {e.message}")
        except anthropic.RateLimitError:
            raise HTTPException(429, "the chatbot is rate-limited right now; try again in a minute")
        except anthropic.BadRequestError as e:
            raise HTTPException(502, f"the model API rejected the request: {e.message}")
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"the model API returned an error ({e.status_code}); try again")
        except anthropic.APIConnectionError:
            raise HTTPException(502, "could not reach the model API")
        except anthropic.AnthropicError as e:
            raise HTTPException(503, f"the chatbot is not configured: {e}. Set ANTHROPIC_API_KEY on the server.")
        if out.get("store", True):
            store.append_chat_messages(session_id, out["new_messages"])
        return {"session_id": session_id, "reply": out["reply"], "tools_used": out["tools_used"],
                "stop_reason": out["stop_reason"], "model": out["model"], "provider": provider}
    finally:
        lock.release()
