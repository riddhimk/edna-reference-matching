"""
eDNA-TaxAI — Hugging Face Gradio Entrypoint

Zero-cost Gradio application for environmental DNA reference matching,
running on Hugging Face Spaces free-tier CPU Basic (sdk: gradio).
Also mounts the custom React dashboard and FastAPI backend under /dashboard and /api.
"""
import csv
import json
import os
import tempfile
import time
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd
from fastapi.staticfiles import StaticFiles

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, *args, **kwargs):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator

from src import chat
from src import run_analysis as ra
from src.api import ARCH_FILES, PRF_FILES, app as fastapi_app
from src.upload_io import UploadError, parse_upload_file

ROOT = os.path.dirname(os.path.abspath(__file__))

MARKER_CHOICES = {
    "Auto-detect (recommended)": "auto",
    "Fish — mitochondrial 16S (MitoFish)": "fish_16s_mitofish",
    "Bacteria/archaea — 16S V4-V5 (SILVA)": "bacteria_16s_v4v5",
    "Eukaryotic plankton — 18S V9 (SILVA)": "eukaryota_18s_v9",
}

SAMPLE_FISH_READ = (
    ">ERR16922870.1_fish_sample\n"
    "AGGACAGATTATATTAACTGCTCTTAACACAAGACGGAAGCGAAATACCCTCCTGTTCCAATGTCTTCGGTTGGGGCGACCACGGGGCAATACAAAACCCCCGCGCGGAATGGAAGCACC"
)

SAMPLE_BACTERIA_READ = (
    ">ERR6090675.1_tara_bacteria_sample\n"
    "GTGCCAGCCGCCGCGGTAATACGAAGGGACCTAGCGTAGTTCGGAATTACTGGGCTTAAAGAGTTCGTAGGTGGTTGAAATAGTTGGTGGTGAAATCCCAGAGCTTAACTCTGGAACTGCCATCAAAACTTTTCAGCTAGAGTTTGATAGAGGAAAGCAGAATTTCTAGTGTAGAGGTGAAATTCGTAGATATTAGAAAGAATACCAATTGCGAAGGCAGCGTTCTGGATCATTACCGAGACTGAGGAACG"
)


def parse_raw_text(text: str) -> Tuple[List[str], List[str]]:
    """Parses raw text containing FASTA, FASTQ, or newline-separated DNA sequences."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("Input sequence text is empty.")

    # Check for FASTA
    if any(l.startswith(">") for l in lines):
        ids, seqs, cur_id, cur_seq = [], [], None, []
        for line in lines:
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur_seq).upper())
                cur_id = line[1:].strip().split()[0] or f"seq_{len(ids)+1}"
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            ids.append(cur_id)
            seqs.append("".join(cur_seq).upper())
        return ids, seqs

    # Plain sequences, one per line
    seqs = [l.upper() for l in lines if set(l.upper()).issubset({"A", "C", "G", "T", "N"})]
    if not seqs:
        raise ValueError("No valid DNA sequence characters (A, C, G, T, N) found in input.")
    ids = [f"read_{i+1}" for i in range(len(seqs))]
    return ids, seqs


@spaces.GPU(duration=60)
def run_pipeline(
    file_obj,
    raw_text: str,
    marker_label: str,
    max_reads: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Executes the reference matching pipeline on uploaded reads."""
    read_ids, sequences = [], []
    if file_obj is not None:
        try:
            parsed = parse_upload_file(file_obj.name)
            read_ids = parsed.read_ids[:max_reads]
            sequences = parsed.sequences[:max_reads]
        except Exception as e:
            raise gr.Error(f"Error parsing file: {e}")
    elif raw_text and raw_text.strip():
        try:
            read_ids, sequences = parse_raw_text(raw_text)
            read_ids = read_ids[:max_reads]
            sequences = sequences[:max_reads]
        except Exception as e:
            raise gr.Error(f"Error parsing sequences: {e}")
    else:
        raise gr.Error("Please upload a FASTA/FASTQ file or paste sequence text.")

    if not sequences:
        raise gr.Error("No valid sequence records found to classify.")

    marker_key = MARKER_CHOICES.get(marker_label, "auto")
    detection = None

    if marker_key == "auto":
        progress(0.1, desc="Detecting marker gene region...")
        try:
            detection = ra.detect_marker(sequences)
            marker_key = detection["marker"]
        except Exception as e:
            raise gr.Error(f"Marker detection failed: {e}")

    marker_obj = ra.MARKERS.get(marker_key)
    if not marker_obj or not ra.marker_available(marker_obj):
        raise gr.Error(f"Reference database for marker '{marker_key}' is not available.")

    progress(0.2, desc=f"Matching reads against {marker_obj.label}...")

    def on_progress(done, total):
        if total > 0:
            progress(0.2 + 0.75 * (done / total), desc=f"Aligned {done}/{total} unique sequences")

    try:
        run_out = ra.run_marker(
            marker_key=marker_key,
            read_ids=read_ids,
            sequences=sequences,
            workers=1,
            on_progress=on_progress,
            resolve_species=True,
            detection=detection,
        )
    except Exception as e:
        raise gr.Error(f"Pipeline execution failed: {e}")

    summary = run_out.summary
    rows = run_out.rows

    dec_map = {d["decision"]: d for d in summary.get("decisions", []) if isinstance(d, dict)}
    assigned_pct = summary.get("assignment_rate_pct", 0.0)
    best_guess_pct = dec_map.get("best_guess", {}).get("pct_of_reads", 0.0)
    best_guess_low_pct = dec_map.get("best_guess_low_confidence", {}).get("pct_of_reads", 0.0)
    unclass_pct = dec_map.get("unclassified", {}).get("pct_of_reads", 0.0)

    assigned_count = dec_map.get("assigned", {}).get("reads", 0)
    best_guess_count = dec_map.get("best_guess", {}).get("reads", 0)
    best_guess_low_count = dec_map.get("best_guess_low_confidence", {}).get("reads", 0)
    unclass_count = dec_map.get("unclassified", {}).get("reads", 0)

    detect_info = ""
    if detection:
        detect_info = f"**Auto-detection**: Selected **{marker_obj.label}** ({detection.get('reason', '')})<br>"

    summary_md = f"""
### 🧬 Classification Results
{detect_info}
**Total Reads Processed**: {len(rows):,} | **Unique Sequences**: {summary.get('input', {}).get('n_unique_sequences_classified', len(rows)):,} | **Elapsed**: {run_out.seconds:.1f}s

| Metric | Assigned (≥95% ID) | Best Guess (75–95%) | Low Confidence (<75%) | Unclassified |
| :--- | :--- | :--- | :--- | :--- |
| **Share** | **{assigned_pct:.1f}%** | {best_guess_pct:.1f}% | {best_guess_low_pct:.1f}% | {unclass_pct:.1f}% |
| **Count** | {assigned_count:,} | {best_guess_count:,} | {best_guess_low_count:,} | {unclass_count:,} |
"""

    taxa_data = summary.get("taxa", {})
    top_orgs = taxa_data.get("top_assigned") or taxa_data.get("top_nearest_all_classified") or []
    top_records = []
    for item in top_orgs:
        top_records.append({
            "Organism / Species": item.get("taxon", ""),
            "Reads": item.get("reads", 0),
            "Share (%)": f"{item.get('pct_of_subset', 0):.1f}%",
            "Mean Identity (%)": f"{item.get('mean_identity', 0):.1f}%",
        })
    top_df = pd.DataFrame(top_records)
    if top_df.empty:
        top_df = pd.DataFrame(columns=["Organism / Species", "Reads", "Share (%)", "Mean Identity (%)"])

    reads_df = pd.DataFrame(rows)

    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="edna_results_")
    reads_df.to_csv(tmp_csv.name, index=False)

    return summary_md, top_df, reads_df, tmp_csv.name


def load_prf_table():
    records = []
    for k, rel in PRF_FILES.items():
        abs_p = os.path.join(ROOT, rel)
        if os.path.exists(abs_p):
            with open(abs_p, encoding="utf-8") as f:
                d = json.load(f)
                for r in d.get("results", []):
                    if r.get("n") == 5000:
                        records.append({
                            "Kingdom": k.capitalize(),
                            "Model Variant": r.get("arm", "").capitalize(),
                            "Precision": f"{r.get('precision', 0):.4f}",
                            "Recall": f"{r.get('recall', 0):.4f}",
                            "F1-Score": f"{r.get('f1', 0):.4f}",
                            "True Positives": r.get("tp", 0),
                        })
    return pd.DataFrame(records)


def load_arch_table():
    records = []
    for k, rel in ARCH_FILES.items():
        abs_p = os.path.join(ROOT, rel)
        if os.path.exists(abs_p):
            with open(abs_p, encoding="utf-8") as f:
                d = json.load(f)
                for a in d.get("architectures", []):
                    records.append({
                        "Kingdom": k.capitalize(),
                        "Architecture Arm": a.get("name", ""),
                        "Recall@10": f"{a.get('recall_at_10', 0):.4f}",
                        "Avg Distance / Score": f"{a.get('avg_score', 0):.4f}" if "avg_score" in a else "-",
                    })
    return pd.DataFrame(records)


def chat_respond(user_message: str, history: list, state_dict: dict):
    if not user_message or not user_message.strip():
        return "", history, state_dict

    session_id = state_dict.get("session_id")
    if not session_id:
        try:
            sess = chat.create_session(chat.NewSession(run_id=None))
            session_id = sess["id"]
            state_dict["session_id"] = session_id
        except Exception as e:
            reply = f"Chat unavailable: {e}"
            history.append((user_message, reply))
            return "", history, state_dict

    try:
        res = chat.post_message(session_id, chat.UserMessage(message=user_message))
        reply = res.get("reply", "No reply generated.")
    except Exception as e:
        reply = f"Error querying model: {e}\n(Tip: configure GEMINI_API_KEY in Hugging Face Space settings)."

    history.append((user_message, reply))
    return "", history, state_dict



def create_app():
    with gr.Blocks(title="eDNA-TaxAI") as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🧬 eDNA-TaxAI: Environmental DNA Reference Matching
                ### Production Hybrid Retrieval Pipeline (BM25 + MMR Diversification + Smith-Waterman Alignment)
                *Validated on real-world sequencing across Fish (MitoFish 16S), Bacteria (SILVA 16S V4–V5), and Eukaryota (SILVA 18S V9).*
                """
            )

        with gr.Tabs():
            # ---------------- TAB 1: RUN MATCHING ----------------
            with gr.Tab("🎯 Match Reads (Pipeline)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        marker_dropdown = gr.Dropdown(
                            choices=list(MARKER_CHOICES.keys()),
                            value="Auto-detect (recommended)",
                            label="Target Gene / Primer Preset",
                            info="Select the marker or let the system auto-detect from 12-mer frequency.",
                        )
                        file_input = gr.File(
                            label="Upload Reads (.fastq, .fasta, .fastq.gz, .fasta.gz)",
                            file_types=[".fastq", ".fasta", ".fq", ".fa", ".gz"],
                        )
                        text_input = gr.Textbox(
                            lines=4,
                            placeholder=">read_1\nACTG...\nOr paste plain ACGT DNA sequences here...",
                            label="Or Paste Sequence(s)",
                        )
                        with gr.Row():
                            fish_example_btn = gr.Button("🐟 Sample Fish Read", size="sm")
                            bact_example_btn = gr.Button("🧫 Sample Bacteria Read", size="sm")
                        max_reads_slider = gr.Slider(
                            minimum=10,
                            maximum=5000,
                            value=500,
                            step=10,
                            label="Max Reads to Process",
                            info="Limits read count for rapid interactive exploration.",
                        )
                        run_btn = gr.Button("🚀 Run Reference Matching", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        summary_output = gr.Markdown(value="*Upload reads and click **Run Reference Matching** to view results.*")
                        with gr.Tab("Top Organisms"):
                            top_orgs_output = gr.DataFrame(
                                headers=["Organism / Species", "Reads", "Share (%)", "Mean Identity (%)"],
                                interactive=False,
                            )
                        with gr.Tab("Per-Read Classification Table"):
                            reads_table_output = gr.DataFrame(interactive=False)
                        download_output = gr.File(label="Download Full Results (.csv)")

                fish_example_btn.click(lambda: SAMPLE_FISH_READ, outputs=text_input)
                bact_example_btn.click(lambda: SAMPLE_BACTERIA_READ, outputs=text_input)

                run_btn.click(
                    fn=run_pipeline,
                    inputs=[file_input, text_input, marker_dropdown, max_reads_slider],
                    outputs=[summary_output, top_orgs_output, reads_table_output, download_output],
                )

            # ---------------- TAB 2: BENCHMARKS ----------------
            with gr.Tab("📊 Benchmark Metrics"):
                gr.Markdown("### Precision, Recall & F1 Evaluations (Held-out 5,000 Read Sets)")
                gr.DataFrame(value=load_prf_table, interactive=False)

                gr.Markdown("### Architecture Comparison Across Kingdoms")
                gr.DataFrame(value=load_arch_table, interactive=False)

            # ---------------- TAB 3: AI ASSISTANT ----------------
            with gr.Tab("💬 AI Biology Assistant"):
                gr.Markdown("### Ask questions about the eDNA pipeline, benchmark findings, hubness, or primers.")
                chat_state = gr.State({})
                chatbot = gr.Chatbot(height=450)

                with gr.Row():
                    chat_msg = gr.Textbox(placeholder="Why is BM25 better than dense embeddings on short amplicons?", scale=4)
                    chat_send = gr.Button("Send", variant="primary", scale=1)

                chat_send.click(chat_respond, inputs=[chat_msg, chatbot, chat_state], outputs=[chat_msg, chatbot, chat_state])
                chat_msg.submit(chat_respond, inputs=[chat_msg, chatbot, chat_state], outputs=[chat_msg, chatbot, chat_state])

            # ---------------- TAB 4: METHODOLOGY ----------------
            with gr.Tab("🔬 Methodology & Findings"):
                gr.Markdown(
                    """
                    ### Core Empirical Findings
                    1. **Region-Specific Primers Matter**: A general embedding model fails when reference databases contain full-length genes while query reads are short amplicons. Calibrated primer trimming ensures queries and references match in length and orientation.
                    2. **BM25 Outperforms Pure Neural Embeddings**: On short DNA amplicons, exact k-mer matching (BM25) recalls far more true references than dense embeddings alone.
                    3. **MMR Eliminates Hubness**: Neural embeddings suffer from hub sequences that falsely attract thousands of unrelated reads. Maximal Marginal Relevance (MMR) diversifies the candidate set.
                    4. **Smith-Waterman Alignment Provides Calibrated Identity**: Matches are verified with true dynamic programming alignment. An assigned decision requires ≥95% identity.
                    """
                )

            # ---------------- TAB 5: FULL DASHBOARD ----------------
            with gr.Tab("🖥️ Full Dashboard"):
                gr.Markdown(
                    """
                    The full dark-mode instrument UI is also mounted at [`/dashboard`](/dashboard).
                    """
                )
                gr.HTML('<iframe src="/dashboard" style="width:100%; height:800px; border:1px solid #334155; border-radius:8px;"></iframe>')

        if os.path.isdir(os.path.join(ROOT, "web")):
            demo.app.mount("/dashboard", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="dashboard")
        demo.app.mount("/api", fastapi_app)

    return demo


demo = create_app()

if __name__ == "__main__":
    launch_kwargs = {"server_name": "0.0.0.0", "server_port": 7860}
    import inspect
    if "ssr_mode" in inspect.signature(demo.launch).parameters:
        launch_kwargs["ssr_mode"] = False
    demo.launch(**launch_kwargs)
