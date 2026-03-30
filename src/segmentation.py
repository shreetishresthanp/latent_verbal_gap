import re
import numpy as np
import pandas as pd
import duckdb
from tqdm.auto import tqdm

from config import TOP_K_FEATURES, MIN_STEP_CHARS, ASSISTANT_ONLY, RNG_SEED

# Step type keyword patterns
STEP_TYPE_PATTERNS = {
    "self_correction": re.compile(
        r"\b(wait|actually|i made a mistake|let me reconsider|"
        r"that('s| is) (wrong|incorrect)|i was wrong|"
        r"re-?check|re-?examine|recalculate|hmm,?\s+let me|"
        r"hold on|no,?\s+(that|this)|i think i|"
        r"let me re-?do|let me re-?think)\b",
        re.IGNORECASE,
    ),
    "uncertainty": re.compile(
        r"\b(not sure|unclear|i('m| am) not certain|"
        r"it's (possible|unclear)|maybe|perhaps|could be|"
        r"i (think|believe|suppose)|seems like|might be|"
        r"i('m| am) unsure)\b",
        re.IGNORECASE,
    ),
    "calculation": re.compile(
        r"(=\s*[\d\.\-]+|\d+\s*[\+\-\*\/]\s*\d+|"
        r"\\frac|\\sqrt|therefore\s+\d|"
        r"\b(compute|calculate|simplif|expand|substitut|plug in)\b)",
        re.IGNORECASE,
    ),
    "setup_planning": re.compile(
        r"\b(let('s| us)|let me (define|set|denote|introduce|"
        r"consider|note|observe|assume)|we (need|want|must|can)|"
        r"first,?|to (solve|find|determine)|the problem (asks|requires)|"
        r"approach|strategy|plan)\b",
        re.IGNORECASE,
    ),
}


def _ends_sentence_punct(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] in ("!", "?"):
        return True
    if stripped[-1] == "." and not re.search(r"\d\.$", stripped):
        return True
    return False


def classify_step_type(text: str) -> str:
    for stype, pat in STEP_TYPE_PATTERNS.items():
        if pat.search(text):
            return stype
    return "other"


def segment_trace(conn: duckdb.DuckDBPyConnection, seq_id: int,
                  assistant_only: bool = ASSISTANT_ONLY,
                  min_chars: int = MIN_STEP_CHARS) -> list[dict]:
    """
    Split one CoT trace into steps using newline + sentence punct boundaries.
    Returns list of dicts: {step_idx, start_token, end_token, text}.
    """
    tokens = conn.execute(
        "SELECT token_idx, decoded_token FROM ds.tokens "
        "WHERE sequence_id = ? ORDER BY token_idx",
        [seq_id],
    ).fetchall()
    if not tokens:
        return []

    start_pos = 0
    if assistant_only:
        for i, tok in enumerate(tokens):
            if "<｜Assistant｜>" in tok[1]:
                start_pos = i
                break

    work_pairs = tokens[start_pos:]
    if not work_pairs:
        return []

    boundary_starts = {work_pairs[0][0]}
    for i in range(len(work_pairs) - 1):
        cur_txt = work_pairs[i][1]
        start_next = work_pairs[i + 1][0]
        nxt_txt = work_pairs[i + 1][1]
        if "\n" in cur_txt and "\n" in nxt_txt:
            boundary_starts.add(start_next)
        elif _ends_sentence_punct(cur_txt):
            boundary_starts.add(start_next)

    boundaries = sorted(boundary_starts)
    max_idx = work_pairs[-1]
    token_dict = dict(work_pairs)

    SPECIAL = {"<｜Assistant｜>", "<｜User｜>", "<｜begin▁of▁sentence｜>",
               "<｜end▁of▁sentence｜>", "<think>", "</think>"}

    steps = []
    step_idx = 0
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else max_idx[0] + 1
        raw_text = "".join(token_dict.get(j, "") for j in range(start, end))
        clean_text = raw_text
        for s in SPECIAL:
            clean_text = clean_text.replace(s, " ")
        clean_text = clean_text.strip()
        if len(clean_text) < min_chars:
            continue
        steps.append({
            "step_idx":    step_idx,
            "start_token": start,
            "end_token":   end,
            "text":        clean_text,
        })
        step_idx += 1

    return steps


def extract_topk_features(conn: duckdb.DuckDBPyConnection, seq_id: int,
                           start_token: int, end_token: int,
                           k: int = TOP_K_FEATURES) -> list[dict]:
    """MAX-pool top-k SAE features over a token span."""
    rows = conn.execute(
        """
        SELECT a.feature_id, MAX(a.strength) AS max_strength, ai.label
        FROM ds.activations a
        JOIN autointerp ai ON a.feature_id = ai.feature_id
        WHERE a.sequence_id = ? AND a.token_idx >= ? AND a.token_idx < ?
        GROUP BY a.feature_id, ai.label
        ORDER BY max_strength DESC
        LIMIT ?
        """,
        [seq_id, start_token, end_token, k],
    ).fetchall()
    return [{"feature_id": int(f), "strength": float(s), "label": str(lab)}
            for f, s, lab in rows]


def build_step_feature_table(conn: duckdb.DuckDBPyConnection,
                              seq_ids: list[int],
                              k: int = TOP_K_FEATURES) -> pd.DataFrame:
    """
    Full Step 1 pipeline: segment all traces, extract top-k features per step,
    add relative step position and step type classification.
    Returns long-format DataFrame.
    """
    records = []
    for seq_id in tqdm(seq_ids, desc="Step 1: segmenting + extracting features"):
        steps = segment_trace(conn, seq_id)
        n_steps = len(steps)
        for s in steps:
            feats = extract_topk_features(conn, seq_id,
                                          s["start_token"], s["end_token"], k=k)
            if not feats:
                continue
            # tot = sum(f["strength"] for f in feats) or 1.0
            step_length = s["end_token"] - s["start_token"]
            tot = sum(f["strength"] / step_length for f in feats) or 1.0
            rel_pos = s["step_idx"] / (n_steps - 1) if n_steps > 1 else 0.0
            step_type = classify_step_type(s["text"])
            for rank, f in enumerate(feats, start=1):
                records.append({
                    "sequence_id":   seq_id,
                    "step_idx":      s["step_idx"],
                    "start_token":   s["start_token"],
                    "end_token":     s["end_token"],
                    "step_text":     s["text"],
                    "step_type":     step_type,
                    "rel_step_pos":  rel_pos,
                    "feature_rank":  rank,
                    "feature_id":    f["feature_id"],
                    "feature_label": f["label"],
                    "strength":      f["strength"],
                    "strength_w": (f["strength"] / step_length) / tot
                })

    df = pd.DataFrame.from_records(records)
    return df