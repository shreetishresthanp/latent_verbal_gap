import json
import time
import numpy as np
import pandas as pd
import anthropic
from tqdm.auto import tqdm
from pathlib import Path
from scipy import stats as scipy_stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import (JUDGE_MODEL, JUDGE_MAX_TOKENS, JUDGE_RETRIES,
                    LABEL_QUALITY_THRESHOLD, N_SAMPLE_STEPS,
                    CONSISTENCY_FRAC, CHECKPOINT_EVERY, RNG_SEED,
                    H1_ALPHA, H1_MIN_COHENS_D)

CLIENT = anthropic.Anthropic()

import json
import time
import numpy as np
import pandas as pd
import anthropic
from tqdm.auto import tqdm
from pathlib import Path
from scipy import stats as scipy_stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import (JUDGE_MODEL, JUDGE_MAX_TOKENS, JUDGE_RETRIES,
                    LABEL_QUALITY_THRESHOLD, N_SAMPLE_STEPS,
                    CONSISTENCY_FRAC, CHECKPOINT_EVERY, RNG_SEED,
                    H1_ALPHA, H1_MIN_COHENS_D)

CLIENT = anthropic.Anthropic()

JUDGE_SYSTEM = """You are evaluating whether a SAE feature label semantically corresponds to a reasoning step from a math chain-of-thought trace.

"Corresponds" means: given this step's content, would you expect this feature to be active? Be strict — a generic label like "mathematical reasoning" should NOT score highly just because both involve math.

Score guide:
0.0–0.2  No relation. Feature describes something entirely absent from this step.
0.3–0.4  Generic overlap only. Label technically applies but isn't specifically triggered by this step.
0.5–0.6  Relevant but imprecise. Label is related but not specific to this step's operation.
0.7–0.8  Strong match. Label clearly describes something present in this step.
0.9–1.0  Precise match. Label describes the dominant operation in this step.

Confidence:
"high"   — correspondence is unambiguous
"medium" — step or label is somewhat ambiguous
"low"    — truncated step, very abstract label, or genuine uncertainty

Return ONLY valid JSON: {"score": float, "confidence": str, "reasoning": str}
One sentence max for reasoning. No preamble, no markdown."""


def judge_pair(step_text: str, feature_label: str) -> dict:
    client = anthropic.Anthropic()
    prompt = (
        f"Reasoning step:\n\"\"\"\n{step_text[:800]}\n\"\"\"\n\n"
        f"SAE feature label: \"{feature_label}\"\n\n"
        f"Does this feature label correspond to this reasoning step?"
    )
    for attempt in range(JUDGE_RETRIES):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=JUDGE_MAX_TOKENS,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if not raw:
                raise ValueError("Empty response from model")
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            assert "score" in parsed and "confidence" in parsed
            parsed["score"] = float(np.clip(parsed["score"], 0.0, 1.0))
            return parsed
        except Exception as e:
            if attempt == JUDGE_RETRIES - 1:
                return {"score": -1.0, "confidence": "error", "reasoning": str(e)}
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s


def stratified_step_sample(step_feat_df: pd.DataFrame,
                            n: int = N_SAMPLE_STEPS,
                            seed: int = RNG_SEED) -> pd.DataFrame:
    """Stratified sample of steps by (step_type x rel_pos_decile)."""
    rng = np.random.default_rng(seed)
    step_summary = (
        step_feat_df[["sequence_id", "step_idx", "step_type", "rel_step_pos"]]
        .drop_duplicates(["sequence_id", "step_idx"])
        .copy()
    )
    step_summary["decile"] = pd.cut(
        step_summary["rel_step_pos"], bins=10, labels=False
    )
    step_summary["stratum"] = (step_summary["step_type"].astype(str)
                                + "_" + step_summary["decile"].astype(str))
    counts = step_summary["stratum"].value_counts()
    total = len(step_summary)
    sampled = []
    for stratum, count in counts.items():
        k = max(1, round(n * count / total))
        rows = step_summary[step_summary["stratum"] == stratum]
        sampled.append(rows.sample(min(k, len(rows)),
                                   random_state=int(rng.integers(1e6))))
    result = pd.concat(sampled).drop_duplicates(["sequence_id", "step_idx"])
    if len(result) > n:
        result = result.sample(n, random_state=42)
    return result[["sequence_id", "step_idx"]].reset_index(drop=True)


def run_scoring(scoring_df: pd.DataFrame,
                cache_path: Path,
                use_cache: bool = True,
                force: bool = False,
                max_workers: int = 3) -> pd.DataFrame:
    """Run judge on all rows, with checkpointing and threading."""
    print("Local Run scoring called")
    if use_cache and not force and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"Loaded scores from cache: {len(df):,} rows")
        return df

    # Load existing checkpoint if partial run exists
    checkpoint_tmp = cache_path.parent / (cache_path.stem + "_tmp.parquet")
    completed = {}
    if checkpoint_tmp.exists():
        tmp = pd.read_parquet(checkpoint_tmp)
        for _, r in tmp.iterrows():
            completed[(r["sequence_id"], r["step_idx"], r["feature_id"])] = r.to_dict()
        print(f"Resuming from checkpoint: {len(completed)} rows already done")

    rows = list(completed.values())
    remaining = scoring_df[
        ~scoring_df.apply(
            lambda r: (r["sequence_id"], r["step_idx"], r["feature_id"]) in completed,
            axis=1
        )
    ]

    def score_row(row_tuple):
        _, row = row_tuple
        result = judge_pair(row["step_text"], row["feature_label"])
        return {
            "sequence_id":   row["sequence_id"],
            "step_idx":      row["step_idx"],
            "rel_step_pos":  row["rel_step_pos"],
            "step_type":     row["step_type"],
            "feature_id":    row["feature_id"],
            "feature_label": row["feature_label"],
            "feature_rank":  row["feature_rank"],
            "strength":      row["strength"],
            "strength_w":    row["strength_w"],
            "lva_score":     result["score"],
            "confidence":    result["confidence"],
            "reasoning":     result.get("reasoning", ""),
            "step_text": row["step_text"]
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_row, r) for r in remaining.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="LVA scoring"):
            rows.append(future.result())
            if len(rows) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_parquet(checkpoint_tmp, index=False)

    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    if checkpoint_tmp.exists():
        os.remove(checkpoint_tmp)
    print(f"Saved {len(df):,} scored rows")
    return df


def compute_step_lva(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(step,feature) scores to per-step LVA score."""
    return (
        scores_df[scores_df["lva_score"] >= 0]
        .groupby(["sequence_id", "step_idx"])
        .apply(lambda g: pd.Series({
            "lva_score":          (g["strength_w"] * g["lva_score"]).sum()
                                  / g["strength_w"].sum(),
            "n_features":         len(g),
            "rel_step_pos":       g["rel_step_pos"].iloc[0],
            "step_type":          g["step_type"].iloc[0],
            "low_conf_rate":      (g["confidence"] == "low").mean(),
        }))
        .reset_index()
    )


def compute_baselines(scores_df: pd.DataFrame,
                      seed: int = RNG_SEED) -> pd.DataFrame:
    """Compute all three baselines, return merged DataFrame."""
    rng = np.random.default_rng(seed + 1)
    clean = scores_df[scores_df["lva_score"] >= 0].copy()

    # Random-feature baseline
    rand = clean.copy()
    rand["lva_score"] = clean["lva_score"].iloc[
        rng.permutation(len(clean))].values
    rand_lva = (rand.groupby(["sequence_id", "step_idx"])
                .apply(lambda g: (g["strength_w"] * g["lva_score"]).sum()
                                  / g["strength_w"].sum())
                .reset_index(name="lva_random"))

    # Step-shuffle baseline
    def shift_steps(g):
        g = g.copy()
        steps = sorted(g["step_idx"].unique())
        m = {s: steps[(i+1) % len(steps)] for i, s in enumerate(steps)}
        g["step_idx"] = g["step_idx"].map(m)
        return g
    shuf = clean.groupby("sequence_id", group_keys=False).apply(shift_steps)
    shuf_lva = (shuf.groupby(["sequence_id", "step_idx"])
                .apply(lambda g: (g["strength_w"] * g["lva_score"]).sum()
                                  / g["strength_w"].sum())
                .reset_index(name="lva_shuffled"))

    # Cross-problem baseline
    cross = clean.copy()
    cross["decile"] = pd.cut(cross["rel_step_pos"], bins=10, labels=False)
    def shuffle_seqs(g):
        g = g.copy()
        ids = g["sequence_id"].unique()
        if len(ids) < 2:
            return g
        perm = rng.permutation(ids)
        g["sequence_id"] = g["sequence_id"].map(dict(zip(ids, perm)))
        return g
    cross = cross.groupby("decile", group_keys=False).apply(shuffle_seqs)
    cross_lva = (cross.groupby(["sequence_id", "step_idx"])
                 .apply(lambda g: (g["strength_w"] * g["lva_score"]).sum()
                                   / g["strength_w"].sum())
                 .reset_index(name="lva_cross"))

    step_lva = compute_step_lva(clean)
    return (step_lva
            .merge(rand_lva,  on=["sequence_id", "step_idx"], how="left")
            .merge(shuf_lva,  on=["sequence_id", "step_idx"], how="left")
            .merge(cross_lva, on=["sequence_id", "step_idx"], how="left"))


def run_h1_tests(h1_df: pd.DataFrame) -> None:
    """Print H1 statistical test results."""
    real = h1_df["lva_score"].dropna().values

    def cohens_d(a, b):
        pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
        return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

    print("\n" + "="*55)
    print("H1: Real LVA vs baselines")
    print("="*55)
    for col, label in [("lva_random",   "Random-feature"),
                       ("lva_shuffled", "Step-shuffle"),
                       ("lva_cross",    "Cross-problem")]:
        base = h1_df[col].dropna().values
        t, p = scipy_stats.ttest_ind(real, base, equal_var=False)
        d = cohens_d(real, base)
        sig = "✓ SIGNIFICANT" if p < H1_ALPHA and d > H1_MIN_COHENS_D else "✗ not significant"
        print(f"\n{label}:  real={np.mean(real):.3f}  base={np.mean(base):.3f}"
              f"  t={t:.3f}  p={p:.4f}  d={d:.3f}  {sig}")

def run_consistency_check(scoring_df: pd.DataFrame,
                           cache_path: Path,
                           use_cache: bool = True,
                           force: bool = False) -> float:
    """Score 10% sample twice independently, return Pearson r between runs."""

    n = max(50, int(len(scoring_df) * CONSISTENCY_FRAC))
    sample = scoring_df.sample(n, random_state=RNG_SEED + 99).reset_index(drop=True)

    _run1_path = cache_path.parent / "lva_consistency_run1.parquet"
    _run2_path = cache_path

    def score_sample(path, desc):
        if use_cache and not force and path.exists():
            df = pd.read_parquet(path)
            print(f"Loaded {desc} from cache ({len(df)} rows)")
            return df
        rows = []
        def score_row(row_tuple):
            _, row = row_tuple
            result = judge_pair(row["step_text"], row["feature_label"])
            return {
                "sequence_id":  row["sequence_id"],
                "step_idx":     row["step_idx"],
                "feature_id":   row["feature_id"],
                "lva_score":    result["score"],
            }
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(score_row, r) for r in sample.iterrows()]
            rows = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                rows.append(future.result())
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return df

    run1 = score_sample(_run1_path, "Consistency run 1")
    run2 = score_sample(_run2_path, "Consistency run 2")

    merged = run1.merge(
        run2.rename(columns={"lva_score": "lva_score_r2"}),
        on=["sequence_id", "step_idx", "feature_id"],
        how="inner"
    )
    valid = merged[(merged["lva_score"] >= 0) & (merged["lva_score_r2"] >= 0)]
    r, p = scipy_stats.pearsonr(valid["lva_score"], valid["lva_score_r2"])
    print(f"\nJudge consistency (n={len(valid)}):  r={r:.3f}  p={p:.2e}")
    if r < 0.8:
        print("  ⚠ r < 0.8 — reduce temperature or tighten rubric before full run")
    else:
        print("  ✓ r >= 0.8 — judge reliable, proceed")
    return r


def apply_label_filter(step_feat_df: pd.DataFrame,
                       conn) -> pd.DataFrame:
    """Filter to features with label quality >= threshold."""
    quality = conn.execute(
        "SELECT feature_id, quality FROM autointerp"
    ).df().set_index("feature_id")["quality"].to_dict()
    step_feat_df = step_feat_df.copy()
    step_feat_df["label_quality"] = step_feat_df["feature_id"].map(quality)
    filtered = step_feat_df[step_feat_df["label_quality"] >= LABEL_QUALITY_THRESHOLD]
    print(f"Label filter (quality >= {LABEL_QUALITY_THRESHOLD}): "
          f"{len(step_feat_df):,} → {len(filtered):,} rows "
          f"({100*len(filtered)/len(step_feat_df):.1f}% retained)")
    return filtered


def flag_generic_features(scores_df: pd.DataFrame,
                           ceiling: float = 0.35) -> pd.DataFrame:
    """
    Add is_generic column based on per-feature mean LVA.
    Features with mean LVA <= ceiling are flagged as generic.
    Call this after run_scoring completes.
    """
    feature_mean = (
        scores_df[scores_df["lva_score"] >= 0]
        .groupby("feature_id")["lva_score"]
        .mean()
        .rename("feature_mean_lva")
    )
    scores_df = scores_df.merge(feature_mean, on="feature_id", how="left")
    scores_df["is_generic"] = scores_df["feature_mean_lva"] <= ceiling
    n_generic  = scores_df[scores_df["lva_score"] >= 0]["is_generic"].sum()
    n_total    = (scores_df["lva_score"] >= 0).sum()
    print(f"Generic pairs  (mean_lva <= {ceiling}): {n_generic:,} ({100*n_generic/n_total:.1f}%)")
    print(f"Specific pairs (mean_lva >  {ceiling}): {n_total-n_generic:,} ({100*(n_total-n_generic)/n_total:.1f}%)")
    return scores_df