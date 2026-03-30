from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/content/drive/MyDrive/latent_verbal_gap")
SRC_DIR       = BASE_DIR / "src"
CACHE_DIR     = BASE_DIR / "checkpoints"
DATA_DIR      = BASE_DIR / "data"

# Database paths (downloaded to VM, not Drive — too large to store on Drive)
AUTOINTERP_DB = Path("/root/math-autointerp.db")
DATASET_DB    = Path("/root/math-0-1.ddb")

# S3 URLs
AUTOINTERP_URL = "https://goodfire-r1-features.s3.us-east-1.amazonaws.com/math/autointerp.db"
DATASET_URL    = "https://goodfire-r1-features.s3.us-east-1.amazonaws.com/math/math-0-1.ddb"

# ── Reproducibility ────────────────────────────────────────────────────────────
RNG_SEED = 7180

# ── Step 1: Segmentation ───────────────────────────────────────────────────────
TOP_K_FEATURES      = 5
MIN_STEP_CHARS      = 20
ASSISTANT_ONLY      = True

# ── Step 2: Scoring ────────────────────────────────────────────────────────────
JUDGE_MODEL         = "claude-haiku-4-5"
JUDGE_MAX_TOKENS    = 150
JUDGE_RETRIES       = 3
LABEL_QUALITY_THRESHOLD = 0.9
N_SAMPLE_STEPS      = 2000
CONSISTENCY_FRAC    = 0.01
CHECKPOINT_EVERY    = 200

# ── Step 3: Analysis ──────────────────────────────────────────────────────────
H1_ALPHA            = 0.05
H1_MIN_COHENS_D     = 0.3
H3_SILENT_PERCENTILE = 5       # bottom 5th percentile = "silent"
H3_MIN_SEQUENCES    = 50       # feature must appear in >= 50 seqs to be evaluated

# ── Step 4: Trajectory ────────────────────────────────────────────────────────
N_REL_POS_BINS      = 10       # deciles for trajectory averaging

CACHE_DIR.mkdir(parents=True, exist_ok=True)