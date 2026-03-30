import duckdb
import json
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

from config import AUTOINTERP_DB, DATASET_DB, AUTOINTERP_URL, DATASET_URL, RNG_SEED
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ automatically

def download_databases(force: bool = False):
    """Download Goodfire DBs to VM local storage if not present."""
    for path, url in [(AUTOINTERP_DB, AUTOINTERP_URL), (DATASET_DB, DATASET_URL)]:
        if not path.exists() or force:
            print(f"Downloading {path.name} ...")
            urlretrieve(url, path)
            print(f"  → saved to {path}")
        else:
            print(f"  {path.name} already present, skipping download")


def connect_dbs() -> duckdb.DuckDBPyConnection:
    """Open autointerp.db and attach dataset.ddb as 'ds'."""
    conn = duckdb.connect(str(AUTOINTERP_DB), read_only=True)
    attached = {row[1] for row in conn.execute("PRAGMA database_list").fetchall()}
    if "ds" not in attached:
        conn.execute(f"ATTACH DATABASE '{str(DATASET_DB)}' as ds")
    return conn


def load_seq_ids(conn: duckdb.DuckDBPyConnection, cache_path: Path,
                 use_cache: bool = True, force: bool = False) -> list[int]:
    """Return list of sequence_ids that have SAE activation records."""
    if use_cache and not force and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            seq_ids = [int(x) for x in json.load(f)]
        print(f"Loaded {len(seq_ids)} seq_ids from cache")
        return seq_ids

    seq_ids = [
        int(r[0])
        for r in conn.execute(
            "SELECT DISTINCT sequence_id FROM ds.activations ORDER BY sequence_id"
        ).fetchall()
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(seq_ids, f)
    print(f"Saved {len(seq_ids)} seq_ids to {cache_path}")
    return seq_ids


def set_seeds():
    import random, torch
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)