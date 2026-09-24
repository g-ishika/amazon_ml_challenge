
"""Stage 0: deterministic split of the TRAIN data into 'worlds' (by S1 entity, via hash of the id).
   val world  : honest evaluation (never used to learn anything)
   dev world  : learn token maps / train the matcher
Each world = its S1 entities + every S2/S3 record owned by them + a same-rate random share of orphan records
(records matched to no S1), so orphan density matches the real data.  Writes parquet files to --work.
Requires: pandas pyarrow"""
import argparse, csv, time
from pathlib import Path
import numpy as np
import pandas as pd
from er_common import save_df


def read_tsv(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                       encoding="utf-8", encoding_errors="replace", on_bad_lines="warn", **kw)


def world_of(keys, val_pct, dev_pct):
    h = (pd.util.hash_pandas_object(pd.Series(keys), index=False).values % 100).astype(int)
    return np.where(h < val_pct, "val", np.where(h < val_pct + dev_pct, "dev", "rest"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder containing train/ and test/")
    ap.add_argument("--work", default="work")
    ap.add_argument("--val-pct", type=int, default=20)
    ap.add_argument("--dev-pct", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=500_000)
    a = ap.parse_args()
    TR, W = Path(a.data) / "train", Path(a.work)
    W.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    s1 = read_tsv(TR / "train_source1.tsv")
    gt = read_tsv(TR / "train_ground_truth.tsv")
    s1["world"] = world_of(s1.entity_id.values, a.val_pct, a.dev_pct)
    g = gt[gt.matched_entity_ids != ""]
    lst = g.assign(rec=g.matched_entity_ids.str.split(",")).explode("rec")
    owner = pd.Series(lst.source1_entity_id.values, index=lst.rec.values)
    assert owner.index.is_unique, "an S2/S3 record is matched to >1 S1 entity"
    del lst, g, gt
    print(f"[{time.time()-t0:.0f}s] S1 {len(s1):,} rows; owner map {len(owner):,} links")

    for w in ("val", "dev"):
        save_df(s1[s1.world == w].drop(columns="world"), W / f"{w}_s1")

    recs = {"val": [], "dev": []}
    for k in (2, 3):
        for ch in read_tsv(TR / f"train_source{k}.tsv", chunksize=a.chunk):
            own = ch.entity_id.map(owner)
            wd = world_of(own.fillna(ch.entity_id).values, a.val_pct, a.dev_pct)  # orphans hashed by own id
            ch["owner"] = own.fillna("")
            ch["src"] = f"S{k}"
            ch["world"] = wd
            for w in recs:
                recs[w].append(ch[ch.world == w].drop(columns="world"))
        print(f"[{time.time()-t0:.0f}s] streamed S{k}")
    for w in recs:
        df = pd.concat(recs[w], ignore_index=True)
        save_df(df, W / f"{w}_rec")
        s = s1[s1.world == w]
        print(f"\n== {w} world: S1 {len(s):,} | records {len(df):,} | orphan share {(df.owner=='').mean():.1%}")
        print(pd.crosstab(df.country, df.src).to_string())
        recs[w] = None
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {W.resolve()}")


if __name__ == "__main__":
    main()