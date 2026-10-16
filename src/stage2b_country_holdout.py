"""Stage 2b: leave-one-country-out check.
Train the matcher on ONE country only, evaluate on the OTHER country.
This estimates how much F0.5 degrades on a country the model never saw in
training -- our best evidence-based proxy for how the model will behave on
France, which has zero training data.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

from er_common import learn_maps, apply_maps, load_df
from er_blocking import BlockCfg
from er_features import HAVE_RF

T0 = time.time()
def T(): return f"[{time.time()-T0:5.0f}s]"

# reuse stage2's exact functions so behavior matches exactly
import importlib.util
spec = importlib.util.spec_from_file_location("stage2", Path(__file__).parent / "stage2_train_eval.py")
stage2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--train-records", type=int, default=200000)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=150000)
    ap.add_argument("--k-combo", type=int, default=10)
    ap.add_argument("--k-name", type=int, default=3)
    ap.add_argument("--k-addr", type=int, default=3)
    a = ap.parse_args()

    W = Path(a.work)
    cfg = BlockCfg(k_combo=a.k_combo, k_name=a.k_name, k_addr=a.k_addr, k_char=5, use_char=False, qch=200)

    dev_s1, dev_rec = load_df(W / "dev_s1"), load_df(W / "dev_rec")
    val_s1, val_rec = load_df(W / "val_s1"), load_df(W / "val_rec")

    # merge dev+val back into one pool per country, since we're doing our own split now
    all_s1 = pd.concat([dev_s1, val_s1], ignore_index=True)
    all_rec = pd.concat([dev_rec, val_rec], ignore_index=True)

    countries = sorted(all_s1.country.unique())
    print(f"{T()} countries available: {countries}")

    for train_c in countries:
        test_c = [c for c in countries if c != train_c][0]
        print(f"\n{'='*70}\n{T()} TRAIN ON {train_c} -> TEST ON {test_c}\n{'='*70}")

        tr_s1 = all_s1[all_s1.country == train_c].copy()
        tr_rec = all_rec[all_rec.country == train_c].copy()
        te_s1 = all_s1[all_s1.country == test_c].copy()
        te_rec = all_rec[all_rec.country == test_c].copy()

        stage2.prep(tr_s1, tr_rec, None)
        stage2.prep(te_s1, te_rec, None)
        maps = stage2.learn_maps(tr_s1, tr_rec)  # maps learned ONLY from the training country
        stage2.docs(tr_s1, {}); stage2.docs(tr_rec, maps)
        stage2.docs(te_s1, {}); stage2.docs(te_rec, maps)  # apply train-country maps to test country -- this IS the France scenario

        tr = tr_rec.sample(min(a.train_records, len(tr_rec)), random_state=0)
        X, y, key = stage2.run_country(tr_s1, tr, cfg, a.batch, labels=True, tag=f"train-{train_c}")
        print(f"{T()} training set: {len(X):,} pairs, {y.mean():.3%} positive")
        model, cols = stage2.train_model(X, y, key)

        te_s1s, te_recs = stage2.subworld(te_s1, te_rec, a.val_frac)
        pr = stage2.run_country(te_s1s, te_recs, cfg, a.batch, model=model, cols=cols, tag=f"test-{test_c}")
        s1_index = pd.Index(te_s1s.entity_id.values)
        n_true = te_recs[te_recs.owner != ""].groupby("owner").size()

        print(f"\n=== CROSS-COUNTRY RESULT: trained on {train_c}, tested on {test_c} ===")
        rows = []
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for delta in (0.1, 0.2, 0.3, 0.5):
                acc = pr[(pr.p1 >= tau) & ((pr.p1 - pr.p2) >= delta)]
                f, *_ = stage2.macro_f05(acc, s1_index, n_true)
                rows.append((f, tau, delta))
        g = pd.DataFrame(rows, columns=["F0.5", "tau", "delta"]).sort_values("F0.5", ascending=False)
        print(g.head(10).to_string(index=False))
        best = g.iloc[0]
        print(f"\nBEST cross-country ({train_c}->{test_c}): F0.5={best['F0.5']:.4f} at tau={best.tau}, delta={best.delta}")
        print(f"(compare to same-country val F0.5 ~0.98 -- the gap estimates the France penalty)")


if __name__ == "__main__":
    main()