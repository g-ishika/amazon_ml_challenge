
"""Stage 2: train the pair matcher on the DEV world, then evaluate the whole pipeline on the VAL world.
 pipeline per country:  S2/S3 record -> retrieve S1 candidates -> pair features -> GBDT probability
 decision:              accept the record's best S1 candidate if p1 >= tau and (p1 - p2) >= delta   (one S1 owner per record)
 metric:                macro F0.5 over all S1 entities of the val (sub)world, singletons included.
Needs: pandas numpy scipy scikit-learn rapidfuzz lightgbm   (falls back to sklearn HistGradientBoosting without lightgbm)."""
import argparse, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
from er_common import (norm_tokens, norm_name_tokens, learn_maps, apply_maps, load_df, save_json)
from er_blocking import BlockCfg, PoolIndex
from er_features import pair_features, record_side_info, HAVE_RF

T0 = time.time()
def T(): return f"[{time.time()-T0:5.0f}s]"


def prep(s1, rec, maps):
    for df in (s1, rec):
        df["n_t"] = df.business_name.map(norm_name_tokens)
        df["a_t"] = df.business_address.map(norm_tokens)


def docs(df, maps):
    n, a = apply_maps(df, maps)
    df["n_doc"], df["a_doc"] = n, a


def subworld(s1, rec, frac, seed=0):
    if frac >= 1:
        return s1, rec
    s1s = s1.sample(frac=frac, random_state=seed)
    ids = set(s1s.entity_id)
    orph = rec[rec.owner == ""].sample(frac=frac, random_state=seed)
    return s1s, pd.concat([rec[rec.owner.isin(ids)], orph])


def run_country(P, Q, cfg, batch, model=None, cols=None, labels=False, tag=""):
    """P: S1 pool df, Q: record df (both with n_doc/a_doc).  labels=True -> return (X, y, key);  else -> per-record top-2 table."""
    P = P.reset_index(drop=True); Q = Q.reset_index(drop=True)
    idx = PoolIndex(P.n_doc.values, P.a_doc.values, cfg)
    pid = {e: i for i, e in enumerate(P.entity_id)}
    own = Q.owner.map(pid).fillna(-1).astype(np.int64).values
    Pn, Pa = P.n_doc.values.astype(object), P.a_doc.values.astype(object)
    info_all = record_side_info(Q.business_name.values, Q.business_address.values, Q.n_doc.values, Q.a_doc.values, Q.src.values)
    outX, outY, outK, top = [], [], [], []
    for s in range(0, len(Q), batch):
        e = min(len(Q), s + batch)
        Qb = Q.iloc[s:e]
        qm = idx.transform_queries(Qb.n_doc.values, Qb.a_doc.values)
        qi, pj, R = idx.retrieve(qm)
        info = {k: v[s:e] for k, v in info_all.items()}
        X = pair_features(idx, qm, Qb.n_doc.values.astype(object), Qb.a_doc.values.astype(object), info, Pn, Pa, qi, pj, R)
        hit = (pj == own[s:e][qi])
        if labels:
            outX.append(X); outY.append(hit.astype(np.int8)); outK.append(qi.astype(np.int64) + s)
        else:
            p = model.predict_proba(X[cols])[:, 1].astype(np.float32)
            d = pd.DataFrame({"qi": qi, "pj": pj, "p": p, "hit": hit}).sort_values(["qi", "p"], ascending=[True, False])
            d["r"] = d.groupby("qi").cumcount()
            first = d[d.r == 0].set_index("qi"); second = d[d.r == 1].set_index("qi")
            t = pd.DataFrame(index=np.arange(e - s))
            t["pj1"] = first.pj.reindex(t.index).fillna(-1).astype(int)
            t["p1"] = first.p.reindex(t.index).fillna(0.0)
            t["p2"] = second.p.reindex(t.index).fillna(0.0)
            t["n_cand"] = d.groupby("qi").size().reindex(t.index).fillna(0).astype(int)
            t["cand_hit"] = d.groupby("qi").hit.any().reindex(t.index).fillna(False)
            t["row"] = np.arange(s, e)
            top.append(t)
        print(f"   {T()} {tag} {e:,}/{len(Q):,} records | {len(qi):,} pairs ({len(qi)/(e-s):.1f}/record)", flush=True)
    if labels:
        return pd.concat(outX, ignore_index=True), np.concatenate(outY), np.concatenate(outK)
    t = pd.concat(top, ignore_index=True)
    out = pd.DataFrame({"rec_id": Q.entity_id.values[t.row], "owner": Q.owner.values[t.row], "country": Q.country.values[t.row],
                        "s1_id": np.where(t.pj1.values >= 0, P.entity_id.values[np.maximum(t.pj1.values, 0)], ""),
                        "p1": t.p1.values, "p2": t.p2.values, "n_cand": t.n_cand.values, "cand_hit": t.cand_hit.values,
                        "row": t.row.values})
    return out


def macro_f05(acc, s1_index, n_true):
    n_pred = acc.groupby("s1_id").size().reindex(s1_index, fill_value=0).values
    tp = acc[acc.s1_id == acc.owner].groupby("s1_id").size().reindex(s1_index, fill_value=0).values
    nt = n_true.reindex(s1_index, fill_value=0).values
    Pr = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
    Rc = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
    F = np.where((Pr + Rc) > 0, 1.25 * Pr * Rc / np.maximum(0.25 * Pr + Rc, 1e-12), 0.0)
    F = np.where((nt == 0) & (n_pred == 0), 1.0, F)
    return F.mean(), F, n_pred, tp, nt


def train_model(X, y, key):
    cols = list(X.columns)
    is_va = (key * 2654435761 % 100) < 10
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.08, num_leaves=63, min_child_samples=50, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
        m.fit(X[~is_va], y[~is_va], eval_set=[(X[is_va], y[is_va])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        print(f"{T()} LightGBM trained, best_iteration={m.best_iteration_}")
        imp = sorted(zip(m.feature_importances_, cols), reverse=True)[:15]
        print("   top features (split count):", ", ".join(f"{c}:{i}" for i, c in imp))
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("[warn] lightgbm not installed -> sklearn HistGradientBoosting fallback (pip install lightgbm recommended)")
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, random_state=0).fit(X, y)
    return m, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--train-records", type=int, default=300000)
    ap.add_argument("--val-frac", type=float, default=0.25, help="<1: evaluate on a random sub-world (smaller pool => slightly optimistic)")
    ap.add_argument("--char", action="store_true", help="add the char-ngram retrieval channel")
    ap.add_argument("--qch", type=int, default=200)
    ap.add_argument("--batch", type=int, default=150000)
    ap.add_argument("--k-combo", type=int, default=10)
    ap.add_argument("--k-name", type=int, default=3)
    ap.add_argument("--k-addr", type=int, default=3)
    ap.add_argument("--k-char", type=int, default=5)
    a = ap.parse_args()
    if not HAVE_RF:
        print("[warn] rapidfuzz not installed -> very slow fallback string metrics. pip install rapidfuzz")
    W = Path(a.work)
    cfg = BlockCfg(k_combo=a.k_combo, k_name=a.k_name, k_addr=a.k_addr, k_char=a.k_char, use_char=a.char, qch=a.qch)

    dev_s1, dev_rec = load_df(W / "dev_s1"), load_df(W / "dev_rec")
    val_s1, val_rec = load_df(W / "val_s1"), load_df(W / "val_rec")
    print(T(), f"dev S1 {len(dev_s1):,} rec {len(dev_rec):,} | val S1 {len(val_s1):,} rec {len(val_rec):,} | cfg {cfg}")

    # ---------------- tokenise, learn maps on DEV, build docs
    prep(dev_s1, dev_rec, None); prep(val_s1, val_rec, None)
    maps = learn_maps(dev_s1, dev_rec)
    save_json(maps, W / "token_maps_v2.json")
    for s1, rec in ((dev_s1, dev_rec), (val_s1, val_rec)):
        docs(s1, {}); docs(rec, maps)
    print(T(), "docs built")

    # ---------------- training data from DEV
    tr = dev_rec.sample(min(a.train_records, len(dev_rec)), random_state=0)
    Xs, ys, ks, off = [], [], [], 0
    for c in sorted(dev_s1.country.unique()):
        P, Q = dev_s1[dev_s1.country == c], tr[tr.country == c]
        print(T(), f"TRAIN candidates [{c}]: pool {len(P):,}, records {len(Q):,}")
        X, y, k = run_country(P, Q, cfg, a.batch, labels=True, tag=f"train-{c}")
        Xs.append(X); ys.append(y); ks.append(k + off); off += len(Q)
    X, y, key = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ks)
    print(T(), f"training set: {len(X):,} pairs, {y.mean():.3%} positive, {X.shape[1]} features")
    model, cols = train_model(X, y, key)
    del X, y, key, Xs, ys, ks

    # ---------------- evaluate on VAL
    vs1, vrec = subworld(val_s1, val_rec, a.val_frac)
    print(T(), f"VAL (sub)world: S1 {len(vs1):,}, records {len(vrec):,}, orphan share {(vrec.owner=='').mean():.1%}")
    preds = []
    for c in sorted(vs1.country.unique()):
        P, Q = vs1[vs1.country == c], vrec[vrec.country == c]
        print(T(), f"VAL scoring [{c}]: pool {len(P):,}, records {len(Q):,}")
        preds.append(run_country(P, Q, cfg, a.batch, model=model, cols=cols, tag=f"val-{c}"))
    pr = pd.concat(preds, ignore_index=True)
    s1_index = pd.Index(vs1.entity_id.values)
    n_true = vrec[vrec.owner != ""].groupby("owner").size()

    matched = pr.owner != ""
    print(f"\n=== BLOCKING (final candidate set fed to the model) ===")
    print(f"avg candidates/record {pr.n_cand.mean():.1f} | p95 {int(pr.n_cand.quantile(.95))} | "
          f"recall ceiling (true S1 among candidates, matched records): {pr.cand_hit[matched].mean():.4f}")
    print(pr[matched].groupby("country").cand_hit.mean().round(4).to_string())

    print("\n=== DECISION GRID: macro F0.5 over val S1 entities (tau = min p1, delta = min p1-p2) ===")
    rows = []
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        for delta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            acc = pr[(pr.p1 >= tau) & ((pr.p1 - pr.p2) >= delta)]
            f, *_ = macro_f05(acc, s1_index, n_true)
            rows.append((f, tau, delta, len(acc)))
    g = pd.DataFrame(rows, columns=["F0.5", "tau", "delta", "accepted"]).sort_values("F0.5", ascending=False)
    print(g.head(10).round(4).to_string(index=False))
    best = g.iloc[0]; tau, delta = float(best.tau), float(best.delta)
    acc = pr[(pr.p1 >= tau) & ((pr.p1 - pr.p2) >= delta)]
    fval, F, n_pred, tp, nt = macro_f05(acc, s1_index, n_true)
    print(f"\nBEST  tau={tau} delta={delta}  ->  macro F0.5 = {fval:.4f}")
    print(f"record-level: precision {(acc.s1_id==acc.owner).mean():.4f} | recall {(acc.s1_id==acc.owner).sum()/matched.sum():.4f}")
    sing = nt == 0
    print(f"singleton S1 entities: {sing.mean():.1%} of S1, mean F {F[sing].mean():.4f} | non-singleton mean F {F[~sing].mean():.4f}")
    cs = pd.Series(F, index=s1_index).groupby(vs1.set_index("entity_id").country.reindex(s1_index).values).mean().round(4)
    print("F0.5 by country:", cs.to_dict())

    s1t = vs1.set_index("entity_id"); rt = vrec.set_index("entity_id")
    fp = acc[(acc.s1_id != acc.owner)].sample(min(8, int((acc.s1_id != acc.owner).sum())), random_state=0)
    print("\n--- sample FALSE MERGES (record | accepted S1 | truth) ---")
    for _, r in fp.iterrows():
        q = rt.loc[r.rec_id]; s = s1t.loc[r.s1_id]
        print(f"  p1={r.p1:.2f} p2={r.p2:.2f} | REC {q.business_name[:45]} | {q.business_address[:50]}\n"
              f"        -> S1 {s.business_name[:45]} | {s.business_address[:50]}   (record is {'an orphan' if r.owner=='' else 'owned by another S1'})")
    fn = pr[matched & ~pr.rec_id.isin(acc[acc.s1_id == acc.owner].rec_id)]
    fn = fn.sample(min(8, len(fn)), random_state=0)
    print("\n--- sample MISSED matches (matched record not correctly accepted) ---")
    for _, r in fn.iterrows():
        q = rt.loc[r.rec_id]; s = s1t.loc[r.owner]
        print(f"  p1={r.p1:.2f} in_candidates={bool(r.cand_hit)} | REC {q.business_name[:45]} | {q.business_address[:50]}\n"
              f"        truth S1 {s.business_name[:45]} | {s.business_address[:50]}")

    with open(W / "matcher_v1.pkl", "wb") as fh:
        pickle.dump({"model": model, "cols": cols, "cfg": cfg, "tau": tau, "delta": delta, "val_F05": float(fval)}, fh)
    print(f"\n{T()} saved {W/'matcher_v1.pkl'}  (val macro F0.5 {fval:.4f})")


if __name__ == "__main__":
    main()