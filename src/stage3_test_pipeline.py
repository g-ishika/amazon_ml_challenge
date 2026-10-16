"""Stage 3: run the trained matcher on the real test data.
Produces matching_results.tsv and candidate_pairs.tsv for submission.

Format (confirmed from validate_submission.py):
  matching_results.tsv  : source1_entity_id \t matched_entity_ids   (comma-joined)
  candidate_pairs.tsv   : source1_entity_id \t candidate_entity_ids (comma-joined)

v2 changes:
  - France uses the SAME tau/delta as US/India (leave-one-country-out check
    showed stricter thresholds do not recover cross-country F0.5 loss --
    the drop comes from distribution shift, not overconfidence).
  - Writes both output files as a CHECKPOINT after every country finishes,
    not just once at the end, so a crash never loses completed work again.
"""
import argparse, pickle, time, json
from pathlib import Path
import numpy as np
import pandas as pd

from er_common import norm_tokens, norm_name_tokens, apply_maps
from er_blocking import BlockCfg, PoolIndex
from er_features import pair_features, record_side_info, HAVE_RF

T0 = time.time()
def T(): return f"[{time.time()-T0:8.1f}s]"


def read_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def prep(s1, rec):
    for df in (s1, rec):
        df["n_t"] = df.business_name.map(norm_name_tokens)
        df["a_t"] = df.business_address.map(norm_tokens)


def docs(df, maps):
    n, a = apply_maps(df, maps)
    df["n_doc"], df["a_doc"] = n, a


def run_country_test(P, Q, cfg, batch, model, cols, tag=""):
    """P: S1 pool df, Q: test record df (S2+S3 combined, with n_doc/a_doc/src).
    Returns (top_df, pairs_df):
      top_df   - one row per record: best S1 candidate id, p1, p2
      pairs_df - EVERY scored (S1, record) pair, for candidate_pairs.tsv"""
    P = P.reset_index(drop=True); Q = Q.reset_index(drop=True)
    idx = PoolIndex(P.n_doc.values, P.a_doc.values, cfg)
    Pn, Pa = P.n_doc.values.astype(object), P.a_doc.values.astype(object)
    info_all = record_side_info(Q.business_name.values, Q.business_address.values,
                                 Q.n_doc.values, Q.a_doc.values, Q.src.values)
    all_pairs, top_rows = [], []
    for s in range(0, len(Q), batch):
        e = min(len(Q), s + batch)
        Qb = Q.iloc[s:e]
        qm = idx.transform_queries(Qb.n_doc.values, Qb.a_doc.values)
        qi, pj, R = idx.retrieve(qm)
        info = {k: v[s:e] for k, v in info_all.items()}
        X = pair_features(idx, qm, Qb.n_doc.values.astype(object), Qb.a_doc.values.astype(object),
                           info, Pn, Pa, qi, pj, R)
        p = model.predict_proba(X[cols])[:, 1].astype(np.float32)

        rec_ids = Qb.entity_id.values[qi]
        s1_ids = P.entity_id.values[pj]
        all_pairs.append(pd.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_id": rec_ids, "score": p}))

        d = pd.DataFrame({"qi": qi, "pj": pj, "p": p}).sort_values(["qi", "p"], ascending=[True, False])
        d["r"] = d.groupby("qi").cumcount()
        first = d[d.r == 0].set_index("qi"); second = d[d.r == 1].set_index("qi")
        t = pd.DataFrame(index=np.arange(e - s))
        t["pj1"] = first.pj.reindex(t.index).fillna(-1).astype(int)
        t["p1"] = first.p.reindex(t.index).fillna(0.0)
        t["p2"] = second.p.reindex(t.index).fillna(0.0)
        t["row"] = np.arange(s, e)
        top_rows.append(t)
        print(f"   {T()} {tag} {e:,}/{len(Q):,} records | {len(qi):,} pairs ({len(qi)/(e-s):.1f}/record)", flush=True)

    pairs_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else \
        pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])
    t = pd.concat(top_rows, ignore_index=True)
    top_df = pd.DataFrame({
        "rec_id": Q.entity_id.values[t.row],
        "s1_id": np.where(t.pj1.values >= 0, P.entity_id.values[np.maximum(t.pj1.values, 0)], ""),
        "p1": t.p1.values, "p2": t.p2.values,
    })
    return top_df, pairs_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to .../dataset (has test/)")
    ap.add_argument("--work", required=True, help="path with matcher_v1.pkl, token_maps_v2.json")
    ap.add_argument("--out", default="../output")
    ap.add_argument("--batch", type=int, default=100000)
    ap.add_argument("--limit", type=int, default=None, help="records per country, for a smoke test")
    args = ap.parse_args()

    W = Path(args.work); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

    with open(W / "matcher_v1.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, cols, cfg = bundle["model"], bundle["cols"], bundle["cfg"]
    BASE_TAU, BASE_DELTA = bundle["tau"], bundle["delta"]
    print(f"{T()} loaded matcher: val_F05={bundle.get('val_F05')}, tau={BASE_TAU}, delta={BASE_DELTA}")

    maps = json.loads((W / "token_maps_v2.json").read_text(encoding="utf-8"))

    D = Path(args.data) / "test"
    s1 = read_tsv(D / "test_source1.tsv")
    s2 = read_tsv(D / "test_source2.tsv"); s2["src"] = "S2"
    s3 = read_tsv(D / "test_source3.tsv"); s3["src"] = "S3"
    rec = pd.concat([s2, s3], ignore_index=True)

    if args.limit:
        rec = rec.groupby("country", group_keys=False).apply(lambda d: d.head(args.limit), include_groups=False)

    print(f"{T()} test S1 {len(s1):,} | records {len(rec):,}")
    print(s1.country.value_counts().to_string())

    prep(s1, rec)
    docs(s1, {})     # S1 pool: no map applied (matches stage2's docs(s1, {}))
    docs(rec, maps)  # records: apply the learned maps (matches stage2's docs(rec, maps))
    print(f"{T()} docs built")

    matches = {sid: [] for sid in s1.entity_id}
    candidates = {sid: [] for sid in s1.entity_id}

    for country in sorted(s1.country.unique()):
        P = s1[s1.country == country]
        Q = rec[rec.country == country]
        if P.empty or Q.empty:
            continue

        # v2: same tau/delta for every country -- leave-one-country-out check
        # showed a stricter France threshold does not recover cross-country
        # F0.5 loss, so there is no evidence-based reason to penalize it here.
        tau, delta = BASE_TAU, BASE_DELTA
        print(f"{T()} [{country}] pool {len(P):,}, records {len(Q):,}, tau={tau}, delta={delta}")

        top_df, pairs_df = run_country_test(P, Q, cfg, args.batch, model, cols, tag=f"test-{country}")

        for s1_id, grp in pairs_df.groupby("source1_entity_id"):
            candidates[s1_id].extend(grp.candidate_entity_id.tolist())
        del pairs_df

        accepted = top_df[(top_df.p1 >= tau) & ((top_df.p1 - top_df.p2) >= delta) & (top_df.s1_id != "")]
        for _, r in accepted.iterrows():
            matches[r.s1_id].append(r.rec_id)
        del top_df

        # checkpoint: write BOTH files after every country, so a later crash
        # never loses already-completed work again.
        res = pd.DataFrame({
            "source1_entity_id": list(matches.keys()),
            "matched_entity_ids": [",".join(v) for v in matches.values()],
        })
        res.to_csv(OUT / "matching_results.tsv", sep="\t", index=False)

        cand = pd.DataFrame({
            "source1_entity_id": list(candidates.keys()),
            "candidate_entity_ids": [",".join(v) for v in candidates.values()],
        })
        cand.to_csv(OUT / "candidate_pairs.tsv", sep="\t", index=False)
        print(f"{T()} [{country}] checkpoint saved to {OUT}/")

    n_matched = (res.matched_entity_ids != "").sum()
    print(f"{T()} done. {n_matched:,}/{len(res):,} S1 entities got >=1 match "
          f"({n_matched/len(res):.1%}). Files in {OUT}/")


if __name__ == "__main__":
    main()