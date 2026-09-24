
"""Stage 1: learn token maps on the DEV world, then measure blocking recall on the VAL world.
Direction: each S2/S3 record retrieves its top-K S1 candidates (each record belongs to at most one S1 entity),
country is a hard key.  Channels: name words, address words, name char-ngrams, and their sum ("combo").
Prints recall@K per channel, union recall vs candidate count, and breakdowns by source / script."""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from er_common import (norm_tokens, has_nonlatin, learn_token_map, apply_maps, make_word_tfidf,
                       make_char_tfidf, topk_sparse, save_json, load_df)

KMAX = 100
KS = [1, 5, 10, 20, 50, 100]


def recall_at(idx, truth, K):
    m = truth >= 0
    return (idx[m][:, :K] == truth[m][:, None]).any(1).mean() if m.any() else float("nan")


def union_stats(tops, chans, truth, K):
    cat = np.concatenate([tops[c][:, :K] for c in chans], axis=1)
    m = truth >= 0
    hit = (cat[m] == truth[m][:, None]).any(1)
    s = np.sort(cat, axis=1)
    uniq = (s[:, 0] >= 0).astype(int) + ((s[:, 1:] != s[:, :-1]) & (s[:, 1:] >= 0)).sum(1)
    return hit, uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--queries", type=int, default=20000, help="val records sampled per country")
    ap.add_argument("--qch", type=int, default=200, help="queries per sparse-product chunk (lower if RAM-bound)")
    ap.add_argument("--word-max-df", type=float, default=0.02)
    ap.add_argument("--char-max-df", type=float, default=0.01)
    ap.add_argument("--no-char", action="store_true", help="skip the (slow) char-ngram channel")
    ap.add_argument("--no-map", action="store_true", help="disable learned token maps (ablation)")
    ap.add_argument("--map-pairs", type=int, default=300000, help="max dev pairs per country for map learning")
    a = ap.parse_args()
    W = Path(a.work)
    rng = np.random.default_rng(0)
    t0 = time.time()
    T = lambda: f"[{time.time()-t0:5.0f}s]"

    dev_s1, dev_rec = load_df(W / "dev_s1"), load_df(W / "dev_rec")
    val_s1, val_rec = load_df(W / "val_s1"), load_df(W / "val_rec")
    print(T(), f"dev: S1 {len(dev_s1):,} rec {len(dev_rec):,} | val: S1 {len(val_s1):,} rec {len(val_rec):,}")

    # ---- learn maps on dev pairs
    maps = {}
    if not a.no_map:
        pairs = dev_rec[dev_rec.owner != ""].merge(
            dev_s1.rename(columns={"entity_id": "owner", "business_name": "s1n", "business_address": "s1a"})[
                ["owner", "s1n", "s1a"]], on="owner")
        for c, d in pairs.groupby("country"):
            d = d.sample(min(len(d), a.map_pairs), random_state=0)
            s1n, s1a = d.s1n.map(norm_tokens), d.s1a.map(norm_tokens)
            rn, ra = d.business_name.map(norm_tokens), d.business_address.map(norm_tokens)
            nm, nst = learn_token_map(s1n, rn)
            am, ast = learn_token_map(s1a, ra)
            maps[c] = {"name": nm, "addr": am}
            print(T(), f"[{c}] learned maps from {len(d):,} pairs: name {len(nm):,} entries, addr {len(am):,} entries")
            for lab, st in (("name", nst), ("addr", ast)):
                top = sorted(st.items(), key=lambda kv: -kv[1][2])[:25]
                print(f"   top {lab}: " + ", ".join(f"{k}->{v[3]}({v[2]})" for k, v in top))
            nl = [(k, v) for k, v in nst.items() if not k.isascii()]
            print(f"   name entries with non-ASCII source token: {len(nl):,}; examples: " +
                  ", ".join(f"{k}->{v[3]}({v[2]})" for k, v in sorted(nl, key=lambda kv: -kv[1][2])[:12]))
        save_json(maps, W / "token_maps.json")
        del pairs

    # ---- tokenise + canonicalise val
    for df in (val_s1, val_rec):
        df["n_t"] = df.business_name.map(norm_tokens)
        df["a_t"] = df.business_address.map(norm_tokens)
    val_s1["n_doc"], val_s1["a_doc"] = apply_maps(val_s1, {})
    val_rec["n_doc"], val_rec["a_doc"] = apply_maps(val_rec, maps)
    val_rec["nonlatin_name"] = val_rec.business_name.map(has_nonlatin)
    val_rec["addr_empty"] = val_rec.business_address.eq("")
    print(T(), "val tokenised")

    chans = ["name_w", "addr_w"] + ([] if a.no_char else ["name_c"]) + ["combo"]
    summary = []
    for c in sorted(val_s1.country.unique()):
        P = val_s1[val_s1.country == c].reset_index(drop=True)
        Q = val_rec[val_rec.country == c]
        Q = Q.sample(min(len(Q), a.queries), random_state=0).reset_index(drop=True)
        pid = {e: i for i, e in enumerate(P.entity_id)}
        truth = Q.owner.map(pid).fillna(-1).astype(int).values
        print(f"\n{T()} === {c}: pool S1={len(P):,} | queries={len(Q):,} (matched {(truth>=0).mean():.1%}, orphan {(truth<0).mean():.1%})")

        vw_n, vw_a = make_word_tfidf(a.word_max_df), make_word_tfidf(a.word_max_df)
        Xp_n = vw_n.fit_transform(P.n_doc); Xq_n = vw_n.transform(Q.n_doc)
        Xp_a = vw_a.fit_transform(P.a_doc); Xq_a = vw_a.transform(Q.a_doc)
        XpT_n, XpT_a = Xp_n.T.tocsr(), Xp_a.T.tocsr()
        if not a.no_char:
            vc = make_char_tfidf(a.char_max_df)
            Xp_c = vc.fit_transform(P.n_doc); Xq_c = vc.transform(Q.n_doc); XpT_c = Xp_c.T.tocsr()
        tops = {ch: np.full((len(Q), KMAX), -1, dtype=np.int32) for ch in chans}
        for s in range(0, len(Q), a.qch):
            sl = slice(s, s + a.qch)
            Sn, Sa = Xq_n[sl] @ XpT_n, Xq_a[sl] @ XpT_a
            parts = {"name_w": Sn, "addr_w": Sa}
            combo = Sn + Sa
            if not a.no_char:
                Sc = Xq_c[sl] @ XpT_c
                parts["name_c"] = Sc
                combo = combo + Sc
            parts["combo"] = combo
            for ch, S in parts.items():
                tops[ch][sl] = topk_sparse(S, KMAX)[0]
            if (s // a.qch) % 20 == 0:
                print(f"   {T()} {s:,}/{len(Q):,}", flush=True)

        rows = [{"K": K, **{ch: round(recall_at(tops[ch], truth, K), 4) for ch in chans}} for K in KS]
        print(f"pair-level recall@K per channel (true S1 owner within the record's top-K):")
        print(pd.DataFrame(rows).to_string(index=False))
        urows = []
        for K in (5, 10, 20, 50):
            hit, uniq = union_stats(tops, chans, truth, K)
            urows.append({"K_per_channel": K, "union_recall": round(hit.mean(), 4), "avg_candidates": round(uniq.mean(), 1),
                          "p95_candidates": int(np.percentile(uniq, 95))})
            summary.append({"country": c, **urows[-1]})
        print("UNION of all channels:"); print(pd.DataFrame(urows).to_string(index=False))

        Kb = 20
        hit, _ = union_stats(tops, chans, truth, Kb)
        Qm = Q[truth >= 0].reset_index(drop=True)
        Qm["hit"] = hit
        print(f"union@{Kb} recall breakdown (matched records only):")
        for col in ("src", "nonlatin_name", "addr_empty"):
            g = Qm.groupby(col).hit.agg(["mean", "size"]).round(4)
            print("  by", col, "|", "; ".join(f"{k}: {r['mean']} (n={int(r['size'])})" for k, r in g.iterrows()))
        miss = Qm[~Qm.hit].head(6)
        print("  sample MISSED records (name | address):")
        for _, r in miss.iterrows():
            o = P.iloc[pid[r.owner]]
            print(f"    REC {r.business_name[:60]} | {r.business_address[:70]}\n    S1  {o.business_name[:60]} | {o.business_address[:70]}")
    print(f"\n{T()} done.")


if __name__ == "__main__":
    main()