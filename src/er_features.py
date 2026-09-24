"""Pair features for the matcher. Input: pool index, query matrices, docs, candidate pairs (qi, pj, ranks)."""
import re
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    f_ratio, f_tsort, f_tset, f_part = fuzz.ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio, fuzz.partial_ratio
    HAVE_RF = True
except ImportError:                                   # slow fallback so the code still runs; install rapidfuzz!
    import difflib
    HAVE_RF = False
    def f_ratio(a, b): return 100 * difflib.SequenceMatcher(None, a, b).ratio()
    def f_tsort(a, b): return f_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))
    def f_tset(a, b):
        A, B = set(a.split()), set(b.split())
        return 100 * len(A & B) / max(1, min(len(A), len(B))) if A and B else 0.0
    def f_part(a, b):
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        return 100 * difflib.SequenceMatcher(None, s, l).find_longest_match(0, len(s), 0, len(l)).size / max(1, len(s))

DOMAIN_RE = re.compile(r"\.(com|net|org|in|co|io|fr|us|info|biz)\b", re.I)


def rowwise_dot(A, B, ri, ci, chunk=400_000):
    out = np.empty(len(ri), dtype=np.float32)
    for s in range(0, len(ri), chunk):
        e = s + chunk
        out[s:e] = np.asarray(A[ri[s:e]].multiply(B[ci[s:e]]).sum(1)).ravel()
    return out


def _f2(fn, a, b, empty_nan=True):
    if empty_nan:
        return np.fromiter((fn(x, y) if (x and y) else np.nan for x, y in zip(a, b)), dtype=np.float32, count=len(a))
    return np.fromiter((fn(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


def _sets(docs):
    return [frozenset(d.split()) for d in docs]


def _jac(sa, sb):
    return np.fromiter((len(x & y) / len(x | y) if (x or y) else np.nan for x, y in zip(sa, sb)), dtype=np.float32, count=len(sa))


def _shared(sa, sb):
    return np.fromiter((len(x & y) for x, y in zip(sa, sb)), dtype=np.float32, count=len(sa))


def record_side_info(raw_names, raw_addrs, n_docs, a_docs, src):
    """Per-record arrays used by the pair features."""
    return {
        "domain": np.array([1.0 if DOMAIN_RE.search(x) else 0.0 for x in raw_names], dtype=np.float32),
        "addr_empty": np.array([1.0 if x == "" else 0.0 for x in raw_addrs], dtype=np.float32),
        "nonascii_n": np.array([np.mean([not t.isascii() for t in d.split()]) if d else 0.0 for d in n_docs], dtype=np.float32),
        "nonascii_a": np.array([np.mean([not t.isascii() for t in d.split()]) if d else 0.0 for d in a_docs], dtype=np.float32),
        "src": np.array([1.0 if s == "S3" else 0.0 for s in src], dtype=np.float32),
    }


def pair_features(idx, qmats, Qn, Qa, Qinfo, Pn, Pa, qi, pj, R):
    """Qn/Qa/Pn/Pa: numpy object arrays of canonical docs.  Returns DataFrame (float32) of features, one row per pair."""
    F = {}
    F["cos_n"] = rowwise_dot(qmats["n"], idx.Xn, qi, pj)
    F["cos_a"] = rowwise_dot(qmats["a"], idx.Xa, qi, pj)
    F["cos_c"] = rowwise_dot(qmats["c"], idx.Xc, qi, pj) if idx.vc is not None else np.zeros(len(qi), np.float32)
    F["combo"] = F["cos_n"] + F["cos_a"] + F["cos_c"]
    for c, name in enumerate(["rank_combo", "rank_name", "rank_addr", "rank_char"]):
        F[name] = R[:, c].astype(np.float32)
    g = pd.DataFrame({"qi": qi, "combo": F["combo"]})
    gm = g.groupby("qi").combo
    F["combo_gap"] = (gm.transform("max") - g.combo).values.astype(np.float32)
    F["combo_rank_u"] = gm.rank(ascending=False, method="min").values.astype(np.float32)
    F["n_cands"] = gm.transform("size").values.astype(np.float32)

    qn, qa, pn, pa = Qn[qi], Qa[qi], Pn[pj], Pa[pj]
    F["n_ratio"], F["n_tsort"] = _f2(f_ratio, qn, pn), _f2(f_tsort, qn, pn)
    F["n_tset"], F["n_part"] = _f2(f_tset, qn, pn), _f2(f_part, qn, pn)
    qns = [x.replace(" ", "") for x in qn]; pns = [x.replace(" ", "") for x in pn]      # handles "philipatribblecustom.com"
    F["n_ns_ratio"], F["n_ns_part"] = _f2(f_ratio, qns, pns), _f2(f_part, qns, pns)
    F["a_ratio"], F["a_tsort"] = _f2(f_ratio, qa, pa), _f2(f_tsort, qa, pa)
    F["a_tset"], F["a_part"] = _f2(f_tset, qa, pa), _f2(f_part, qa, pa)

    sqn, spn, sqa, spa = _sets(qn), _sets(pn), _sets(qa), _sets(pa)
    F["n_jac"], F["a_jac"] = _jac(sqn, spn), _jac(sqa, spa)
    F["n_shared"], F["a_shared"] = _shared(sqn, spn), _shared(sqa, spa)
    nq = [frozenset(t for t in s if t.isdigit()) for s in sqa]
    np_ = [frozenset(t for t in s if t.isdigit()) for s in spa]
    F["num_jac"], F["num_shared"] = _jac(nq, np_), _shared(nq, np_)
    F["num_both"] = np.fromiter((1.0 if (x and y) else 0.0 for x, y in zip(nq, np_)), dtype=np.float32, count=len(qi))
    fq = [next((t for t in d.split() if t.isdigit()), "") for d in qa]
    fp = [next((t for t in d.split() if t.isdigit()), "") for d in pa]
    F["num_first_eq"] = np.fromiter((1.0 if (x and x == y) else (0.0 if (x and y) else np.nan) for x, y in zip(fq, fp)),
                                    dtype=np.float32, count=len(qi))
    ln_q = np.fromiter((len(x) for x in qn), np.float32, len(qi)); ln_p = np.fromiter((len(x) for x in pn), np.float32, len(qi))
    F["len_ratio_n"] = np.minimum(ln_q, ln_p) / np.maximum(np.maximum(ln_q, ln_p), 1)
    F["q_ntok"] = np.fromiter((len(s) for s in sqn), np.float32, len(qi)); F["p_ntok"] = np.fromiter((len(s) for s in spn), np.float32, len(qi))
    F["q_atok"] = np.fromiter((len(s) for s in sqa), np.float32, len(qi)); F["p_atok"] = np.fromiter((len(s) for s in spa), np.float32, len(qi))
    for k in ("domain", "addr_empty", "nonascii_n", "nonascii_a", "src"):
        F["q_" + k] = Qinfo[k][qi]
    F["first_tok_eq"] = np.fromiter((1.0 if (x and y and x.split()[0] == y.split()[0]) else 0.0 for x, y in zip(qn, pn)), np.float32, len(qi))
    F["last_tok_eq"] = np.fromiter((1.0 if (x and y and x.split()[-1] == y.split()[-1]) else 0.0 for x, y in zip(qn, pn)), np.float32, len(qi))
    return pd.DataFrame(F)