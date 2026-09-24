"""Shared helpers for the ER pipeline: normalisation, learned token maps, sparse retrieval."""
import json, re, unicodedata
from collections import Counter
import numpy as np

# \w covers Latin + Indic letters; Indic vowel signs (marks) are NOT \w, so add the Indic blocks explicitly
TOKEN_RE = re.compile(r"[\w\u0900-\u0DFF\u200c\u200d]+")
COMB_RE = re.compile(r"[\u0300-\u036f]+")            # Latin combining accents only (keeps Indic marks)
NONLATIN_RE = re.compile(r"[\u0370-\u1fff\u3000-\uffff]")


def norm_tokens(s):
    """lowercase, '&'->'and', strip Latin accents (é->e), split on punctuation. Language-agnostic."""
    s = s.lower().replace("&", " and ").replace("_", " ")
    if not s.isascii():
        s = COMB_RE.sub("", unicodedata.normalize("NFKD", s))
    return TOKEN_RE.findall(s)


def has_nonlatin(s):
    return bool(NONLATIN_RE.search(s))


def _prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def learn_token_map(s1_toks, rec_toks, min_count=3, min_cond=0.5, min_dice=0.25, max_combos=40):
    """Learn rec-token -> S1-token canonicalisation from labelled (S1, record) pairs.
    Looks at tokens present only in the record (A) vs only in S1 (B) and keeps (a,b) that co-occur consistently.
    Learns e.g.  texas->tx, rd->road, pvt->private, Devanagari word -> its Latin S1 spelling.
    Uses ONLY the provided training pairs (no external resources).
    Returns (map, stats) where stats[a] = (dice, prefix_len, support, b)."""
    cnt_a, cnt_b, diffs = Counter(), Counter(), []
    for T1, TR in zip(s1_toks, rec_toks):
        S, R = set(T1), set(TR)
        A, B = R - S, S - R
        if A and B and len(A) * len(B) <= max_combos:
            cnt_a.update(A); cnt_b.update(B); diffs.append((A, B))
    good_a = {t for t, c in cnt_a.items() if c >= min_count}
    good_b = {t for t, c in cnt_b.items() if c >= min_count}
    co = Counter()
    for A, B in diffs:
        A2, B2 = A & good_a, B & good_b
        for a in A2:
            for b in B2:
                co[(a, b)] += 1
    cand = {}
    for (a, b), c in co.items():
        if c < min_count:
            continue
        if c / cnt_a[a] < min_cond:
            continue
        dice = 2 * c / (cnt_a[a] + cnt_b[b])
        if dice < min_dice:
            continue
        cand.setdefault(a, []).append((round(dice, 2), _prefix_len(a, b), c, b))
    stats = {a: max(v) for a, v in cand.items()}
    return {a: v[3] for a, v in stats.items()}, stats


def apply_maps(df, maps, ncol="n_t", acol="a_t"):
    """Return (name_doc, addr_doc) space-joined canonical token strings; countries without a map pass through unchanged."""
    out_n, out_a = [], []
    for nt, at, c in zip(df[ncol], df[acol], df["country"]):
        m = maps.get(c)
        if m:
            nm, am = m["name"], m["addr"]
            nt = [nm.get(t, t) for t in nt]
            at = [am.get(t, t) for t in at]
        out_n.append(" ".join(nt)); out_a.append(" ".join(at))
    return out_n, out_a


def make_word_tfidf(max_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer=str.split, lowercase=False, max_df=max_df, sublinear_tf=True, dtype=np.float32)


def make_char_tfidf(max_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), lowercase=False, max_df=max_df,
                           sublinear_tf=True, dtype=np.float32)


def topk_sparse(S, K):
    """Row-wise top-K of a sparse matrix -> (idx[n,K] padded with -1, score[n,K])."""
    S = S.tocsr()
    n = S.shape[0]
    idx = np.full((n, K), -1, dtype=np.int32)
    sc = np.zeros((n, K), dtype=np.float32)
    ip, ind, dat = S.indptr, S.indices, S.data
    for i in range(n):
        a, b = ip[i], ip[i + 1]
        if a == b:
            continue
        d = dat[a:b]
        sel = np.argpartition(-d, K - 1)[:K] if b - a > K else np.arange(b - a)
        order = sel[np.argsort(-d[sel])]
        idx[i, :len(order)] = ind[a:b][order]
        sc[i, :len(order)] = d[order]
    return idx, sc


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_df(df, path):
    """Parquet if pyarrow is installed, else pickle (same stem, different extension)."""
    from pathlib import Path
    p = Path(path)
    try:
        import pyarrow  # noqa
        df.to_parquet(p.with_suffix(".parquet"), index=False)
    except ImportError:
        df.to_pickle(p.with_suffix(".pkl"))


def load_df(path):
    import pandas as pd
    from pathlib import Path
    p = Path(path)
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    return pd.read_pickle(p.with_suffix(".pkl"))


# ---------------------------------------------------------------- name-specific normalisation
_LEET = str.maketrans("0158", "olsb")      # OCR-style digit-for-letter typos seen in the data: fami1y, r0cky, 8ay, 5ecure
_ALPHA = re.compile(r"[a-z]")
_DIGIT = re.compile(r"\d")
_ORD = re.compile(r"^\d+(st|nd|rd|th)$")


def fix_leet(toks):
    """Only touches tokens that mix letters and digits (not ordinals like 4th)."""
    return [t.translate(_LEET) if (_DIGIT.search(t) and _ALPHA.search(t) and not _ORD.match(t)) else t for t in toks]


def norm_name_tokens(s):
    return fix_leet(norm_tokens(s))


def learn_maps(s1_df, rec_df, max_pairs=300000, verbose=True):
    """Learn per-country name/address token maps from labelled dev pairs.
    s1_df needs n_t,a_t ; rec_df needs n_t,a_t,country,owner.  Returns {country: {'name':..., 'addr':...}}."""
    pairs = rec_df[rec_df.owner != ""].merge(
        s1_df[["entity_id", "n_t", "a_t"]].rename(columns={"entity_id": "owner", "n_t": "s1n", "a_t": "s1a"}), on="owner")
    maps = {}
    for c, d in pairs.groupby("country"):
        d = d.sample(min(len(d), max_pairs), random_state=0)
        nm, _ = learn_token_map(d.s1n, d.n_t)
        am, _ = learn_token_map(d.s1a, d.a_t)
        maps[c] = {"name": nm, "addr": am}
        if verbose:
            print(f"   [{c}] maps from {len(d):,} pairs: name {len(nm):,}, addr {len(am):,} entries")
    return maps