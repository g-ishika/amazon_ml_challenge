"""Candidate generation: per-country TF-IDF index over the S1 pool; each S2/S3 record retrieves candidates from
several channels (combo, name words, address words, optional name char-ngrams); union + per-channel ranks are returned."""
from dataclasses import dataclass
import numpy as np
from er_common import make_word_tfidf, make_char_tfidf, topk_sparse

CH = ["combo", "name", "addr", "char"]     # column order of the rank matrix


@dataclass
class BlockCfg:
    k_combo: int = 10
    k_name: int = 3
    k_addr: int = 3
    k_char: int = 5
    use_char: bool = False
    word_max_df: float = 0.02
    char_max_df: float = 0.01
    qch: int = 200


class PoolIndex:
    """TF-IDF index of one country's S1 pool (n_doc = canonical name string, a_doc = canonical address string)."""

    def __init__(self, n_docs, a_docs, cfg: BlockCfg):
        self.cfg = cfg
        self.n = len(n_docs)
        self.vn, self.va = make_word_tfidf(cfg.word_max_df), make_word_tfidf(cfg.word_max_df)
        self.Xn, self.Xa = self.vn.fit_transform(n_docs), self.va.fit_transform(a_docs)
        self.XnT, self.XaT = self.Xn.T.tocsr(), self.Xa.T.tocsr()
        self.vc = None
        if cfg.use_char:
            self.vc = make_char_tfidf(cfg.char_max_df)
            self.Xc = self.vc.fit_transform(n_docs)
            self.XcT = self.Xc.T.tocsr()

    def transform_queries(self, n_docs, a_docs):
        q = {"n": self.vn.transform(n_docs), "a": self.va.transform(a_docs)}
        if self.vc is not None:
            q["c"] = self.vc.transform(n_docs)
        return q

    def retrieve(self, q, progress=None):
        """Returns qi (int32), pj (int32), R (n_pairs x 4 int16 ranks per channel, 99 = not retrieved by that channel)."""
        cfg, nq = self.cfg, q["n"].shape[0]
        Ks = {"combo": cfg.k_combo, "name": cfg.k_name, "addr": cfg.k_addr, "char": cfg.k_char if self.vc is not None else 0}
        QI, PJ, RR = [], [], []
        for s in range(0, nq, cfg.qch):
            e = min(nq, s + cfg.qch); m = e - s
            Sn, Sa = q["n"][s:e] @ self.XnT, q["a"][s:e] @ self.XaT
            combo, Sc = Sn + Sa, None
            if self.vc is not None:
                Sc = q["c"][s:e] @ self.XcT
                combo = combo + Sc
            src = {"combo": combo, "name": Sn, "addr": Sa, "char": Sc}
            keys, chan, rank = [], [], []
            for ci, ch in enumerate(CH):
                K = Ks[ch]
                if K == 0 or src[ch] is None:
                    continue
                idx = topk_sparse(src[ch], K)[0]
                ok = idx >= 0
                qq = np.repeat(np.arange(m, dtype=np.int64), K).reshape(m, K)[ok]
                keys.append(qq * self.n + idx[ok])
                chan.append(np.full(len(qq), ci, dtype=np.int8))
                rank.append(np.tile(np.arange(K, dtype=np.int16), (m, 1))[ok])
            if not keys:
                continue
            keys, chan, rank = np.concatenate(keys), np.concatenate(chan), np.concatenate(rank)
            uk, inv = np.unique(keys, return_inverse=True)
            R = np.full((len(uk), 4), 99, dtype=np.int16)
            R[inv, chan] = rank
            QI.append((uk // self.n + s).astype(np.int32)); PJ.append((uk % self.n).astype(np.int32)); RR.append(R)
            if progress and (s // cfg.qch) % 100 == 0:
                progress(s, nq)
        if not QI:
            return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 4), np.int16)
        return np.concatenate(QI), np.concatenate(PJ), np.concatenate(RR)