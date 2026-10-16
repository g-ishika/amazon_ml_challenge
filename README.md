# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [your team name]
**Team Members:** [your name(s)]
**Submission Date:** [today's date]

---

## 1. Executive Summary
We built a blocking-plus-classifier entity resolution pipeline: per-country
TF-IDF candidate retrieval feeding a LightGBM matcher trained on 41 pairwise
similarity features, with a tuned probability-margin decision rule. The
pipeline generalizes reasonably to an unseen test country (France) with no
country-specific hacks, validated by a leave-one-country-out check.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA showed matching is asymmetric and lossy: no S2/S3 record ever matches
more than one S1 entity (7.6M links, zero exceptions), country never differs
within a true pair, postal codes are largely unusable (present in <10% of
true pairs), and roughly a quarter of S2/S3 records are true orphans. India
names are frequently pure transliterations in Devanagari/Tamil/Telugu/etc.
(28% of S2, 18% of S3), which we found to be the largest single source of
missed matches.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (LightGBM), single-owner-per-record
decision rule.
**Core Innovation:** learned (not hand-coded) per-country abbreviation and
transliteration maps applied before blocking; a leave-one-country-out
validation protocol used to make threshold decisions with evidence rather
than intuition, specifically for the unseen France test country.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** TF-IDF over normalized name tokens, normalized
  address tokens, and their combination, unioned per-channel top-K
  (k_combo=10, k_name=3, k_addr=3 per record); country used as a hard
  partition (never blocked across countries).
- **Candidate pairs generated:** ~11-13 candidates/record on average
  (p95 ≈ 13-27 depending on country) across ~9.97M test S2/S3 records.
- **Recall protection:** measured recall ceiling directly (union@20 ≈ 0.979
  India, 0.993 US on held-out val); union of multiple cheap retrieval
  channels rather than one, so no single channel's blind spot is fatal.

---

## 4. Matching Model

**Features used:**
- Name features: Levenshtein-family ratios (ratio/token-sort/token-set/
  partial via rapidfuzz), Jaccard on tokens, TF-IDF cosine, length ratio,
  first/last-token equality, space-stripped variants (for domain-style names).
- Address features: same string-similarity family, token Jaccard, digit-only
  token Jaccard/overlap (postal/house-number proxy), first-numeric-token
  equality.
- Other: rank and score-gap within each record's candidate list, non-ASCII
  token share, empty-address flag, source (S2 vs S3) flag.

**Model type:** LightGBM (gradient-boosted trees), early stopping on a
held-out slice of the training pairs.
**Threshold selection method:** grid search over (tau, delta) maximizing
macro F0.5 on a held-out validation world (split by S1 entity, not by row,
to prevent leakage); cross-checked with a leave-one-country-out protocol to
avoid over-trusting in-distribution calibration for the unseen France split.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro), validation:** 0.9844 (tau=0.3, delta=0.5)
- **F0.5 Score (macro), leaderboard/test:** 0.893
- **Common false positives (wrong merges):** near-duplicate business names at
  different addresses (e.g. two branches/related entities sharing a name
  stem), and orphan records with an unusually strong lexical match to an
  unrelated S1 entity.
- **Common false negatives (missed matches):** non-Latin-script India names
  the learned maps hadn't seen; OCR-style digit-for-letter typos beyond what
  the tokenizer's leet-speak correction covers; off-by-one house numbers
  combined with an otherwise-clean name match.

---

## 6. Conclusion
The pipeline reaches near-0.98 F0.5 on in-distribution validation and 0.893
on the real test set, with the gap traced specifically to blocking recall on
India's non-Latin business names rather than to threshold miscalibration
(confirmed via a leave-one-country-out experiment showing stricter
thresholds do not recover cross-country F0.5 loss). The clearest next step
is a multilingual embedding retrieval channel targeted at that specific
subgroup.

---

## Appendix

### A. Code Artefacts
Full runnable code ships under `code/business_entity_resolution/`:
`src/` (er_common.py, er_blocking.py, er_features.py, stage0_split.py,
stage1_blocking_eval.py, stage2_train_eval.py, stage2b_country_holdout.py,
stage3_test_pipeline.py), `README.md` (exact reproduction commands), and
`requirements.txt` (pinned versions). Entry point: run stage0 through
stage3 in order as documented in README.md to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
Leave-one-country-out (stage2b_country_holdout.py): train India → test US,
F0.5 = 0.9649; train US → test India, F0.5 = 0.8781 — both peaking at the
same (tau=0.3, delta=0.3), evidence that generalization loss to an unseen
country is not fixable by threshold strictness alone.