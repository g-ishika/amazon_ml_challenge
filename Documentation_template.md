# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Unsupervised
**Team Members:** Ishika Gupta, Avani Maheshwari, Nidhi Kumari, Deepmala Deepak
**Submission Date:** September 27, 2026

---

## 1. Executive Summary
We built a blocking-plus-classifier entity resolution pipeline: per-country TF-IDF candidate retrieval feeding a LightGBM matcher trained on 41 pairwise similarity features, with a tuned probability-margin decision rule. The pipeline generalizes to an unseen test country (France) with no country-specific hacks, a decision validated by a leave-one-country-out experiment rather than assumption.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA showed the matching problem is highly structured: no S2/S3 record ever matches more than one S1 entity (confirmed across 7.6M links, zero exceptions), country never differs within a true pair, and postal codes are largely unusable (present in under 10% of true pairs). Roughly a quarter of S2/S3 records are true orphans with no match. The largest source of noise is India business names given as pure transliterations in Devanagari, Tamil, Telugu, and other scripts (28% of S2, 18% of S3 names) — this was later confirmed via blocking recall measurement to be the single biggest driver of missed matches.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (LightGBM), with a single-owner-per-record decision rule exploiting the one-S1-per-record structural constraint found during EDA.
**Core Innovation:** learned (not hand-coded) per-country abbreviation and transliteration maps applied before blocking, plus a leave-one-country-out validation protocol used to make threshold decisions for the unseen France test country based on evidence rather than intuition.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** TF-IDF cosine similarity over normalized name tokens, normalized address tokens, and their sum, unioned across per-channel top-K retrieval (k_combo=10, k_name=3, k_addr=3 per record); country used as a hard partition key.
- **Candidate pairs generated:** ~11-13 candidates per record on average (p95 ≈ 13-27 depending on country) across ~9.97M test S2/S3 records.
- **How you ensured true matches were not lost:** measured recall ceiling directly against ground truth on a held-out validation split (union@20 recall ≈ 0.979 for India, 0.993 for US); combined several cheap retrieval channels rather than relying on one, so a single channel's blind spot could still be caught by another.

---

## 4. Matching Model

**Features used:**
- Name features: Levenshtein-family ratios (ratio, token-sort, token-set, partial via rapidfuzz), token Jaccard, TF-IDF cosine, length ratio, first/last-token equality, space-stripped variants (for domain-style names like "philipatribblecustom.com").
- Address features: the same string-similarity family, token Jaccard, digit-only token Jaccard/overlap (a postal-code/house-number proxy given real postal codes were mostly absent), first-numeric-token equality.
- Other: rank and score-gap of each candidate within its record's retrieved list, non-ASCII token share (a transliteration flag), empty-address flag, source (S2 vs S3) indicator.

**Model type:** LightGBM (gradient-boosted decision trees) with early stopping on a held-out slice of training pairs.
**Threshold selection method:** grid search over (tau, delta) — minimum accept probability and minimum margin over the runner-up candidate — maximizing macro F0.5 on a validation world split by S1 entity (not by row, to prevent leakage). Cross-checked with a leave-one-country-out protocol (train on one country, test on the other) to avoid over-trusting in-distribution calibration when applying it to the unseen France split.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** 0.9844 (tau=0.3, delta=0.5)
- **F_0.5 Score (macro), leaderboard/test:** 0.893
- **Common false positives (wrong merges):** near-duplicate business names at different addresses (e.g., related entities sharing a name stem, such as "Ikshana Homes" vs "Ikshana Agro"), and orphan records with an unusually strong lexical match to an unrelated S1 entity.
- **Common false negatives (missed matches):** non-Latin-script India names the learned maps hadn't seen; OCR-style digit-for-letter typos beyond what the tokenizer's correction covers (e.g., "CONTlNENTAL" with a lowercase L); off-by-one house numbers combined with an otherwise clean name match.

---

## 6. Conclusion
The pipeline reaches near-0.98 F0.5 on in-distribution validation and 0.893 on the real test set, with the gap traced specifically to blocking recall on India's non-Latin business names rather than to threshold miscalibration — confirmed via a leave-one-country-out experiment showing that stricter thresholds do not recover cross-country F0.5 loss. The clearest next step, scoped but not completed given time constraints, is a multilingual embedding retrieval channel targeted at that specific subgroup.

---

## Appendix

### A. Code Artefacts
Full runnable code ships under `code/business_entity_resolution/`: `src/` (er_common.py, er_blocking.py, er_features.py, stage0_split.py, stage1_blocking_eval.py, stage2_train_eval.py, stage2b_country_holdout.py, stage3_test_pipeline.py), `README.md` (exact reproduction commands), and `requirements.txt` (pinned versions). Entry point: run stage0 through stage3 in order as documented in README.md to reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
Leave-one-country-out check (stage2b_country_holdout.py): train India → test US, F0.5 = 0.9649; train US → test India, F0.5 = 0.8781 — both peaking at the same threshold setting (tau=0.3, delta=0.3), evidence that generalization loss to an unseen country is not fixable by threshold strictness alone.

---