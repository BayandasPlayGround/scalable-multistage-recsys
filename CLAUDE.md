# CLAUDE.md — Amazon RecSys: BLAIR Retrieval Upgrade

This file is loaded automatically by Claude Code on every session. Its purpose is to give a future
Claude (or a human reader resuming this work) full context on the in-progress neural-retrieval
upgrade without having to re-read the entire chat history.

## TL;DR

We are upgrading the multi-stage recommender's neural retrieval path from a from-scratch
dual-tower model ("DAT-Lite") to a **frozen pretrained text encoder** ("BLAIR", from the dataset's
own paper). The change is staged through six small, opt-in phases plus a final prod-scale run.
At medium debug scale (273 eval users, single category) BLAIR-rich produced a **+50 % test
recall@10 lift** versus the existing TF-IDF baseline. The remaining work is validating the same
direction at full prod scale (1.24 M items, ~14 M positives, three categories).

## Why we are doing this

### The failure that started this work

The prior bundle `prod-2026-05-10-dat-v2` was an attempt to fix the recall problem in the
`prod-2026-05-10-recovery-v1` bundle. DAT-Lite (a partial implementation of the DAT paper) was
introduced as a "bleeding-edge" neural retrieval variant. It failed every numeric gate of its
own plan:

| Gate | Target | DAT-v2 observed |
|---|---|---|
| `candidate_union` known-user recall (test) | ≥ 0.15 | 0.108 |
| `ranker_candidates` known-user recall (test) | ≥ 0.12 | 0.064 |
| `anonymous_no_history` candidate_union | > 0.052 | 0.052 (flat) |
| Final test recall@100 | > 0.052 | 0.052 (flat) |

Source: the bundle's [evaluation_summary.json](artifacts/amazon_recsys/bundles/prod-2026-05-10-dat-v2/evaluation_summary.json).

### Three root causes diagnosed from the code

1. **Serving-time query distribution mismatch.** DAT-Lite was trained with a learned user
   tower, but at serving time the user encoder is discarded
   ([bundles.py:81](src/amazon_recsys/ml/bundles.py#L81)) and the query becomes
   `mean(item_embeddings[history])` ([core.py:2699](src/amazon_recsys/ml/core.py#L2699)). The
   ANN index returns near-zero true positives over the full catalog even though the in-batch
   discrimination sanity score was positive. This is the dominant failure mode.

2. **Asymmetric Adaptive-Mimic Mechanism.** The DAT paper (Yu et al., 2021) specifies augmented
   vectors on **both** towers. The implementation at
   [core.py:2104](src/amazon_recsys/ml/core.py#L2104) only adds them on the item side, halving
   the paper's intended inter-tower interaction.

3. **Silent-drop sanity gate.** `_retriever_recovers_positives`
   ([core.py:3092](src/amazon_recsys/ml/core.py#L3092)) drops a broken neural retriever from
   `session.retrievers`, which makes the bundle's `manifest.json` `retriever_variants` end up
   `["content_based", "latent_cf"]` even when the user asked for a neural bundle. The failure is
   silent.

### The bet we are placing

The dataset is **Amazon Reviews 2023** (Hou et al., 2024). The same paper proposes **BLAIR**, a
RoBERTa-base sentence encoder contrastively pretrained on Amazon item text + reviews. Two
properties make it a better fit than DAT-Lite for this codebase:

- It is **frozen** — no training distribution mismatch is possible because we precompute item
  embeddings once and serve them via the same `mean(history_embeddings)` aggregator that the
  existing `two_tower`/`dat_lite`/`content_based` paths already use. The mismatch that broke
  DAT-Lite is structurally impossible.
- It is **purpose-built for this dataset**. The paper reports significant lifts on Amazon
  Reviews 2023 retrieval tasks vs. from-scratch dual-tower baselines, particularly on cold-start
  and long-tail categories — which are the failure modes in the DAT-v2 acceptance run.

We deliberately did **not** implement the full DAT paper (symmetric AMM + Category Alignment
Loss) because the dataset paper's own recommended encoder is a faster path to comparable or
better quality with much less moving machinery.

## What has been built

The implementation is split into six small phases. Each phase is opt-in via env vars, leaves
prior behaviour bit-for-bit identical when not opted in, and was validated against a
single-category debug run before the next phase began. The discipline is documented at the end
of this file under "Process notes".

### Phase 1 — BLAIR opt-in path (validated)

A new variant `"blair_text"` was added to `VALID_NEURAL_RETRIEVER_VARIANTS`. When
`AMAZON_RECSYS_NEURAL_RETRIEVER_VARIANT=blair_text` (and neural retriever enabled),
[train_retrievers](src/amazon_recsys/ml/core.py) dispatches to a new
[train_blair_retriever](src/amazon_recsys/ml/retrievers/blair.py) function that encodes the
item catalog once with `sentence-transformers` loading `hyp1231/blair-roberta-base` (BLAIR
proper) with `sentence-transformers/all-MiniLM-L6-v2` as a fallback. The resulting embeddings
are stored under the existing `"two_tower"` source alias so the ranker's feature schema
(`from_two_tower`, `score_two_tower`, `rank_two_tower`) is unchanged.

Phase 1 acceptance test: 8-user debug run on All_Beauty. BLAIR fed 4/8 positives into the
candidate union (parity with content_based and latent_cf at debug scale). Bundle
`retriever_variants` contains `"two_tower"`. ✅

### Phase 2 — Rich item text reload (validated)

Phase 1 fed BLAIR `title + source_category` only because `keep_columns` in
[_build_item_features](src/amazon_recsys/ml/core.py#L1125) drops `description_text`,
`features_text`, `store`, and `categories_text` before persisting `item_features.parquet`.
Phase 2 added an in-memory metadata reload inside `train_blair_retriever` (no parquet schema
change, no persisted state change). When `core.load_metadata` returns successfully, BLAIR sees:

```
meta_title + store + categories_text + description_text + features_text
```

When metadata is unavailable, the per-row text degrades to the Phase 1 fallback. The retriever
records `item_text_columns` and `item_text_source_counts` (`{"rich": N, "fallback": M}`) in its
metadata.

Phase 2 acceptance test: 273-user single-category debug run. **BLAIR-rich beat content_based by
+50% on test recall@10 (0.060 vs 0.040) and +29% on test recall@100 (0.110 vs 0.085).** By-source
contribution at the union: BLAIR 24/273, content_based 20/273 — BLAIR is now the largest
contributor. ✅

### Phase C — Bundle audit metadata persistence (validated)

Previously, `_write_retriever_artifacts` wrote `"model": {"retriever": variant}` — discarding
any `encoder_name` or audit fields the BLAIR retriever attached in memory. Phase C added:

- `_AUDIT_METADATA_KEYS`: a JSON-safe whitelist of fields (encoder_name,
  configured_model_name, fallback_model_name, embedding_dim, batch_size, max_seq_length,
  serving_query, item_text_columns, item_text_source_counts, source_alias)
- `_is_json_safe_scalar`: recursive primitive filter — defends against sklearn objects on
  `latent_cf.model` that would otherwise break JSON serialisation
- A new `"audit_metadata"` block in each retriever's `metadata.json`

Round-trip tested in [tests/test_bundle_audit_metadata.py](tests/test_bundle_audit_metadata.py).
Legacy bundles without `audit_metadata` still load (the field defaults to absent without raising).

Phase C acceptance: the `beauty-medium-hardneg` bundle's
[retrievers/two_tower/metadata.json](artifacts/amazon_recsys/bundles/beauty-medium-hardneg-20260511T095614Z/retrievers/two_tower/metadata.json)
shows the full populated block. ✅

### Phase D — Ranker hardneg mix (opt-in, did not help at debug scale)

[_rebalance_ranker_candidates](src/amazon_recsys/ml/core.py) now accepts a `hardneg_mix` triple
of weights for `(popularity, cooccurrence, random)` sources. The default `"0,0,1.0"` is
bit-for-bit identical to the pre-D uniform sampler (verified by
[test_default_mix_matches_legacy_pure_random_behaviour](tests/test_ranker_hardneg_mix.py)).
Opting in via `AMAZON_RECSYS_RANKER_HARDNEG_MIX="0.6,0.3,0.1"` biases negatives toward
popularity-weighted and cooccurrence-hard candidates.

Phase D acceptance test: same 273-user setup as Phase 2A, mix toggled on. Result: **val recall@10
dropped from 0.062 to 0.048; test recall@10 dropped from 0.040 to 0.037**. Hypothesised cause:
the ranker is data-bound at this scale (1,615 training examples; training-validation ndcg
ceilings at ~0.95 regardless of negative distribution), not negative-distribution-bound. The
mix may pay off at prod scale (50K+ training examples), but at debug scale it has no signal to
work with. The code remains opt-in and reverts to legacy behaviour when not enabled. ⚠️

### Phase F — Acceptance gate validator (validated)

[validate_acceptance_gates](src/amazon_recsys/ml/bundles.py) reads CSVs from `eval_dir` after
Stage 5 (evaluation summary) and compares the configured profile's floors. Failure raises
`GateValidationError` before bundle export — so a regression cannot be silently activated.

Three profiles:

- **`off`** (default): no validation. Identical to pre-F behaviour.
- **`recovery-v1`**: single floor at `final test recall@100 >= 0.052` (the current prod
  baseline). Catches catastrophic regression but not marginal-but-real improvements.
- **`blair-v1`**: seven floors targeted at proving BLAIR beats the current prod by ≥35%:
  blair_text retriever recall@100 ≥ 0.06, candidate_union recall ≥ 0.16, ranker_candidates
  recall ≥ 0.13, anonymous_no_history candidate_union > 0.052, Industrial_and_Scientific
  candidate_union > 0.06, final recall@100 ≥ 0.07, final recall@10 ≥ 0.028.

The gates are calibrated for **prod-scale** runs. They will fail at debug scale by design —
use `off` or `recovery-v1` for debug runs.

Phase F acceptance: synthesised CSVs of the failed DAT-v2 numbers correctly trip `blair-v1`
gates (proving the gate would have caught the regression that triggered this whole work);
the same numbers pass `recovery-v1`. ✅

### Phase G — TF-IDF memory robustness (just applied during the prod-scale run)

[_fit_text_features](src/amazon_recsys/ml/core.py#L1097) raised `MemoryError` inside sklearn's
`_count_vocab` at 1.24 M items because `ngram_range=(1, 2)` roughly doubles the per-doc index
list that sklearn extends linearly. Three-line edit:

1. `ngram_range=(1, 2)` → `ngram_range=(1, 1)` — unigrams only.
2. `min_df=2` → `min_df=5` — drops ultra-rare terms (in fewer than 5 docs) from the vocabulary.
3. `gc.collect()` before the fit + `del texts; gc.collect()` in a `finally` block — frees the
   1.24 M-element Python list as soon as the sparse matrix is built.

No new config knobs, no new tests beyond the existing TF-IDF coverage. Trade-off: `content_based`
retriever loses bigram signal, but `content_based` was already weaker than BLAIR (+50 % gap at
medium scale), and BLAIR is the primary neural retriever now. ✅ (committed before the in-progress
prod run hit the same step).

### Phase E — Prod-scale run (in progress at time of writing)

The current run is `prod-2026-05-11-blair-v1` with `--run-profile quality-neural`. Env config:

```
AMAZON_RECSYS_CATEGORIES='["All_Beauty","Automotive","Industrial_and_Scientific"]'
AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER=true
AMAZON_RECSYS_NEURAL_RETRIEVER_VARIANT=blair_text
AMAZON_RECSYS_GATE_PROFILE=recovery-v1
```

Conservative gate profile (recovery-v1) because this is the first prod-scale BLAIR run; the
goal is "beats current prod baseline of 0.052 final test recall@100" rather than the more
ambitious blair-v1 floors. If recovery-v1 passes, a follow-up run can tighten to blair-v1.

## Architecture: how the retrieval stack now works

```
                ┌──────────────────────────────────────────────────┐
                │  Stage 1: prepare_corpus                         │
                │  - extracts review signals (cached per run-name) │
                │  - computes k-core (cached per run-name)         │
                │  - fits TfidfVectorizer + TruncatedSVD           │
                │  - builds item_features.parquet                  │
                └──────────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────────┐
                │  Stage 2: make_splits                            │
                │  - chronological train/val/test                  │
                │  - cooccurrence dict, popularity counters        │
                └──────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Stage 3: train_retrievers                                      │
   │                                                                 │
   │   content_based  ──  TF-IDF/SVD over item_text  (always)        │
   │   latent_cf      ──  Truncated SVD over user×item  (always)     │
   │                                                                 │
   │   neural slot (key="two_tower" in dict) — only when enabled:    │
   │     ┌─ two_tower    : TF dual-encoder, trained from scratch     │
   │     ├─ dat_lite     : two_tower + asymmetric AMM + CAL (legacy) │
   │     └─ blair_text   : frozen pretrained sentence-transformer    │
   │                        ↑ NEW PRIMARY PATH                       │
   │       text source: meta_title + store + categories_text         │
   │                  + description_text + features_text             │
   │                  (reloaded in-memory from raw metadata files)   │
   │       encoder: hyp1231/blair-roberta-base                       │
   │                (fallback: all-MiniLM-L6-v2)                     │
   │       serving query: mean(item_embeddings[history])             │
   │       ANN: Annoy, angular metric                                │
   └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Stage 4: train_ranker (XGBoost lambdarank)                     │
   │  - candidate union from all retrievers (source-balanced quotas) │
   │  - features: source flags, retrieval scores, ranks,             │
   │              embedding similarity (selected_retriever_variant)  │
   │  - NEGATIVE SAMPLING: configurable (Phase D, opt-in)            │
   │      AMAZON_RECSYS_RANKER_HARDNEG_MIX="popularity,cooc,random"  │
   └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Stage 5: evaluate (writes CSVs to eval_dir)                    │
   │  Stage 5b: validate_acceptance_gates                            │
   │           (Phase F — raises before export if gates fail)        │
   │  Stage 6: MLflow logging                                        │
   └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Bundle export (only if gates pass)                             │
   │  - retrievers/two_tower/item_embeddings.npy + item_index.ann    │
   │  - retrievers/two_tower/metadata.json (NEW: audit_metadata)     │
   │  - models/ranker.onnx                                           │
   │  - manifest.json (lists retriever_variants)                     │
   └─────────────────────────────────────────────────────────────────┘
```

### Why BLAIR works where DAT-Lite did not

The single most important architectural property is that **BLAIR's serving query path is
identical to its training-time view of items**. Both training and serving see normalised text
embeddings of items. The query at retrieval time is `mean(item_embeddings[history])` — a
pre-existing helper at [core.py:2699](src/amazon_recsys/ml/core.py#L2699) that all vector
retrievers share. There is no separate "user tower" whose output distribution differs from
what serving can compute.

In contrast, DAT-Lite trained a user tower (a separate neural network that consumed user
sequence features and produced a user-side vector), but at serving time it was discarded
because the bundle export path strips encoders ([bundles.py:81](src/amazon_recsys/ml/bundles.py#L81)).
The serving query then collapsed to the same `mean(item_embeddings[history])` aggregator —
but the item embeddings had been trained against the *learned* user-tower output, not the
history-mean. Result: a distribution mismatch that produced 0.0 recall over the catalog despite
positive in-batch pair discrimination.

This is the most important sentence in the document: **BLAIR cannot have this failure mode by
construction.**

### Why rich text matters at scale (Phase 2 motivation)

`keep_columns` drops the description, features, and store columns from `item_features.parquet`
to keep the persisted parquet small. Phase 1 BLAIR therefore had access only to `title +
source_category`. At an ~8-item All_Beauty catalog this is fine. At a 1.24 M-item catalog with
descriptions and features that the BLAIR encoder was specifically pretrained on, dropping them
throws away most of the semantic signal BLAIR was designed to exploit.

Phase 2's `_load_rich_metadata` reloads raw metadata in-memory at training time, derives the
five-column concat, and discards it after encoding. The persisted `item_features.parquet` is
unchanged in size. Memory cost: an extra DataFrame of ~1.24M rows × ~5 text columns lives in
RAM during BLAIR's encoding phase, then is freed.

The Phase 2 medium-scale result (BLAIR-rich +50 % test recall@10 vs content_based at 273 users
vs Phase 1's parity at 8 users) confirms the signal-vs-scale trend the BLAIR paper predicts.

### Why we kept Phase D as opt-in despite its debug-scale flop

The 273-user medium-scale test showed that the XGBoost ranker hits its training-validation
ndcg ceiling (~0.95) regardless of negative-distribution composition, because there are only
1,615 ranker training examples at debug scale. Changing the negatives can't add information
that isn't in the data. At prod scale the ranker has ~50,000 training examples (decided by
the `quality-neural` profile's `ranker_train_example_cap`), where the data ceiling is much
higher and the negative-distribution choice may genuinely matter. Phase D's code change costs
nothing when disabled (verified by an exact-equivalence unit test against the legacy uniform
sampler) and is available as a tuning knob if the prod-scale ranker needs it.

## What is still open

1. **The current prod-scale run.** Outcome will determine whether BLAIR-rich actually beats
   the current `prod-2026-05-10-recovery-v1` baseline (final test recall@100 = 0.052) at prod
   scale. If yes, activate. If marginally yes, consider tightening to `blair-v1` gates on a
   follow-up run. If no, examine the failure mode before iterating further.

2. **The ranker overfit pattern.** Even at medium scale the ranker's test recall@10 (0.040) was
   below BLAIR's retriever-only test recall@10 (0.060) — the ranker is **demoting** good
   candidates. Possible follow-ups: investigate XGBoost hyperparameters at prod scale; try
   Phase D's hardneg mix; investigate feature collinearity. Out of scope for the current run.

3. **anonymous_no_history is structurally uncovered by vector retrievers.** All three vector
   paths (BLAIR, content_based, latent_cf) serve via `mean(history_embeddings)`, which is the
   zero vector when history is empty. Only cooccurrence and popularity contribute to this
   slice. This pre-dates the BLAIR work. Addressing it requires a different query path for
   anonymous users (e.g., category-conditioned popularity embeddings) and is a separate
   workstream.

4. **OneDrive sync interference at scale.** Both the original DAT-v2 run and the first
   attempt at this BLAIR run hit `np.save → 0 written` errors caused by OneDrive locking
   partially-written files. Mitigation today is to pause OneDrive sync for the duration of
   training. A more permanent fix would be atomic-rename writes around every large `.npy`
   site; this was prototyped and reverted earlier in this work. It is a candidate Phase H if
   the issue recurs.

## Config surface added

All new env vars are opt-in with defaults that preserve pre-existing behaviour:

| Env var | Default | Phase | Purpose |
|---|---|---|---|
| `AMAZON_RECSYS_NEURAL_RETRIEVER_VARIANT` | `two_tower` | 1 | Adds `blair_text` as a valid value |
| `AMAZON_RECSYS_BLAIR_MODEL_NAME` | `hyp1231/blair-roberta-base` | 1 | HuggingFace id for the frozen encoder |
| `AMAZON_RECSYS_BLAIR_FALLBACK_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 1 | Used when BLAIR cannot be loaded |
| `AMAZON_RECSYS_BLAIR_BATCH_SIZE` | `64` | 1 | Encoder batch size |
| `AMAZON_RECSYS_BLAIR_MAX_SEQ_LENGTH` | `256` | 1 | Tokenizer max sequence length |
| `AMAZON_RECSYS_RANKER_HARDNEG_MIX` | `"0,0,1.0"` | D | Source weights `popularity,cooccurrence,random` |
| `AMAZON_RECSYS_GATE_PROFILE` | `off` | F | One of `off`, `recovery-v1`, `blair-v1` |

## Files modified

| Path | Phase(s) |
|---|---|
| [src/amazon_recsys/ml/core.py](src/amazon_recsys/ml/core.py) | 1, 2, D, G |
| [src/amazon_recsys/ml/bundles.py](src/amazon_recsys/ml/bundles.py) | C, F |
| [src/amazon_recsys/ml/pipelines.py](src/amazon_recsys/ml/pipelines.py) | 1, D, F |
| [src/amazon_recsys/config/settings.py](src/amazon_recsys/config/settings.py) | 1, D, F |
| [src/amazon_recsys/ml/retrievers/blair.py](src/amazon_recsys/ml/retrievers/blair.py) | 1, 2 (new file) |
| [src/amazon_recsys/ml/retrievers/__init__.py](src/amazon_recsys/ml/retrievers/__init__.py) | 1 (new file) |
| [tests/test_blair_phase1.py](tests/test_blair_phase1.py) | 1, 2 (new file) |
| [tests/test_bundle_audit_metadata.py](tests/test_bundle_audit_metadata.py) | C (new file) |
| [tests/test_ranker_hardneg_mix.py](tests/test_ranker_hardneg_mix.py) | D (new file) |
| [tests/test_acceptance_gates.py](tests/test_acceptance_gates.py) | F (new file) |
| [.env](.env), [.env.example](.env.example) | 1 (BLAIR fields added) |

77 tests pass (49 pre-existing + 28 new).

## References

- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. (2024). **Bridging Language and
  Items for Retrieval and Recommendation.** arXiv:2403.03952.
  Introduces BLAIR and the Amazon Reviews 2023 dataset that this codebase trains on. Cited in
  the dataset documentation at [README.md](README.md). The BLAIR HuggingFace repo is at
  `hyp1231/blair-roberta-base`.

- Yu, Y., Wang, W., Feng, Z., Xue, D. (2021). **A Dual Augmented Two-tower Model for Online
  Large-scale Recommendation.** DLP-KDD '21.
  PDF in [Research/A Dual Augmented Two-tower Model for Online Large-scale.pdf](Research/A%20Dual%20Augmented%20Two-tower%20Model%20for%20Online%20Large-scale.pdf).
  Introduces the Adaptive-Mimic Mechanism and Category Alignment Loss that DAT-Lite partially
  implemented. The path we did *not* take — see "Three root causes" above.

- Hou, Y., Mu, S., Zhao, W. X., Li, Y., Ding, B., Wen, J. (2022). **Towards Universal Sequence
  Representation Learning for Recommender Systems.** KDD '22.
  Background for using pretrained text encoders as item representations in sequential recsys.

- McAuley, J., Ni, J., et al. (Amazon Reviews 2023 dataset documentation).
  [https://amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io).
  Source for the review and metadata JSONL files this codebase trains on.

## Process notes (for the next session)

A few discipline rules that emerged painfully during this work. Future Claude should follow them:

1. **Minimum-surface phases.** Each phase is the smallest change that produces an observable
   result. No "while I'm in there" cleanups, no convenience refactors, no surprise scope
   creep. If a change isn't required for the phase's stated goal, it doesn't go in.

2. **Defaults never silently change pipeline behaviour.** Every new config field has a default
   chosen so that an unaware user gets identical results to pre-change code. Phase D's default
   `"0,0,1.0"` is the canonical example: it's bit-for-bit identical to the legacy uniform
   sampler.

3. **Validate at the scale that matters.** Synthetic-fixture unit tests are necessary but not
   sufficient. Each phase was gated by a real single-category debug run before the next phase
   began. Failures at scale that "passed all tests" were the dominant pain.

4. **Never change `keep_columns` on persisted parquets to side-channel data into a downstream
   path.** Phase 2 explicitly rebuilds rich text in-memory rather than persisting it — because
   bloating `item_features.parquet` was the proximate cause of an `np.save → OneDrive
   sync` failure in an earlier iteration of this work.

5. **Don't impose workflow on a working CLI.** The application's existing
   `python -m amazon_recsys.cli.main export-bundle ...` invocation pattern was correct;
   wrapping it in `Tee-Object`, `Start-Transcript`, `2>&1`, or log-file-plus-tail patterns was
   noise that obscured normal stderr behaviour and frustrated the user. Use the CLI as it was
   designed.

6. **`2>&1` on a native exe inside PowerShell 5.1 wraps every stderr line in a NativeCommandError
   ErrorRecord.** This makes a healthy run look like a failure. Either don't redirect stderr,
   or use `Start-Transcript`.

7. **When the user pushes back on a diagnosis, look at your own diff first.** Don't claim
   "pre-existing" without verifying. The most expensive mistake of this work was attributing
   an `np.save → 0 written` failure to OneDrive when the proximate cause was a 3× growth in
   `item_features.parquet` from a `keep_columns` change in the same session.

## How to run the current state

For a debug-scale validation (~10 min) on a single category:

```powershell
$env:AMAZON_RECSYS_CATEGORIES='["All_Beauty"]'
$env:AMAZON_RECSYS_DEV_MODE="true"
$env:AMAZON_RECSYS_DEV_FRACTION="0.1"
$env:AMAZON_RECSYS_K_CORE="2"
$env:AMAZON_RECSYS_TRAIN_POSITIVE_CAP="50000"
$env:AMAZON_RECSYS_SPLIT_EVAL_EXAMPLE_CAP="1000"
$env:AMAZON_RECSYS_RANKER_TRAIN_EXAMPLE_CAP="1000"
$env:AMAZON_RECSYS_RANKER_VAL_EXAMPLE_CAP="250"
$env:AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER="true"
$env:AMAZON_RECSYS_NEURAL_RETRIEVER_VARIANT="blair_text"
python -m amazon_recsys.cli.main export-bundle --run-name beauty-small-blair --run-profile debug --activate
```

For the prod-scale run currently in progress:

```powershell
$env:AMAZON_RECSYS_CATEGORIES='["All_Beauty","Automotive","Industrial_and_Scientific"]'
$env:AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER="true"
$env:AMAZON_RECSYS_NEURAL_RETRIEVER_VARIANT="blair_text"
$env:AMAZON_RECSYS_GATE_PROFILE="recovery-v1"
python -m amazon_recsys.cli.main export-bundle --run-name prod-2026-05-11-blair-v1 --run-profile quality-neural --version prod-2026-05-11-blair-v1
```

Pause OneDrive sync first (system tray → 8 hours) to avoid `np.save → 0 written` interference
on the ~3.6 GB BLAIR item embeddings.

After the run, inspect:

- `notebooks/artifacts/amazon_recsys/prod-2026-05-11-blair-v1/evaluation/blair_text_retriever_metrics.csv`
- `notebooks/artifacts/amazon_recsys/prod-2026-05-11-blair-v1/evaluation/hybrid_union_xgboost_ranker_metrics.csv`
- `artifacts/amazon_recsys/bundles/prod-2026-05-11-blair-v1/retrievers/two_tower/metadata.json`
  (look for `audit_metadata.item_text_source_counts.rich` near the catalog size)
- `artifacts/amazon_recsys/bundles/prod-2026-05-11-blair-v1/manifest.json`
  (`retriever_variants` must contain `"two_tower"`)

If the recovery-v1 gate passes, activate manually:

```powershell
python -m amazon_recsys.cli.main activate-bundle prod-2026-05-11-blair-v1
```

If the gate fails, the bundle does not export and the existing prod bundle
`prod-2026-05-10-recovery-v1` remains active. Read the GateValidationError message; it lists
every gate that failed with the observed value vs the floor.
