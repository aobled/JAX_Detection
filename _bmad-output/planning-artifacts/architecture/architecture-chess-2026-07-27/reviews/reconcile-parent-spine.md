# Reconciliation Review — Chess Epic Spine vs Parent Spine

- Parent: `architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md` (defines AD-9..AD-20, inherits AD-1..AD-8 from `architecture-JAX_Detection-2026-07-12`)
- Child: `architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md` (defines AD-21..AD-25, inherits AD-3, AD-14, AD-17, AD-18)
- Method: read both spines in full; cross-checked every line-number/name citation against the actual repo state (`main.py`, `data_management.py`, `loss_functions.py`, `trainer.py`, `task_strategies.py`, `dataset_configs.py`, the PRD).

## 1. Accuracy of inherited invariants (AD-3, AD-14, AD-17, AD-18)

All four are faithful, non-drifted restatements of the parent text.

- **AD-3** — child's "même fallback de chemin 3 niveaux + réinit des `batch_stats`... aucun chargement nu spécifique" matches the parent verbatim in substance, correctly attributed as passing through the parent from the grandparent (`architecture-JAX_Detection-2026-07-12`).
- **AD-14** — child correctly carries forward "entraînement modulaire, jamais unifié." One minor looseness: the child appends "aucun graphe d'inférence unifié n'existe ni n'est requis par cette epic," which is true but is really a Non-Goal-driven consequence, not literally part of AD-14's own Rule (AD-14's rule is specifically about training staying separate from the JAX Single-Pass *inference* graph, a different concern). Not a misrepresentation, just a slightly generous paraphrase — no action needed.
- **AD-17** — accurate, and re-verified against current code, not just the parent doc: the "3 points de dispatch" citation `main.py:121-166`, `task_strategies.py`, `data_management.py` is correct as of the current codebase (verified `main.py` dispatch block is exactly lines 121-166; parent's own citation was `107-143`, so the child correctly re-checked against current line numbers rather than copying stale ones).
- **AD-18** — accurate. Child's claim that `chess_target_encoding.py` will "mirror `detection_target_encoding.py`" is grounded in reality: `detection_target_encoding.py` exists in the repo root with its own test file (`tests/test_detection_target_encoding.py`), confirming this is a real established pattern, not an invented one.

## 2. New ADs (AD-21..AD-25) vs full parent set (AD-1..AD-20)

No contradictions or weakening found.

- **AD-20 vs AD-21 (the flagged assumption)** — confirmed correct and non-overlapping. AD-20 (parent) binds exclusively to the *old* segmentation pipeline: `FIGHTERJET_DETECTION`, `AircraftDetectorUNet`, `DetectionStrategy`, `DetectionDataset`, `decode_segmentation_and_detect(_batch)`. AD-21 (child) binds to the *new* CenterNet pipeline: `JAX_DETECTOR` config, `aircraft_detector_centernet(_lite)`, `CenterNetDetectionStrategy`, `CenterNetDetectionDataset` — verified all these names/classes exist in the current codebase (`dataset_configs.py:379`, `data_management.py:429`). The two ADs protect two distinct, coexisting pipelines; AD-20 does not cover `JAX_DETECTOR`. The child's own note ("même précédent qu'AD-20, appliqué ici à JAX_DETECTOR qu'AD-20 ne couvre pas") is accurate, not just asserted.
- **AD-24** citations independently verified: `compute_centernet_loss` is at `loss_functions.py:547`; `trainer.py`'s per-step/per-epoch logging (`train_epoch`/`evaluate`, ~245-274 and 493-499) does carry only a single scalar loss and single scalar metric, confirming the "no `trainer.py` change needed" premise is real, not assumed. `generate_reports()`, `primary_metric_name`, `optimization_mode` are indeed existing per-strategy hooks in `task_strategies.py` (multiple existing strategies implement all three), so AD-24's "hook déjà existant, pas de changement d'interface" claim holds.
- **AD-21's FR-7 baseline-diff claim** verified against the PRD: FR-7 and the Non-Goals (§5, chess_game.py integration deferred) exist as cited, worded consistently with what AD-14/AD-21/AD-22/Deferred attribute to them.
- AD-22, AD-23, AD-25 touch only new chess-only artifacts (policy head, attention bottleneck, dataset schema) with no binds overlapping any parent artifact — no conflict surface exists.

## 3. Omitted parent ADs (AD-1, AD-2, AD-6, AD-7, AD-8, AD-9..AD-16, AD-19)

Omissions are defensible, not gaps.

- **AD-1, AD-2, AD-6, AD-8** — all scoped to `inference_utils.py` / the JAX Single-Pass inference composition / video throughput / `tools/` consumers of the *inference* path. The chess epic explicitly excludes inference (`chess_game.py` integration is a PRD §5 Non-Goal); nothing in the epic touches `inference_utils.py`. Correctly out of scope.
- **AD-7** — a sequencing/process precedent for the JAX Single-Pass build order, not a standing artifact-level rule the chess epic must re-cite; low materiality, its omission doesn't create risk.
- **AD-9, AD-10, AD-11, AD-12, AD-13, AD-15, AD-16, AD-19** — all bind exclusively to `JAX_DETECTOR`'s detection architecture (CenterNet decode, crop/resize, canonical resolution, RESCALE symmetry, fixed-size output, two-config split, anchor-based deferral). The chess epic doesn't modify or depend on any of this machinery (AD-21 explicitly isolates it) — correctly irrelevant.
- **AD-14** was inherited but AD-4/AD-5 (already excluded even in the parent) require no re-check.

No material gaps or conflicts found. All four inherited invariants are accurately restated (one minor stylistic looseness noted on AD-14, not requiring a fix), AD-20/AD-21 are correctly non-overlapping as the author suspected, and every omitted parent AD is legitimately out of scope for a training-only, inference-excluded epic.
