---
title: Reviewer Gate — Rubric Walk
target: architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md
reviewer: rubric-walker (subagent)
date: 2026-07-27
---

# Reviewer Gate — Rubric Walk: architecture-chess-2026-07-27

## Verdict

**Conditional pass — two Should-Fix gaps before build starts.** The spine's core mechanics (paradigm ratification, five new ADs, dependency graph, Structural Seed) are sound, enforceable, and consistent with the inherited parent. All spot-checked codebase citations that were actually written into the spine's rules (trainer.py, loss_functions.py, main.py, flax/chess versions) verified exactly. The gaps found are: one bound requirement (NFR-2) with no enforcing rule anywhere in the body, and one Structural Seed path that contradicts the existing brownfield layout convention.

## Findings by severity

- **Should-Fix: 2**
- **Minor/Nit: 2**
- **Blocking: 0**

## Findings

### 1. [Should-Fix] NFR-2 is bound in frontmatter but has zero enforcing Rule in the body

The frontmatter `binds:` list includes `NFR-2` (PRD: "no external chess engine dependency for label generation, not even as a fallback — excluded by construction"). Searched the full spine body for any AD, Rule, Consistency Convention, or Deferred item addressing this: none exists. The word "Stockfish" / "externe" / "moteur" appears nowhere outside the frontmatter binds list.

This is a real, PRD-flagged temptation, not a hypothetical: PRD §8 Open Question #5 explicitly floats noisy human-move labels and value-label noise as an open risk, and §5 Non-Goals explicitly forecloses "external engine as fallback if the value head proves noisy." AD-25 fixes the dataset *example format* (3 fields, no metadata) but says nothing about *how* the value/policy fields must be sourced — a builder hitting noisy value labels in practice has no architectural rule stopping them from reaching for Stockfish as a fallback, which is exactly the scenario NFR-2 is meant to prevent. Add a one-line Rule (either folded into AD-25 or a new short AD) stating labels are sourced exclusively from PGN replay via `python-chess`, never from engine evaluation, with no fallback exception.

### 2. [Should-Fix] Structural Seed places `chess_pgn_dataset_tools.py` at repo root, contradicting the actual dataset-tool-builder convention

Verified against the live tree: every existing dataset-builder script (`jax_detector_dataset_tools.py`, `fighterjet_detection_dataset_tools.py`, `fighterjet_classification_dataset_tools.py`, `cifar10_classification_dataset_tools.py`, `kepler_dataset_tools.py`) lives under `dataset_builder/`, not at the package root. The parent spine's own Structural Seed (architecture-jax_supervised_training-2026-07-15) also places its new `fighterjet_detection_dataset_tools_v2.py` implicitly at root in text but the actual codebase already moved this whole family into `dataset_builder/` since — the child spine's seed didn't check the current location and just mirrors the older (now-superseded) root-level pattern.

The child spine's Structural Seed lists `chess_pgn_dataset_tools.py` directly under `jax_supervised_training/` alongside `main.py`/`model_library.py`, not under `dataset_builder/`. A builder following the seed literally creates a new script that breaks the established one-drawer-per-domain convention for dataset builders. Low effort fix: move the line to `dataset_builder/chess_pgn_dataset_tools.py` in the seed.

### 3. [Minor] Consistency Conventions table cites the wrong existing config key

"`dataset_configs.py` = `CHESS` (majuscules, cohérent avec `JAX_DETECTOR`/`KEPLER`/`FIGHTERJET_DETECTION`)" — the actual key in `dataset_configs.py:230` is `JAX_KEPLER`, not `KEPLER`. `CHESS` itself is still a fine, convention-consistent choice; only the parenthetical precedent citation is wrong. Likely inherited from the PRD's own informal table (§1), which also says "KEPLER" loosely rather than the real constant.

### 4. [Minor] Inherited AD-17 citation claims a `task_type` dispatch point in `task_strategies.py` that doesn't exist

The child spine restates (from the parent) that the `task_type` literal is "référencé identiquement aux 3 points de dispatch vérifiés dans le code actuel : `main.py:121-166`, `task_strategies.py`, `data_management.py`." Verified: `main.py:121-166` is an exact match (confirmed line-for-line — `task_type = config.get(...)` through the closing `raise ValueError`), and `data_management.py` does dispatch on `task_type` (`data_management.py:613-664`). But `task_strategies.py` has zero occurrences of the string `task_type` — it only defines the strategy classes (`ClassificationStrategy`, `DetectionStrategy`, `CenterNetDetectionStrategy`, `KeplerStrategy`); dispatch-by-literal happens exclusively in `main.py` and `data_management.py`. This inaccuracy originates in the parent spine's own AD-17 text and is carried forward as if freshly verified. Not risky enough to block (the actual enforceable content — one class, one literal, defined once — still holds), but a builder scanning `task_strategies.py` for a `task_type` string to mirror will find nothing there.

## Rubric walk detail

- **Fixes the real divergence points, misses none** — Mostly yes (double-loss TaskStrategy, dataset exchange format, non-regression isolation, output space, bottleneck mechanism are all real divergence points the epic introduces, all covered). One miss: NFR-2 (see Finding 1).
- **Every AD's Rule is enforceable and prevents its stated divergence** — Yes for all five new ADs (AD-21 through AD-25). Each names concrete files/functions/values and a verification method (AD-21's baseline-diff, AD-24's exact loss/metric shape). No vague "should" language found.
- **Nothing under Deferred could let two units diverge** — Checked each Deferred item; the two that touch a shared boundary (move→index encoding scheme, dataset metadata) are both pinned to live in the single shared module mandated by inherited AD-18, so leaving the exact schema unspecified at spine level doesn't create a two-implementations risk. K (token count) is purely internal to one model class, not shared. Fine.
- **Named tech is verified-current** — Confirmed exactly: `flax` 0.10.7 and `chess` (python-chess) 1.11.2 are the installed versions in `jax_env`, matching the Stack table precisely, no new dependency required.
- **Ratifies rather than contradicts brownfield** — Mostly yes, with the one real contradiction in Finding 2 (dataset-tool file placement).
- **Covers the driving spec's capabilities** — FR-1 through FR-7 all map to at least one AD or inherited invariant, except NFR-2's "no external engine" constraint (Finding 1).
- **No new AD weakens/contradicts an inherited one** — Checked AD-3, AD-14, AD-17, AD-18: all four are extended consistently, none narrowed or contradicted.
- **Every owned dimension decided/deferred/open** — Deployment/environment is explicitly addressed in Deferred ("hérité sans changement... aucune nouvelle dimension opérationnelle") per the parent precedent — correctly not left silent.

## Codebase spot-checks performed

| Citation in spine | Verified against | Result |
|---|---|---|
| `trainer.py:245-274,493-499` — single scalar loss/metric per step | Read lines 245-274 and 493-499 | Exact match — `train_step`/`eval_step` return one `loss`/`acc` scalar pair; epoch log prints one `Loss=`/`{primary_metric_name}=` pair |
| `loss_functions.py:547` — `compute_centernet_loss` composite-loss pattern | `grep -n "^def compute_centernet_loss"` | Exact line match; function does `heatmap_weight * heatmap_loss + size_weight * size_loss`, the pattern AD-24 mirrors |
| `main.py:121-166` — task_type dispatch | Read lines 100-170 | Exact match — line 121 is `task_type = config.get(...)`, line 166 is the closing `raise ValueError` |
| `flax.linen` 0.10.7 / `chess` 1.11.2 already installed | `pip show` in `jax_env` | Both versions match exactly, no new dependency needed |
| AD-21 binds: `aircraft_detector_centernet(_lite)`, `CenterNetDetectionStrategy`, `CenterNetDetectionDataset` | grep in `model_library.py`/`task_strategies.py`/`data_management.py` | All exist as named |
| `detection_target_encoding.py` exists (AD-18 mirror precedent) | `ls` | Exists at repo root |
| `data_management.py` task_type dispatch (AD-17 inherited) | Read lines 605-665 | Confirmed, three-way `if/elif` on `task_type` |
