---
name: 'Review — Version & Reality-Check Verification'
type: reviewer-gate
target: architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md
lens: 'Verify every committed decision was web-researched or reality-checked rather than asserted from training data'
date: '2026-07-27'
---

# Review — Version & Reality-Check Verification

## Verdict: PASS with 2 minor findings (non-blocking)

The core Stack claims (flax 0.10.7, python-chess 1.11.2, `nn.MultiHeadDotProductAttention`) are
**confirmed accurate** by direct inspection of the running environment — not asserted from training
data. Two smaller inaccuracies were found elsewhere in the spine, both reality-check misses against
the existing project rather than library-version risk.

## 1. Installed versions — CONFIRMED

Checked directly in the project's actual environment (`jax_env`, per prior memory note on
`jax_env` vs `.venv`):

```
$ which python
/home/aobled/anaconda3/envs/jax_env/bin/python

$ pip show flax
Name: flax
Version: 0.10.7
Location: .../envs/jax_env/lib/python3.10/site-packages

$ pip show chess
Name: chess
Version: 1.11.2
Location: .../envs/jax_env/lib/python3.10/site-packages
```

Both exactly match the versions cited in the spine's Stack table. **Confirmed**, not asserted.

## 2. `nn.MultiHeadDotProductAttention` in flax 0.10.7 — CONFIRMED

```
$ python -c "import flax.linen as nn; print(nn.MultiHeadDotProductAttention)"
<class 'flax.linen.attention.MultiHeadDotProductAttention'>
```

The class exists, is exported at `flax.linen.MultiHeadDotProductAttention`, and is importable in the
installed package — not merely documented in a version that might differ from what's installed. AD-23's
dependency on this symbol is safe.

## 3. Currency of flax 0.10.7 / python-chess 1.11.2 — CONFIRMED CURRENT (live-checked)

Checked live against PyPI's index (not training-data recall, which would be unreliable for "is this
the latest" as of any recent date):

```
$ pip index versions flax
flax (0.10.7)
  INSTALLED: 0.10.7
  LATEST:    0.10.7

$ pip index versions chess
chess (1.11.2)
  INSTALLED: 1.11.2
  LATEST:    1.11.2
```

Both are the latest published release on PyPI at review time. **No staleness flag warranted** — the
spine's "no new dependency needed" claim is not just true today but sitting on the current version,
so there is no hidden upgrade-debt being smuggled in.

## 4. Other technology/reality claims in the spine — spot-checked

The spine makes several other "verified in code" claims (mostly in AD-17, AD-24, and the Consistency
Conventions table) that fall under this lens because they assert facts about the existing project
rather than external libraries. Spot-checked against the actual repo:

| Claim | Location in spine | Result |
| --- | --- | --- |
| Dispatch points `main.py:121-166`, `task_strategies.py`, `data_management.py` | AD-17 | **Confirmed** — `main.py:121-166` is exactly the `task_type` if/elif dispatch block; `ClassificationStrategy`/`DetectionStrategy`/`CenterNetDetectionStrategy`/`KeplerStrategy` all exist in `task_strategies.py`. |
| `trainer.py:245-274,493-499` log/scalar-only claim | AD-24 | **Confirmed** — those line ranges are exactly the `train_step`/`eval_step` scalar loss+metric delegation and the epoch print loop; no per-head loss logging exists there. |
| `compute_centernet_loss` at `loss_functions.py:547` | AD-24 | **Confirmed** — line 547 is precisely the `def compute_centernet_loss(...)` line. |
| `chess/chess_game.py` already uses `python-chess` | Design Paradigm / Stack | **Confirmed** — `chess/chess_game.py:1-2` does `import chess` / `import chess.engine`. |
| `detection_target_encoding.py` exists at repo root (mirror target for `chess_target_encoding.py`) | Structural Seed | **Confirmed** — file exists at root. |
| Naming convention "cohérent avec `JAX_DETECTOR`/`KEPLER`/`FIGHTERJET_DETECTION`" | Consistency Conventions table | **Inaccurate** — see Finding A below. |
| `chess_pgn_dataset_tools.py` "miroir de `jax_detector_dataset_tools.py`", placed at repo root in Structural Seed | Consistency Conventions / Structural Seed | **Inaccurate placement** — see Finding B below. |

### Finding A (minor) — `dataset_configs.py` key is `JAX_KEPLER`, not `KEPLER`

The spine's naming-convention row states the new `CHESS` entry is "majuscules, cohérent avec
`JAX_DETECTOR`/`KEPLER`/`FIGHTERJET_DETECTION`". Checked `dataset_configs.py` directly:

```
$ grep -n '"JAX_DETECTOR"\|"KEPLER"\|"FIGHTERJET_DETECTION"' dataset_configs.py
140:    "FIGHTERJET_DETECTION": {
379:    "JAX_DETECTOR": {
$ grep -n "KEPLER" dataset_configs.py
230:    "JAX_KEPLER": {
```

The real key is `JAX_KEPLER`, not `KEPLER`. Cosmetic — doesn't change the `CHESS` naming decision
itself — but it's a reality-check miss: the citation implies verified code inspection and the exact
string is wrong.

### Finding B (worth a check before implementation) — mirror-file location not verified

The Structural Seed places `chess_pgn_dataset_tools.py` directly under `jax_supervised_training/`
(repo root), and the Consistency Conventions table describes it as "miroir de
`jax_detector_dataset_tools.py`". Checked where the actual mirror target lives:

```
$ find . -iname "jax_detector_dataset_tools.py"
./dataset_builder/jax_detector_dataset_tools.py

$ ls dataset_builder/
cifar10_classification_dataset_tools.py
fighterjet_classification_dataset_tools.py
fighterjet_detection_dataset_tools.py
jax_detector_dataset_tools.py
kepler_dataset_tools.py
```

`jax_detector_dataset_tools.py` — and every sibling `*_dataset_tools.py` file in the project — lives
in `dataset_builder/`, not at the repo root. The spine's Structural Seed tree diagram shows
`chess_pgn_dataset_tools.py` as a top-level file alongside `main.py`, `model_library.py`, etc., which
contradicts the very mirror pattern the spine cites to justify the filename. This wasn't reality-checked
against the actual `dataset_builder/` directory. It's a placement detail, not an architectural
decision reversal (AD-25's data-format rule is unaffected either way), but it should be corrected —
either move the seed entry into `dataset_builder/`, or if root placement is intentional, drop the
"miroir de `jax_detector_dataset_tools.py`" justification since it currently points at a location
convention the new file doesn't follow.

## Summary

- flax 0.10.7 / python-chess 1.11.2 installed-version claims: **verified true**, not asserted.
- `nn.MultiHeadDotProductAttention` availability: **verified true** by direct import.
- Both versions are **current** (latest on PyPI), confirmed live — no upgrade-check debt hidden here.
- Two secondary claims in the spine (KEPLER config key naming; `jax_detector_dataset_tools.py` mirror
  location) were asserted without matching what's actually in the repo. Neither invalidates an AD;
  both are cheap to fix before story-writing.
