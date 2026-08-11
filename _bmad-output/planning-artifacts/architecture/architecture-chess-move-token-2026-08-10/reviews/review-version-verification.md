# Review — Version & Reality-Check Verification

**Lens:** every committed decision was web-researched or reality-checked rather than asserted from training data (library/framework versions, technology existence/fit, live starter defaults where applicable).

**Target:** `ARCHITECTURE-SPINE.md` (Epic CHESS_MOVE_TOKEN, 2026-08-10)

**Reviewer method:** re-ran the spine's own verification commands in the actual `jax_env` conda environment (note: the spine calls it a venv; it is in fact a conda env at `/home/aobled/anaconda3/envs/jax_env`, currently active as default `python3`), inspected installed `flax.linen` source directly, checked `tf.data.Dataset.padded_batch` signature, ran two web searches on Flax's Linen/NNX status and current released version, and spot-checked five code references (`main.py:141-200`, `data_management.py:867-940`, `loss_functions.py:561`, `model_library.py` `MODELS` dict, `task_strategies.py` strategy `__init__`/`primary_metric_name`/`optimization_mode` patterns) against the real files.

---

## Findings

### 1. Core technical claim confirmed accurate, not fabricated (PASS)

Re-ran the checks: `flax.__version__` → `0.10.7`, `jax.__version__` → `0.6.2`, both matching the spine exactly. `hasattr(nn, 'make_causal_mask')` and `hasattr(nn, 'combine_masks')` both `True`. Inspected `nn.MultiHeadDotProductAttention.__call__`'s live signature — it does accept a keyword-only `mask: Array | None = None` argument, matching the spine's claim in the Design Paradigm section and AD-30. Inspected the source of `make_causal_mask`/`combine_masks` directly (`inspect.getsource`) — both present, no deprecation warning, no shim redirecting to `nnx`. The spine's central technical bet is real, not asserted from memory.

### 2. Flax is meaningfully behind current upstream, and the spine doesn't disclose or evaluate this gap (MEDIUM)

Web search shows the current released Flax is **0.12.8** (July 2026), while the project has **0.10.7** installed — roughly a dozen minor releases behind. The spine states version 0.10.7 as if it were simply "the" current fact about Flax, with no acknowledgment that it's an old pin relative to upstream, and no check for whether anything relevant changed in `make_causal_mask`/`combine_masks`/`MultiHeadDotProductAttention` between 0.10.7 and 0.12.8 (e.g., new deprecation warnings, signature changes, or a push toward `nnx`-equivalent helpers). For a spine explicitly building the project's *first* causal-attention pattern, checking only the installed version and not diffing against latest upstream is a gap: if the epic's implementation later needs `pip install -U flax` for any reason (bug fix, CI image bump, etc.), the causal-mask API surface hasn't been verified stable across that jump. This is a real, checkable question the spine did not ask.

### 3. Linen-vs-NNX direction is real but under-flagged (LOW-MEDIUM)

Web search confirms Flax Linen is **not being deprecated in the near term** (most users still on it), but Google's own long-term stated direction is for NNX (or something like it) to become the primary API, and NNX is what's recommended for new projects going forward. The spine's Design Paradigm section asserts "`flax.linen` fournit déjà `nn.make_causal_mask`/`nn.combine_masks`... aucune nouvelle dépendance" — true and adequate for this epic's immediate scope (the project is fully on `linen`, switching is out of scope) — but the spine never states *why* staying on `linen` for a brand-new causal-decoder pattern (the kind of workload NNX's docs increasingly target) is the right call versus a one-time NNX evaluation. This isn't wrong, but it's an asserted-not-reasoned choice on a fork in the road that the epic is explicitly calling "first use of a causal pattern in this codebase" — exactly the moment where a two-line "why not NNX" note would cost little and close the gap. Should be at minimum a Deferred entry with rationale, not silent.

### 4. Code line-number/pattern references are accurate — genuinely verified, not memory-asserted (PASS)

Spot-checked five references against the live files:
- `main.py:141-200` — confirmed: this is exactly the `if task_type == "classification": ... elif ... else: raise ValueError(...)` dispatch block, ending at line 200.
- `data_management.py:867-940` — confirmed: `get_datasets()` function with the `if/elif` task_type dispatch, ending at the `else: raise ValueError(...)` at line 940.
- `loss_functions.py:561` — confirmed: `def compute_chess_policy_loss(policy_logits, policy_targets, label_smoothing=0.0):` starts exactly at line 561.
- `model_library.py` `MODELS` dict — confirmed: `MODELS = {...}` is a literal dict (line 1122) mapping model name strings to factory functions, consumed by `get_model()` via `MODELS[model_name](**kwargs)` — matches the spine's claim that a new entry is a dict addition, not an `if/elif` branch.
- `task_strategies.py` — confirmed `ChessLegalMovesStrategy.__init__(self, metric_threshold: float = 0.5, loss_params: dict = None)` (line 546) and `ChessPolicyValueStrategy.__init__(self, loss_params: dict = None)` (line 476), matching the spine's claims about the `loss_params`-dict constructor pattern.

None of these read as hallucinated; all were reality-checked against the current tree, not just plausible-sounding.

### 5. `tf.data`/`padded_batch` claim is accurate but trivial to verify — no issue found (PASS, minor note)

`tf.data.Dataset.padded_batch(batch_size, padded_shapes=None, padding_values=None, drop_remainder=False, name=None)` exists as described in the installed TensorFlow (2.20.0, confirmed via `tf.__version__`). Spine correctly treats this as "standard API, not an addition" (AD-28). No fabrication here. Minor: the spine's Stack table doesn't state the TF version the way it states flax/jax versions — inconsistent level of rigor across the three Stack rows, though low-stakes since `padded_batch` has been stable across many TF releases.

---

## Overall Verdict

The spine's headline claim — that `flax.linen` 0.10.7 already provides `nn.make_causal_mask`, `nn.combine_masks`, and a `mask`-accepting `MultiHeadDotProductAttention` — is **genuinely verified**, not asserted from training data; I reproduced the exact same checks independently and they match. The code-location citations (`main.py`, `data_management.py`, `loss_functions.py`, `model_library.py`, `task_strategies.py`) are also accurate on spot-check, which is a meaningfully higher bar than most spines clear.

The gap is upstream-currency, not correctness: the spine verified "does this work today, in this repo" but not "is this still the recommended path going forward" — it doesn't note that installed Flax (0.10.7) is many minor versions behind current (0.12.8), and doesn't explicitly reason about the Linen-vs-NNX fork it's implicitly taking at the exact moment it's introducing the codebase's first causal-attention pattern. Neither invalidates the architecture for this scoped spike epic, but both are the kind of thing this review lens exists to catch, and I'd flag both as gap items rather than fabrications.
