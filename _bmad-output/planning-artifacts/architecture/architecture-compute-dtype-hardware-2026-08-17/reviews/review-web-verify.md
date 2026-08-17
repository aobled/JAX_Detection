# Web/Reality Verification Review — architecture-compute-dtype-hardware-2026-08-17

Lens: every committed decision must be web-researched or reality-checked, not asserted from
training data alone. Focus: AD-4's BatchNorm claim, currency of named libraries, and version
consistency with the actually installed environment.

## Method

1. Read `jax`/`flax` directly from the project's active environment (`jax_env`) rather than
   trusting comments or prior memory.
2. Read the installed `flax==0.10.7` source for `flax.linen.BatchNorm`/`LayerNorm`
   (`flax/linen/normalization.py`) line by line.
3. Ran a throwaway script that actually instantiates and applies `nn.BatchNorm(dtype=jnp.bfloat16)`
   and inspects the resulting `batch_stats` dtypes empirically — not just read the source.
4. Fetched the current `main` branch of `google/flax` on GitHub to confirm the mechanism is
   unchanged in the latest upstream code, not just the older installed version.
5. Queried the PyPI JSON API directly (`pypi.org/pypi/<pkg>/json`) for latest released versions
   and upload dates of `jax` and `flax`, compared against installed versions.

## 1. AD-4 BatchNorm claim — VERIFIED, mechanism precisely confirmed (with one nuance)

**Claim under test:** "`flax.linen.BatchNorm` accumule ses statistiques en `float32` en interne
quel que soit `dtype`, donc aucune exclusion n'est nécessaire."

**Result: TRUE, both by source inspection and empirical execution against the installed
`flax==0.10.7`.**

Exact mechanism (not what AD-4's one-line summary implies, but consistent with it):

- `BatchNorm` has **three** separate dtype-related knobs, not one: `dtype` (default `None`,
  infers from input — this is what `compute_dtype` will be passed into), `param_dtype` (default
  `jnp.float32`, governs `scale`/`bias` param init), and `force_float32_reductions` (default
  **`True`**, a distinct boolean flag).
- The running-average variables `batch_stats/mean` and `batch_stats/var` are initialized as:
  `jnp.float32 if self.force_float32_reductions else self.param_dtype` — i.e. their dtype is
  driven by `force_float32_reductions` (and secondarily `param_dtype`), **never by `dtype`**.
- Per-batch statistics computed in `_compute_stats` are promoted via
  `jnp.promote_types(dtype, jnp.float32)` whenever `force_float32_reductions=True` — so even if
  `dtype=bfloat16` is passed, the stats used to update the running average are computed and
  stored in float32.
- `dtype` only governs the **output** of the layer (`_normalize`'s returned normalized
  activations) — confirmed both by source (`_normalize(..., dtype, param_dtype, ...)`, doc
  comment "`dtype`: the dtype of the result") and empirically below.

**Empirical confirmation (executed against installed flax 0.10.7):**

```python
bn = nn.BatchNorm(dtype=jnp.bfloat16, use_running_average=False)
v = bn.init(key, x_bfloat16)
# v['batch_stats'] -> {'mean': float32, 'var': float32}
y, new_state = bn.apply(v, x, mutable=['batch_stats'])
# y.dtype -> bfloat16          (output/compute precision, as requested)
# new_state['batch_stats']    -> {'mean': float32, 'var': float32}  (unchanged, still float32)
```

This is exactly the invariant AD-4 needs: passing `dtype=self.compute_dtype` (including
`bfloat16` on TPU) to `nn.BatchNorm` changes only the forward-pass output dtype, never the
running-mean/running-var storage dtype, which stays `float32` because of the *separate*,
independently-defaulting `force_float32_reductions=True` flag — a flag the AD-4 plan does not
touch (it only sets `dtype=self.compute_dtype`, leaving `param_dtype` and
`force_float32_reductions` at their class defaults).

**Nuance worth noting in the spine (minor, not a correctness bug):** AD-4's phrasing
("quel que soit dtype") reads as if `dtype` itself is simply ignored for statistics, which is
the *practical* outcome here but not the literal mechanism — the real reason is a second,
independent parameter (`force_float32_reductions`, default `True`) that this refactor is not
changing. If a future change to `model_library.py` ever passed `force_float32_reductions=False`
explicitly (nobody currently does), the "no exclusion needed" guarantee would silently stop
holding, because stats would then fall back to `param_dtype` (default float32, but also
overridable). Recommend a one-line clarifying footnote in AD-4 or AD-6, e.g.: "guarantee holds
as long as `force_float32_reductions` (default `True`) is left untouched," so a future reader
doesn't try to "fully unify" precision by touching that flag too and unknowingly break the
invariant this AD relies on.

**Cross-version check:** fetched `google/flax` `main` branch (current upstream, well past the
installed 0.10.7) — `force_float32_reductions: bool = True` and the same float32-forced
`ra_mean`/`ra_var` initialization are still present, unchanged. The mechanism is not a
version-specific quirk of the older installed release; it is stable across the version range
this project could plausibly upgrade into.

**`LayerNorm` sanity check:** `nn.LayerNorm` has no `batch_stats` collection at all — no running
average is accumulated; mean/var are recomputed per call from the current activations. So there
is no equivalent "stats precision" question for `LayerNorm` in the first place — AD-4's
"no exception needed" for `LayerNorm` is correct for a different, simpler reason than for
`BatchNorm` (nothing persists, so nothing to protect). The spine doesn't distinguish this, but
it doesn't need to for the rule to hold.

**Signature check:** `dtype` is a real constructor parameter on `nn.Conv`, `nn.Dense`,
`nn.BatchNorm`, and `nn.LayerNorm` in the installed flax 0.10.7 (confirmed via
`inspect.signature`) — AD-4's "applies uniformly to all four call sites" is mechanically valid;
all four accept the same named parameter.

## 2. Installed vs. latest library versions

| Package | Installed (jax_env) | Installed release date | Latest on PyPI (checked live) | Latest release date |
| --- | --- | --- | --- | --- |
| `jax` | 0.6.2 | 2025-06-17 | 0.11.0 | 2026-07-16 |
| `flax` | 0.10.7 | 2025-07-02 | 0.12.8 | 2026-07-20 |

Both are meaningfully behind current upstream (roughly a year, several minor versions). This is
**pre-existing project state, not something this spine introduces or needs to fix** — the spine
correctly scopes itself to the installed environment (it doesn't claim to target latest
upstream), and the BatchNorm mechanism this spine leans on (AD-4) is confirmed unchanged between
installed 0.10.7 and current `main`, so nothing in AD-4/AD-6 is at risk from this version gap.
Flagging only so it's visible: if `jax`/`flax` are ever bumped, AD-4's guarantee should be
re-checked the same way (quick empirical script), since it depends on a default value
(`force_float32_reductions=True`) that a library upgrade could in principle change — though no
such change is signaled in the current upstream source.

No version pin for `jax`/`flax` exists in `requirements.txt` itself (the file documents that
these are platform-dependent and installed separately for CUDA/TPU builds) — consistent with
prior architecture reviews in this repo (`architecture-jax_supervised_training-2026-07-15`,
`architecture-chess-2026-07-27`) which already verified and recorded `jax==0.6.2` /
`flax==0.10.7` as the live installed versions. This spine's claims are consistent with that
established baseline; no drift found.

## 3. Other claims in the spine referencing library behavior

- AD-3's claim that `aircraft_detector_unet`/`centernet`/`centernet_lite` accept `**kwargs` but
  never forward it to their constructor is a claim about **this project's own code**, not a
  third-party library — out of scope for web verification, but worth noting it's an internal
  code claim, not an external one, so it doesn't carry the same staleness risk this lens is
  checking for.
- No other AD in this spine makes a specific claim about third-party library internals besides
  AD-4. AD-1/AD-2/AD-3/AD-5/AD-6/AD-7 are about this project's own `main.py`/`model_library.py`/
  `dataset_configs.py` conventions, not framework behavior, so they carry no equivalent
  verification risk.
- No named technology in this spine (`flax.linen`, `jax`) is deprecated, renamed, or replaced as
  of 2026-08 — both remain the actively maintained, correct choice; `jax.default_backend()` and
  `jnp.dtype` are still current APIs in the installed and latest versions alike (checked source
  during this review; no deprecation warnings encountered).

## Verdict

**PASS.** AD-4's load-bearing claim is correct and was independently verified — not just
re-derived from training data — via direct source reading of the installed flax 0.10.7,
empirical execution of the actual API, and cross-check against current upstream `main`. One
minor wording/robustness nuance flagged above (the guarantee rides on `force_float32_reductions`
defaulting `True`, a second parameter distinct from `dtype`, which the spine's prose doesn't
name) — recommended as a one-line clarifying addition, not a blocker. No other web-verifiable
claim in the spine was found asserted without backing, and no version-incompatibility risk was
found between the spine's assumptions and the actually installed `jax==0.6.2`/`flax==0.10.7`.
