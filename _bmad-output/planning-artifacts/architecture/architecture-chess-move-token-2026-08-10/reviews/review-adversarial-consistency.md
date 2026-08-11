# Adversarial Consistency Review — Epic CHESS_MOVE_TOKEN spine

**Lens:** construct two spine-compliant builders one level down and find where they'd build incompatible things — clashing shapes, dual ownership, conflicting mutation paths.

**Target:** `_bmad-output/planning-artifacts/architecture/architecture-chess-move-token-2026-08-10/ARCHITECTURE-SPINE.md`

**Method:** read the spine, its two parents (`architecture-chess-2026-07-27`, `architecture-jax_supervised_training-2026-07-15`) for AD-3/14/17/18/21/22, and the actual code paths the spine binds to (`main.py`, `trainer.py`, `task_strategies.py`, `data_management.py`, `model_library.py`). Two claims below were verified experimentally rather than by reading alone (see evidence).

**Verdict: NOT READY.** One finding (dtype cast) is a concrete, verified correctness bug that the spine's own "zero-touch Trainer" invariant walks straight into; a compliant builder produces a silently-corrupted model with no error, no test able to catch it by shape-checking alone. A second (left-padding via `padded_batch`) is a verified API-existence gap: the mechanism AD-28 mandates does not exist as described. Both need a new/tightened AD before this spine is safe to hand to two independent builders.

---

## Finding 1 (Critical) — Global float16 cast in `trainer.py` collides PAD with BOS and corrupts move-token IDs above 2048

**Binds violated/unaddressed:** AD-28, AD-29, AD-30, and the spine's own Consistency Convention "Aucune modification de `trainer.py` (même discipline 'zero-touch Trainer' qu'AD-24 du spine chess)".

`main.py:35` and `:40` set `dtype = jnp.float16` unconditionally (TPU branch and GPU branch both), passed into `Trainer(dtype=dtype)`. `trainer.py:313` and `:430` then do:

```python
images = jnp.array(images_np, dtype=self.dtype)
```

— before calling `state.apply_fn(vars, images, training=..., ...)`. This cast applies to **every** `task_type`, including a would-be `chess_move_token`, because the spine explicitly forbids touching `trainer.py`. Every previous domain fed this line pixel/plane floats (already ≤1.0 or already lossy-tolerant), so the cast was invisible. `chess_move_token` is the first domain whose "images" tensor is actually a batch of **integer token IDs** (0–4673, AD-29) that must survive to an embedding lookup untouched.

float16 only represents integers exactly up to 2048; beyond that the spacing doubles at each power of two. Verified directly:

```
2049 -> float16 -> 2048.0   (corrupted)
4095 -> float16 -> 4096.0   (corrupted)
4097 -> float16 -> 4096.0   (corrupted)
4672 (BOS) -> float16 -> 4672.0   (ok, by luck of alignment)
4673 (PAD) -> float16 -> 4672.0   (corrupted — collides with BOS)
```

**PAD (4673) is indistinguishable from BOS (4672) after this cast.** Any model that derives its padding mask by comparing token id == PAD_ID (the only mechanism available — see Finding 1b) will misclassify every PAD position as BOS, and every real move token above 2048 collapses onto a neighboring, different, real move — silent training-data corruption, not a crash.

Two spine-compliant builders diverge exactly as the lens predicts:
- **Builder A** (dataset + strategy) follows AD-27/AD-28/AD-29 to the letter, trusts the "zero-touch Trainer" convention, and never looks at `trainer.py` — ships `move_tokens` as int32, unaware they get force-cast to float16 downstream.
- **Builder B** (model) reads AD-30, builds the causal+padding mask by comparing `tokens == PAD_ID`, correct in isolation — but receives a float16 tensor with already-collided BOS/PAD and already-collided high move indices, and has no way to recover the original integers.

Neither builder is individually non-compliant with any stated AD. The spine never surfaces that its own inherited "don't touch `trainer.py`" invariant (AD-24 heritage, restated in Consistency Conventions) is incompatible with feeding raw integer indices through the one tensor channel `trainer.py` forwards to the model.

**Fix direction:** a new AD is needed, e.g. "the model's embedding lookup must reconstruct exact integer token IDs from the float16-cast tensor it receives from `trainer.py` (e.g. round-then-cast to int32 *before* any further float16 use), and the spine must state the exact float16-safe token-ID ceiling (2048) as a design constraint — or explicitly permit a `trainer.py` touch (e.g. a `preserve_dtype`/task_type-conditional cast) as an exception to zero-touch, scoped and justified." Silence on this is not a safe default.

---

## Finding 1b — AD-28's Binds line misattributes where the padding mask can be built

**Binds:** AD-28 (`Binds:` line names `ChessMoveTokenDataset.create_tf_dataset`, `ChessMoveTokenStrategy.preprocess_batch`, and the model).

`trainer.py:230-272` shows the actual call contract: `train_step`/`eval_step` call `self.strategy.preprocess_batch(images, targets, ...)` and then forward **only** the returned `images` (a single tensor) to `state.apply_fn(vars, images, training=..., mutable=[...], rngs=...)`. There is no second argument slot for a mask. `preprocess_batch` cannot inject a separate padding-mask array into the model call without changing `trainer.py`'s signature — which the spine forbids.

So AD-28 listing `ChessMoveTokenStrategy.preprocess_batch` as a bound owner of "the padding mask" invites exactly the two-owner clash the lens is looking for: a builder implementing `preprocess_batch` could reasonably try to return `(tokens, padding_mask)` packed as `images` (e.g., a dict or stacked array) to satisfy AD-28's Binds line, while a builder implementing the model expects a plain `(B, L)` int tensor per AD-30's seed description ("embedding → N blocs → Dense(4672)"). One of the two has to be wrong, and the spine doesn't say which — the mask, in reality, can *only* be built inside the model itself given `trainer.py`'s fixed signature. AD-28's Rule text (which does put "Le masque d'attention combine..." in the context of the model) already gets this right; the `Binds:` line contradicts it by implying `preprocess_batch` has a real role here.

**Fix direction:** tighten AD-28's `Binds:` to drop `ChessMoveTokenStrategy.preprocess_batch` (or explicitly state its role is limited to dtype/shape passthrough, no mask logic), and add an explicit sentence: "the padding mask is derived entirely inside `chess_move_token_transformer` from the token-id tensor it receives — `trainer.py`'s fixed `apply_fn(vars, images, ...)` signature carries no second channel for a mask, and `preprocess_batch` must not attempt to smuggle one through the `images` return value."

---

## Finding 2 (High) — AD-28 mandates left-padding via `padded_batch`, but `tf.data.Dataset.padded_batch` has no left-padding mode

Verified directly:

```python
ds = tf.data.Dataset.from_generator(lambda: iter([[1,2,3],[4,5]]),
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32))
ds = ds.padded_batch(2, padding_values=99)
# -> [[1,2,3],[4,5,99]]   (padding appended at the END, not the start)
```

`padded_batch`'s `padded_shapes`/`padding_values` control target shape and fill value, but padding is always appended at the end of each ragged dimension — there is no argument to left-align. AD-28's Rule states flatly: "`padded_batch` avec le token `PAD` (AD-29) en **padding à gauche**" as if this were a direct API call, but the described behavior requires an additional transformation the spine never names (the standard workaround is reverse-sequence → `padded_batch` (right-pad) → reverse-back, or building the padded tensor manually outside `padded_batch`).

This is a real fork point for two builders:
- **Builder A** reverses each sequence (after BOS-prepend) before batching, then reverses the batched+padded tensor back — correct, but the reversal must happen *after* BOS is prepended (BOS ends up leftmost of the real tokens) or *before*, and the spine doesn't say which; get it backwards and BOS lands at the wrong end relative to PAD.
- **Builder B**, unaware `padded_batch` can't left-pad, falls back to constructing the padded array manually in Python/NumPy inside `gen()` — which directly collides with AD-27's own explicit prohibition ("un builder qui matérialise `move_tokens`/`move_token_offsets` en séquences paddées AVANT le pipeline `tf.data`... reconstruire les séquences à la volée depuis les offsets CSR dans `gen()`"). AD-27 forbids pre-materializing padded sequences before the `tf.data` pipeline; AD-28 demands a padding behavior the pipeline's own API can't natively produce. The two ADs jointly leave no compliant path spelled out.

**Fix direction:** AD-28's Rule needs to name the actual mechanism (e.g., "reverse each sequence after BOS-prepend, `padded_batch` right-pad with PAD, reverse the batched tensor back along the sequence axis — this is what 'padding à gauche' means operationally here") so two builders converge on the same tensor layout instead of each inventing their own.

---

## Finding 3 (Medium) — AD-27's "split by fraction of examples" doesn't rule out reintroducing a bias the sibling loaders were fixed specifically to avoid

`data_management.py`'s existing chess loaders (`ChessPolicyValueDataset`, `ChessLegalMovesDataset`) both shuffle chunks with a **fixed seed before slicing** the val fraction, and the in-code comment says explicitly this exists to avoid "un split 'toujours les derniers par ordre alphabétique' [qui] biaise systématiquement vers les dernières parties du fichier PGN source (trouvé en code review, 2026-07-27)".

AD-27's Rule for the new single-file loader says only: "split train/val par fraction d'**exemples** (pas de notion de chunk possible ici...)" — it does not say the examples must be shuffled (with a fixed, reproducible seed) before the split. Since `move_tokens`/`position`/`policy` in the spike npz are concatenated **per game, in file order** (per `docs/spike-chess-move-token-dataset-schema.md`), a builder who takes a contiguous slice (e.g., `positions[:n_train]` / `positions[n_train:]`) reproduces exactly the same "last games always in val" bias the sibling loaders' shuffle-then-split was built to prevent — and does so while being fully AD-27-compliant, since AD-27 never says "shuffle first."

A second builder who *does* carry over the sibling convention (shuffle with `shuffle_seed=42` before slicing) produces a materially different train/val partition (different generalization signal, different leakage profile across positions from the same game) than the first builder — both spine-compliant, functionally incompatible for any CAP-4 comparison against the `CHESS_SEARCH_TEACHER` baseline that this spike explicitly exists to support.

**Fix direction:** AD-27's Rule should state explicitly whether the per-example split is a contiguous slice or a seeded shuffle-then-slice, and if seeded, name the same `shuffle_seed` convention/default as the sibling loaders for consistency (or explicitly justify a different one).

---

## Finding 4 (Medium) — BOS/PAD (4672/4673) lack the "single named constant" discipline AD-22 established for `NUM_MOVES`

The parent spine's AD-22 explicitly closes exactly this class of hole for the policy output size: "**Taille de l'espace de sortie à source unique :** `chess_target_encoding.py` définit une constante nommée unique... jamais un littéral dupliqué." AD-29 here states the BOS/PAD values in prose ("BOS = 4672 (préfixé...)", "PAD = 4673") but does not carry forward an equivalent "defined once, imported everywhere, never a duplicated literal" rule.

Given AD-18's constraint that `jax_supervised_training` never reimplements `chess_target_encoding.py` (relocated to `chess_ai`) and treats `move_tokens`/`policy` as opaque integers, there is no single shared module on the `jax_supervised_training` side that both `ChessMoveTokenDataset` (which must prepend BOS and choose the PAD fill value for `padded_batch`) and `chess_move_token_transformer` (which must derive its pad mask by comparing token ids to PAD — see Finding 1b) are pointed at. Two builders can each hardcode `4672`/`4673` (or independently derive them as `config["num_classes"]` / `config["num_classes"] + 1`) and stay individually AD-29-compliant while having no shared source of truth — a future change to `NUM_MOVES` upstream (chess_ai) would silently desync one side from the other, exactly the failure mode AD-22 was written to prevent for the sibling constant.

**Fix direction:** add a sentence to AD-29 naming where `BOS_TOKEN_ID`/`PAD_TOKEN_ID` are defined once (e.g., derived from `config["num_classes"]` in `dataset_configs.py`'s `CHESS_MOVE_TOKEN` entry, or as named constants in `data_management.py` imported by `model_library.py`) and forwarded via `model_kwargs`/config rather than each side hardcoding `4672`/`4673`.

---

## Finding 5 (Low-Medium) — Structural Seed names only the strategy-dispatch edit point in `main.py`, not the model_kwargs forwarding edit point AD-30's seed hyperparameters actually need

`main.py`'s existing pattern (verified, lines ~118-131) shows every model-specific hyperparameter (`heatmap_prior`, `num_bottleneck_tokens`, `token_dim`) requires its own explicit conditional block —

```python
if "num_bottleneck_tokens" in config:
    model_kwargs["num_bottleneck_tokens"] = config["num_bottleneck_tokens"]
```

— separate from, and upstream of, the `task_type == "..."` strategy dispatch branch. AD-30 explicitly calls the transformer's hyperparameters (block count, `d_model`, heads, dropout) a "seed... ajustée empiriquement" living in `dataset_configs.py`/`CHESS_MOVE_TOKEN`, and its `Binds:` line does name `model_kwargs (main.py)`. But the spine's **Structural Seed** section only lists `main.py # + branche task_type == "chess_move_token" (main.py:141-200)` — the strategy-dispatch location — with no mention of the separate, earlier `model_kwargs` conditional-forwarding block.

A builder who implements the strategy branch exactly where the Structural Seed points, and reads `dataset_configs.py`'s new hyperparameters as "the model already gets these because they're in the config," has no signal that a second, separate edit to `main.py` is required. The failure mode is silent: `get_model()` receives no `num_layers`/`d_model`/etc. kwargs, the model dataclass falls back to its own defaults, training runs to completion, no error — and the "seed hyperparameters" in `dataset_configs.py` are simply never read. Two builders (one only touching the seed-named location, one who happens to notice the existing `model_kwargs` pattern) diverge on whether config-specified hyperparameters have any effect at all.

**Fix direction:** extend the Structural Seed's `main.py` line to also name the `model_kwargs` conditional-forwarding block as a required edit point for AD-30's hyperparameters, mirroring the existing `heatmap_prior`/`num_bottleneck_tokens`/`token_dim` precedent.

---

## Non-findings (checked, no contradiction with inherited ADs)

- AD-3 (checkpoint fallback + `batch_stats` reinit): spine restates it unchanged, no touch to the loading path described by AD-30's Structural Seed. No conflict found.
- AD-14 (modular training, never merged): `chess_move_token` is a fully separate config/strategy/loader per the Structural Seed; no shared training graph proposed. No conflict.
- AD-17 (dedicated class + single dispatch literal, no conditional branch): verified against current code — `model_library.py`'s `MODELS` dict (not `if/elif`) at `model_library.py:1121-1131`, and the spine's own AD-17 restatement correctly reflects this (dict, not `if/elif`, as it explicitly flags being a correction to the parent's text). Consistent.
- AD-18 (single producer/consumer format module, no reimplementation): spine correctly defers to `chess_ai`'s relocated `chess_target_encoding.py` and treats tokens as opaque ints; consistent with the split (2026-08-09). No conflict — though see Finding 4 for a related but distinct gap (no *jax_supervised_training*-side single constant for BOS/PAD, which AD-18 doesn't cover since those tokens don't exist in `chess_target_encoding.py` at all).
- AD-21 (non-regression of existing chess domains + JAX_DETECTOR): the new dispatch branches are additive (`elif task_type == "chess_move_token"` pattern, `MODELS` dict entry, new loader class) mirroring exactly the pattern already used three times successfully (`chess_policy_value`, `chess_legal_moves`, plus `JAX_DETECTOR`'s own `AD-17` precedent) without touching existing branches' code. Low residual risk here relative to the findings above; no concrete incompatibility found in the dispatch wiring itself.
- AD-22 (fixed policy output space, no illegal-move masking): AD-26 explicitly restates this unchanged (`Dense(4672)` strict, no BOS/PAD in output space per AD-29). Consistent, not weakened.

## Recommended action

Do not promote this spine to `status: final` until Findings 1, 1b, and 2 are closed with concrete Rule text (not just Binds-list mentions) — they are the ones where "two spine-compliant builders" verifiably produce incompatible or silently-wrong artifacts, one of which was reproduced directly against the actual runtime (`tf.data`, `numpy.float16`) rather than inferred from reading. Findings 3-5 should be closed before implementation starts, but are lower blast-radius (data-split reproducibility, constant duplication, config-forwarding oversight) than the dtype/mask findings.
