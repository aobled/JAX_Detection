# Review Lens: Web/Reality Verification — AD-21 (build_kwargs_from_config)

**Target:** `ARCHITECTURE-SPINE.md`, AD-21 amendment (2026-08-19)
**Lens:** Verify every committed decision was web-researched or reality-checked rather than asserted from training data — current library/framework versions, that each named technology still exists and fits, and (greenfield) live starter defaults. Flag anything out of date and unconfirmed against the web, the existing project, or the current starter.

## Scope note

AD-21 introduces **no new external dependency, no library, no version claim**. It is a pure Python-stdlib refactor (`inspect.signature()` + dict dispatch) applied to code already present in this repo. There is therefore nothing here that requires web research — no framework version to check, no "does this library still exist" question, no starter defaults to confirm. This review lens is satisfied by a **reality check against the actual codebase** instead, which is what follows.

## Verification performed

1. Read `main.py:120-276` (the `model_kwargs` construction and the `task_type` if/elif dispatch AD-21 claims to replace).
2. Ran `grep -n "def __init__" task_strategies.py` and read the matched constructors.
3. Ran a live Python check of `inspect.signature()` parameter-kind classification (`POSITIONAL_OR_KEYWORD` / `KEYWORD_ONLY` vs `VAR_KEYWORD`) to confirm the stdlib behavior AD-21's design depends on.
4. Read the `MODELS` factory functions in `model_library.py` that the conditional branches in `main.py` currently feed (`heatmap_prior`, `num_bottleneck_tokens`, `token_dim`, `num_layers`, `d_model`, `num_heads`, `num_trunk_layers`).

## Findings

### 1. `inspect.signature()` parameter-kind claim — CONFIRMED correct (no flag)

AD-21's rule depends on being able to tell an explicitly named parameter apart from a `**kwargs` catch-all. Live-checked:

```python
def target(a, b=1, *, c=2, **kwargs): pass
# a      -> POSITIONAL_OR_KEYWORD
# b      -> POSITIONAL_OR_KEYWORD
# c      -> KEYWORD_ONLY
# kwargs -> VAR_KEYWORD
```

`inspect.Parameter.kind` does distinguish `VAR_KEYWORD` from `POSITIONAL_OR_KEYWORD`/`KEYWORD_ONLY` as claimed. This is long-stable stdlib behavior (`inspect` module, unchanged for many Python major versions) — nothing here is version-sensitive or plausibly out of date. No web verification needed or warranted.

### 2. Repetitive-pattern description in `main.py` — CONFIRMED accurate

`main.py:135-188` does contain the described chain of `if "X" in config: model_kwargs["X"] = config["X"]` branches (7 of them: `heatmap_prior`, `num_bottleneck_tokens`, `token_dim`, `num_layers`, `d_model`, `num_heads`, `num_trunk_layers`), followed by the AD-3-style introspection-based `compute_dtype` injection at lines 184-188. `main.py:198-275` does contain a 9-way `if/elif` (`classification`, `detection`, `kepler`, `detection_centernet`, `chess_policy_value`, `chess_legal_moves`, `chess_move_token`, `chess_token`, `chess_token_1_move`) constructing each `Strategy` with its own hand-picked kwarg subset. `task_strategies.py` constructors (via `grep -n "def __init__"`) confirm none of the 9 `Strategy.__init__` signatures accept `**kwargs` — all params are explicitly named. So for the **Strategy side**, AD-21's design (introspect `target`, forward only named params) is factually sound and matches reality with no gaps.

### 3. **HIGH — model-factory side: AD-21's own anti-pattern is already present in 6 of the 7 branches it proposes to replace**

This is the one substantive reality-check finding, and it's significant enough to warrant amendment before implementation.

AD-21's rule states the helper "ne forwarde que les paramètres **nommés explicitement**... jamais un `**kwargs` catch-all" — explicitly modeled on AD-3 (compute_dtype), where this works because `create_aircraft_detector_unet`, `create_aircraft_detector_centernet`, and `create_kepler_1d_cnn` all declare `compute_dtype` as an explicit named parameter.

But reading the other `MODELS` factories that the *other* 6 conditional branches in `main.py` currently feed shows they do **not** declare those parameters by name — they only accept them through a `**kwargs` catch-all that is forwarded, un-renamed, straight into the underlying `nn.Module`:

- `create_chess_cnn_attention_policy_value(num_classes, dropout_rate=0.1, **kwargs)` (model_library.py:925) — docstring literally says: *"`**kwargs` transmis tel quel à `ChessCnnAttentionPolicyValue` - permet de surcharger les hyperparamètres 'ajustables' documentés sur la classe (`token_dim`, `num_bottleneck_tokens`, `num_heads`)"*.
- `create_chess_cnn_attention_legal_moves(num_classes, dropout_rate=0.1, **kwargs)` (model_library.py:1028) — same pattern; the class itself (`ChessCnnAttentionLegalMoves`) declares `token_dim`, `num_bottleneck_tokens`, `num_heads` as dataclass fields, but the **factory** does not.
- `create_chess_move_token_transformer(num_classes, dropout_rate=0.1, compute_dtype="float32", **kwargs)` (model_library.py:1168) — `num_layers`, `d_model`, `num_heads` reach the model only via `**kwargs`.
- `create_chess_token_candidate_model(num_classes, dropout_rate=0.1, **kwargs)` (model_library.py:1423) — docstring: *"`**kwargs` transmis tel quel... permet de surcharger `token_dim`/`num_bottleneck_tokens`/`num_heads`/`num_trunk_layers`"*.
- `create_chess_token_one_move_model(num_classes=None, dropout_rate=0.1, **kwargs)` (model_library.py:1659) — same pattern.

Only `heatmap_prior` (on `create_aircraft_detector_centernet`) and `compute_dtype` are explicit named parameters on their factories; the remaining five config keys currently forwarded by `main.py`'s conditional branches — `num_bottleneck_tokens`, `token_dim`, `num_layers`, `d_model`, `num_heads`, `num_trunk_layers` — are **only** reachable today because `main.py` unconditionally stuffs them into `model_kwargs` whenever the config key is present, relying on each factory's own `**kwargs` to relay them to the `nn.Module`.

If `build_kwargs_from_config(target, config, **overrides)` is implemented literally as AD-21 specifies — introspecting `target` = the `MODELS` factory function and refusing to use its `**kwargs` — it will **silently stop forwarding all six of those parameters** the moment `main.py` switches to the helper, because none of them are named parameters of the factory functions themselves. This is not a hypothetical edge case: it is the exact chess-spike hyperparameter-sweep mechanism (`num_bottleneck_tokens`/`token_dim`/`num_heads` bottleneck search, `num_trunk_layers` for `chess_token_candidate_model`, `num_layers`/`d_model` for `chess_move_token_transformer`) that recent commit history and MEMORY show has been actively used (`chess_bottleneck_genetic_poc`, `chess_token_candidate_model_spec`).

The renaming the factories perform (`num_classes` → `num_moves`/`num_candidates`) is also why the introspection target can't simply be swapped to the underlying `nn.Module` class instead — the factory is where the config-key-to-constructor-param mapping actually happens, and it's exactly the layer that currently doesn't name these six params.

AD-21's own "Prevents" list calls out this exact failure mode ("le piège d'absorption silencieuse par un `**kwargs` catch-all") but does not notice that eliminating the conditional-branch escape hatch removes the only thing currently making these six parameters work. Nothing in the spine (Rule text, Structural Seed, or elsewhere) acknowledges this gap or specifies a resolution (e.g., adding the missing named parameters to the five factories as a prerequisite, or scoping AD-21 to exclude these factories until they're updated).

**Recommendation:** amend AD-21's Rule or add a prerequisite/Deferred note requiring the five `**kwargs`-only factories (`create_chess_cnn_attention_policy_value`, `create_chess_cnn_attention_legal_moves`, `create_chess_move_token_transformer`, `create_chess_token_candidate_model`, `create_chess_token_one_move_model`) to gain explicit named parameters for their currently-`**kwargs`-only tunables *before* (or as part of) the `main.py` cutover to `build_kwargs_from_config` — otherwise the migration is a functional regression for every chess spike config that sets these fields.

### 4. Minor — `dropout_rate`/`num_classes` as `overrides` — consistent with reality, not a flag

AD-21's Rule text states `dropout_rate`/`num_classes` are `overrides`, not read from `config`. Verified: `dropout_rate = backend_config["dropout_rate"]` (main.py:129) comes from a nested per-backend sub-dict, not the flat `config` the helper would introspect against — so it genuinely can't be found by a flat `config` lookup and must be an override. `num_classes` is technically present at the flat `config["num_classes"]` (main.py:84) so listing it as an override too is redundant but not incorrect (override value equals the config value either way). No action needed.

### 5. Stack section — CONFIRMED accurate

"Aucune nouvelle dépendance externe introduite" is correct for AD-21: `inspect` is Python stdlib, no new import beyond what's already used elsewhere in the file for the `compute_dtype` mechanism (`main.py:172-188` already imports and uses `inspect.signature`).

## Verdict

Everything that is genuinely new-technology/version territory in AD-21 is non-existent (correctly — this is a pure stdlib refactor, nothing to web-verify). The `inspect.signature()` mechanics the design leans on are confirmed correct against a live interpreter check. However, the reality check against the actual codebase surfaces one concrete, high-severity gap: AD-21's chosen introspection target (the `MODELS` factory function) does not, for 5 of the 6 non-`compute_dtype`/`heatmap_prior` config keys currently forwarded by `main.py`'s conditional branches, declare those parameters by name — they only exist behind each factory's own `**kwargs`, which AD-21 explicitly forbids using. As written, adopting AD-21 would silently drop `num_bottleneck_tokens`, `token_dim`, `num_heads`, `num_layers`, `d_model`, and `num_trunk_layers` forwarding for the chess-spike models. This should be resolved (factory signatures updated, or AD-21 scoped/sequenced accordingly) before implementation.
