# Rubric Walker Review — AD-21 amendment to ARCHITECTURE-SPINE.md (JAX Single-Pass)

Reviewed: `architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md`
Ground truth: `.memlog.md` (last ~10 entries, "Resumed 2026-08-19..." onward)
Cross-check: `architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md` (AD-1, AD-3)
Also inspected: `main.py` lines 1-290 (the actual code AD-21 describes and will replace)

## Verdict

AD-21 is well-grounded, correctly traceable to the memlog, and does not structurally
contradict the inherited AD-1/AD-3 (sibling spine) or AD-17 (task_type single dispatch
point). The AD-15 correction is directionally accurate. However, the amendment misses
one concrete, checkable divergence point it should have caught (error-handling
behavior when the Strategy dispatch becomes a dict), leaves the `overrides` scoping
vs. AD-3's binding ambiguous, silently assumes default-value equivalence for several
Strategy kwargs that AD-21 never states, and contains a minor factual miscount plus a
wording tension in the corrected AD-15. None of these are fatal to the amendment, but
each is the kind of thing that should be nailed down before or during implementation
rather than discovered mid-story.

## Findings

### 1. [Medium] STRATEGIES dict dispatch drops the current explicit `ValueError` on unrecognized `task_type` — not addressed by AD-21

`main.py:274-275` today ends the if/elif chain with:
```python
else:
    raise ValueError(f"task_type '{task_type}' non reconnu.")
```
AD-21's Rule says the if/elif "à 8 branches" becomes `STRATEGIES = {task_type: Classe}`,
but neither the AD-21 text nor the memlog entry that motivated it (`.memlog.md` line 65)
says anything about how the dict-based dispatch preserves this explicit, helpful error.
A naive `STRATEGIES[task_type]` raises a raw `KeyError` instead — a different exception
type with a worse message, and anything that ever wraps this call expecting `ValueError`
(none today, but that's exactly the kind of latent behavior change AD-21 exists to
prevent) would silently stop working.

This is doubly notable because **the very same file, a few lines above**, already solved
this exact class of problem for the model-lookup path: `main.py:179-184` explicitly uses
`MODELS.get(model_name)` instead of `MODELS[model_name]` specifically so an invalid
`model_name` doesn't raise a raw `KeyError` here but instead falls through to
`get_model()`'s own explicit `ValueError`. AD-21 doesn't extend that same discipline to
the new `STRATEGIES` dict it introduces. This is a real "divergence point for the level
below" that AD-21's own Prevents clause (silent field/behavior drift) should have caught
but didn't.

**Recommendation:** either add a line to AD-21's Rule requiring `STRATEGIES.get(task_type)`
+ an explicit `ValueError` fallback (mirroring the `MODELS.get()` precedent), or explicitly
defer it with the same "single function, no cross-unit divergence" reasoning already used
for the missing-required-param Deferred item.

### 2. [Medium] `overrides` scoping vs. sibling AD-3's binding is left ambiguous — risk of silently widening compute_dtype injection

AD-21's Rule states `compute_dtype devient une entrée de overrides comme les autres
valeurs calculées côté main.py (dropout_rate, num_classes)` and that `main.py construit
model_kwargs et les kwargs de chaque Strategy via cet unique helper`. It never states
whether the `overrides` dict passed to the two `build_kwargs_from_config` call sites
(model factory vs. Strategy class) is the same object or built separately per call.

The sibling spine's AD-3 explicitly scopes compute_dtype injection to `"point d'injection
main.py → get_model"` only — not Strategy construction. If a future builder reuses one
shared `overrides` dict (a very natural refactor shortcut given `num_classes` is needed
by both call sites — `ClassificationStrategy`/`KeplerStrategy` both take `num_classes`),
and a future Strategy class ever declares a parameter literally named `compute_dtype`,
it would silently start receiving the injection — outside AD-3's stated binding, with
nothing in AD-21 flagging this as out of bounds. This is exactly the kind of subtle,
non-obviously-wrong drift AD-21 exists to prevent, and it's the one place where AD-21
touches AD-3's territory without pinning the boundary down.

**Recommendation:** AD-21 should state explicitly that `compute_dtype` is only ever
included in the `overrides` passed to the model-factory call, never to the Strategy-kwargs
call (or, if it is intentionally shared, say so and confirm no current/near-term Strategy
declares that parameter name).

### 3. [Low] Behavior-preservation gap for Strategy kwargs default equivalence — not covered by AD-21 or the Deferred note

Several Strategy constructor calls today receive values pre-extracted from `config` with
main.py-local fallback defaults, e.g. (`main.py:200-267`):
- `label_smoothing=config.get("label_smoothing", 0.0)`
- `mixup_alpha=config.get("mixup_alpha", 0.0)`
- `loss_method=config.get("loss_method", "cross_entropy")`
- `metric_threshold=config.get("metric_threshold", 0.5)`

Under `build_kwargs_from_config`'s stated rule ("value from overrides, else config"), a
parameter absent from both simply won't be forwarded — meaning the target constructor's
*own* Python-level default takes over. This only reproduces today's behavior if each
Strategy class's own default for these parameters exactly matches the value currently
hardcoded in `main.py`'s `config.get(key, default)` calls. AD-21 never states this
equivalence requirement, even though the existing Deferred section already tracks the
adjacent case ("comportement face à un paramètre requis absent") — the same reasoning
("détail d'implémentation laissé à la story, aucun risque de divergence inter-unités")
would apply here too, but this specific risk isn't named anywhere, so a story
implementer has no explicit prompt to verify it.

**Recommendation:** either fold this into the existing Deferred item's scope explicitly,
or add one sentence to it.

### 4. [Low] Minor factual miscount: "if/elif à 8 branches"

AD-21's Rule and `.memlog.md` (line 65) both describe the current Strategy dispatch as
"if/elif à 8 branches." Actual count in `main.py:198-273` is **9** task_type branches
(classification, detection, kepler, detection_centernet, chess_policy_value,
chess_legal_moves, chess_move_token, chess_token, chess_token_1_move). Doesn't affect
the validity of the fix (a dict works the same regardless of branch count), but it's a
checkable inaccuracy in the architecture record that should be corrected at next distill.

### 5. [Low] AD-15 correction: minor wording tension, not a real contradiction

The corrected AD-15 text reads: `"...portent des valeurs non-nulles dérivées du fond de
l'image (ni NaN, ni -1, ni zéro garanti — corrigé...)"`, then later: `"...y compris le
cas où elle vaudrait zéro par coïncidence."` The first clause asserts the values *are*
non-null/non-zero; the later clause explicitly allows for a coincidental zero. Read
carefully these aren't strictly contradictory (the second is arguably a general
"don't rely on any field's value, even if it happens to be 0" warning that predates
this specific correction), but on a first read it's confusing right where precision
matters most. The memlog's own ground-truth phrasing ("invalid slots actually carry
non-zero, background-derived values — not NaN, not -1 either") is cleaner and doesn't
carry this tension. Otherwise the correction is accurate: it correctly overturns the
old "zero-filled" claim, correctly cites Story 8.6 as the source, and correctly keeps
`valid_mask` as sole authority — no new *substantive* ambiguity, just wording that could
be tightened.

## Checklist-by-checklist summary

- **Fixes real divergence points, misses none:** Mostly yes for `model_kwargs`
  (matches the literal `if "X" in config` pattern in the real code closely, including
  correctly routing `dropout_rate`/`num_classes` through `overrides` since they aren't
  literal top-level config keys). Misses the `STRATEGIES` dict error-handling
  divergence (Finding 1) and the Strategy-kwargs default-equivalence risk (Finding 3).
- **Rule enforceable / prevents its stated scenarios:** Yes, mechanically — single
  helper function, signature-driven, no `**kwargs` catch-all — this is a testable,
  code-reviewable design.
- **Deferred doesn't let two units diverge:** Correct as scoped — the one Deferred
  item added (required-param-absent behavior) is legitimately single-function,
  no cross-unit risk. (See Finding 3 for an adjacent risk that arguably belongs here too.)
- **Doesn't weaken/contradict sibling AD-1/AD-3:** No outright contradiction. AD-1
  (never read compute_dtype from config) and AD-3 (named-parameter introspection only,
  no catch-all) are both correctly generalized rather than violated. Finding 2 flags an
  ambiguity in scope, not a contradiction.
- **Consistent with AD-17:** Yes — the dict is explicitly justified as a valid
  implementation of "single dispatch point per file," doesn't touch task_strategies.py/
  data_management.py's own dispatch points, doesn't introduce a second literal.
- **AD-15 correction accurate, no new ambiguity:** Accurate; minor wording tension
  only (Finding 5), not a functional ambiguity.
- **Placeholder/TODO/internal inconsistency:** None found. Structural Seed and Deferred
  references to AD-21 match AD-21's own text (helper location in `model_library.py`,
  testable-in-isolation framing, `STRATEGIES` dict, generic print) — no drift between
  sections.
