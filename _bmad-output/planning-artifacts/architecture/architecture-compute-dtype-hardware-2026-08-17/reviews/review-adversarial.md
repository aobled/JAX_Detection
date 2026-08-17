# Adversarial Review — compute-dtype-hardware Architecture Spine

**Reviewer lens:** adversarial (Reviewer Gate)
**Target:** `_bmad-output/planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md`
**Method:** construct concrete scenarios where two units one level down each obey every AD to the letter yet build incompatibly.

## Overall verdict

**Not ready to adopt as-is.** Two of the six findings below are not hypothetical edge cases but *guaranteed* collisions grounded in code/tests that exist in the repo right now (F1) or in the spine's own cited source document (F2). Both need an explicit AD amendment — not just a note — before implementation starts. The remaining four are genuine future-incompatibility holes that a tightened AD closes cheaply now, while it's still architecture and not four call sites of drift.

---

## F1 — [CRITICAL, code-verified] AD-2's literal instruction breaks an existing regression test that AD-7 promises not to break

AD-2's Rule says: *"La résolution/validation interne à `create_chess_move_token_transformer` (`model_library.py:1218-1221`) est retirée au profit de ce point unique."*

That block is exactly this (verified in the current file):

```python
resolved_dtype = getattr(jnp, compute_dtype, None)
if resolved_dtype is None:
    raise ValueError(
        f"compute_dtype='{compute_dtype}' inconnu de jax.numpy - attendu 'float32'/'bfloat16'/'float16'"
    )
```

And `tests/test_chess_move_token_model.py:257-265` has a test that depends on exactly this block:

```python
def test_compute_dtype_factory_rejects_unknown_string():
    try:
        create_chess_move_token_transformer(num_classes=NUM_MOVES, compute_dtype="not_a_real_dtype")
        assert False, "attendu ValueError sur un compute_dtype invalide"
    except ValueError as e:
        print(f"OK - compute_dtype invalide rejete explicitement : {e}")
```

If a developer implements AD-2 literally — deletes the getattr/raise block, leaves the factory as a thin passthrough — `create_chess_move_token_transformer(compute_dtype="not_a_real_dtype")` no longer raises `ValueError` at all (Flax `nn.Module` construction doesn't validate field values; the bad string just gets stored). The test's `assert False` fires, and the suite goes red.

AD-7 says: *"CHESS_MOVE_TOKEN continue de recevoir `compute_dtype=bfloat16` sur TPU exactement comme avant... Même garantie que le précédent."* This is scoped to the success path only — it never says what happens to the existing *failure-path* contract test. So two equally-compliant developers diverge:

- **Dev A** follows AD-2 to the letter (deletes the block) → breaks the pinned regression test, silently, because nothing in the spine tells them the test exists or what to do with it.
- **Dev B** reads AD-7's "no regression" broadly, keeps the validation block to keep the test green → violates AD-2's "un seul point de vérité" / duplicate-resolution ban.

Both are defensible readings of the spine as written. Neither is flagged as wrong by any AD.

**Fix:** AD-2 (or a new AD) must explicitly dispose of this test — e.g. "the string-rejection test is rewritten to assert against `main.py`'s central resolver, not the factory" or "the factory keeps a `try/except`-free passthrough and this specific test is deleted as part of this migration, superseded by a resolver-level test." Silence here is not neutral; it's a coin flip between breaking CI and violating AD-2.

---

## F2 — [CRITICAL, doc-verified] The spine's adopted paradigm contradicts its own cited SPEC's explicit, dated decision

The spine's `sources` frontmatter cites `_bmad-output/specs/spec-compute-dtype-hardware/SPEC.md`. That SPEC contains a dated decision (`.memlog.md`, same day, 2026-08-17, "OQ1"):

> *"Résolution OQ1 (Aymeric, 2026-08-17) : forwarding UNCONDITIONNEL, pas conditionnel/registre. Si backend détecté == TPU, compute_dtype=bfloat16 s'applique quel que soit le modèle — chaque classe modèle doit structurellement déclarer/consommer le champ (pas un opt-in par introspection)."*

SPEC.md itself (line 29) is even more explicit: *"chaque classe modèle touchée doit structurellement déclarer/consommer le champ pour éviter un `TypeError`... — pas d'opt-in par registre/introspection."*

The spine does the opposite. Its Design Paradigm is literally *"Progressive Enhancement via Centralized, **Capability-Probed Injection**"* and AD-3 is introspection-based opt-in: *"N'injecte que si ce paramètre nommé existe... Un modèle qui ne déclare pas le champ n'est simplement pas concerné."*

This is not a subtlety — it's the spine's central paradigm choice directly reversing a decision its own source document recorded on the same date, with no AD or note acknowledging the override or explaining why Winston chose differently from Aymeric's recorded OQ1 answer.

**Concrete incompatible pair:** a future implementer picking up the Deferred "rollout to 10 other models" item, reading SPEC.md's dated, explicit "quel que soit le modèle... pas d'opt-in par introspection," will reasonably conclude they must add `compute_dtype` fields to *every* remaining `MODELS` entry to honor the unconditional contract. Another implementer trusting AD-3 will conclude the opposite — untouched models are fine by design, no changes needed. These are flatly incompatible obligations for the same task, both textually "compliant" with a live governance artifact.

**Fix:** the spine needs an explicit line (ideally inside AD-3's Rule, or a new AD) stating that AD-3 supersedes SPEC OQ1's "unconditional, no introspection" resolution, with the one-line rationale (avoiding a hand-maintained registry / avoiding forced edits to unrelated models). Without that, SPEC.md remains a live, dated, signed-off contradiction sitting right next to the spine.

---

## F3 — [HIGH] The "unconditional, no opt-in" SPEC framing is a ready-made loophole back into AD-1's forbidden per-dataset flag

Because SPEC.md frames the target trajectory as unconditional-regardless-of-model, and because AD-1 forbids any per-dataset `compute_dtype` config, a future implementer who needs a *specific* model to opt out of `bfloat16` on TPU for numerical-stability reasons has, by the SPEC's own framing, no sanctioned lever: AD-3's introspection escape hatch is exactly what SPEC OQ1 rejected, so a developer trying to honor "unconditional" in good faith is pushed toward reinventing a per-dataset override (e.g. `config["compute_dtype_override"] = "float32"`) as the only place left to express "this model needs an exception" — reproducing verbatim the anti-pattern AD-1 exists to prevent (a string codegen'd by dataset config instead of derived from hardware), just reframed as a legitimate "opt-out," not the old "opt-in."

**Fix:** add an explicit line — most naturally to AD-1 or AD-3 — stating that the *only* sanctioned way for a model to be exempt from hardware-derived `compute_dtype` is to not declare the named parameter at all (AD-3's mechanism); a per-dataset override flag is forbidden even when framed as an "exception," not just as the general case.

---

## F4 — [HIGH] AD-3's introspection proves the parameter exists, not that the factory does anything correct with it

AD-3's own "Prevents (c)" language holds up named-parameter presence as safer than a `**kwargs` catch-all, citing today's real bug: `aircraft_detector_unet`/`centernet`/`centernet_lite` accept `**kwargs` but never forward it to the constructor. But a *named* parameter can suffer the exact same fate — nothing about `def create_x(..., compute_dtype=jnp.float32, **kwargs)` forces the factory to actually pass `compute_dtype=compute_dtype` into `SomeModel(...)`.

**Concrete pair:** Dev A (correct) adds `compute_dtype` to a new factory's signature *and* forwards it into the underlying `nn.Module` and every submodule call site, matching AD-4/AD-5. Dev B adds `compute_dtype` to the factory's signature only — satisfying AD-3's introspection test byte-for-byte — but forgets to actually pass it to the `SomeModel(...)` constructor call, structurally identical to today's `create_aircraft_detector_unet(dropout_rate=0.2, **kwargs): return AircraftDetectorUNet(dropout_rate=dropout_rate)` drop bug, just one abstraction layer deeper (now it's a named kwarg being dropped, not a `**kwargs` bag). `main.py` injects it silently, no error is raised, logs look identical to the success case, and the model computes in `float32` on TPU regardless — reproducing the exact "config claimed bfloat16 without checking backend" failure mode AD-1 exists to prevent, now happening one hop later and even harder to notice because AD-3's own text implies named-parameter presence is a stronger guarantee than it is.

**Fix:** AD-3 (or a companion AD) should require a cheap runtime or test-level contract check per adapted model — e.g. the same pattern already used in `tests/test_chess_move_token_model.py:207` (`test_compute_dtype_bfloat16_params_and_output_stay_float32`) — asserting the instantiated model's params/behavior actually differ under `compute_dtype=bfloat16` vs `float32`, not just that the factory call didn't raise `TypeError`.

---

## F5 — [MEDIUM] AD-5's shared submodules: partial propagation inside a future parent is invisible and unforbidden

`SeparableConv`/`SEBlock`/`SpatialAttention` are already shared by `SophisticatedCNN128Plus` (~9 call sites, out of scope, Deferred) and `AircraftDetectorCenterNet`/`CenterNetLite` (~14 call sites). AD-5 giving these submodules a *defaulted* `compute_dtype: Any = jnp.float32` field is safely backward-compatible for those today (default preserves current behavior, confirmed no crash for existing callers).

The forward risk: when `SophisticatedCNN128Plus` is eventually adapted (Deferred item), AD-4's "sans exception" rule is textually scoped to `SophisticatedCNN32Plus`/`SophisticatedCNN128Lite` only (AD-4's own Binds line) and talks about direct `nn.Conv`/`nn.Dense`/`nn.BatchNorm`/`nn.LayerNorm` calls — a `SeparableConv(96, (3,3))(x, training)` call is not literally one of those. A developer could add `compute_dtype` to `SophisticatedCNN128Plus`'s own class and to its bare `nn.Conv`/`nn.BatchNorm` calls, while missing 2 of 9 `SeparableConv(...)` call sites (easy — they're visually identical to the compliant ones since AD-5 mandates *explicit*, not inherited, passing). Result: a model that silently mixes bfloat16 and float32 sub-networks, with no error, no log difference, and no AD violated on its face — worse than not adapting the model at all, since it looks fully compliant.

**Fix:** generalize AD-4's "sans exception" invariant beyond its current two Binds targets into a durable rule for *any* future class gaining `compute_dtype`: every call site of a `compute_dtype`-aware submodule counts as a site requiring explicit forwarding, not just direct layer calls — and back it with something checkable (a grep-based lint, or a pytree-homogeneity test asserting no stray `float32` params survive under a `bfloat16` construction) rather than prose review alone.

---

## F6 — [MEDIUM] Nothing pins the type of a factory's *own* default value when main.py's injection doesn't apply

Today's precedent (`create_chess_move_token_transformer(..., compute_dtype="float32", ...)`) defaults to a **string**. AD-2 mandates removing the internal string-resolution logic but the spine never states what the factory's own default should become. Two developers, each adding `compute_dtype` to a different future factory, could pick incompatible conventions — one copies the pattern actually still visible in git history/tests (`compute_dtype="float32"`, a string default), the other follows AD-5's submodule convention (`compute_dtype: Any = jnp.float32`, a real dtype default). Both pass AD-3's naming-only introspection test. Both work identically when called *through* `main.py` (which always sends a resolved `jnp.dtype`, masking the divergence). They diverge the moment either factory is instantiated directly — tests, notebooks, or any other call site that bypasses `get_model` — and a string flowing into `nn.Dense(dtype="float32")` may or may not behave identically to `jnp.float32` depending on downstream identity/equality checks (`self.compute_dtype is jnp.bfloat16`-style code, plausible given the existing `ChessMoveTokenTransformer` docstring's emphasis on precise dtype semantics).

**Fix:** add one line to the Consistency Conventions table: the parameter's own default value, used whenever main.py's injection does not apply, must itself already be a resolved `jnp.dtype` object (e.g. `jnp.float32`), never a string — closing this at the naming-convention level rather than leaving it to be inferred from AD-2's phrasing.

---

## Summary table

| # | Severity | Type | One-line |
| --- | --- | --- | --- |
| F1 | Critical | code-verified regression | AD-2's literal removal instruction breaks `tests/test_chess_move_token_model.py::test_compute_dtype_factory_rejects_unknown_string`, contradicting AD-7. |
| F2 | Critical | spec-vs-spine contradiction | Spine's introspection-opt-in paradigm (AD-3) directly reverses SPEC.md's dated, explicit "unconditional, no opt-in by introspection" decision, unacknowledged. |
| F3 | High | loophole | SPEC's "unconditional" framing gives a well-intentioned future dev a path back to a per-dataset override flag, reintroducing what AD-1 forbids. |
| F4 | High | forwarding gap | AD-3 tests parameter *name* only; a named-but-dropped `compute_dtype` reproduces the aircraft_detector kwargs-drop bug one layer deeper, silently. |
| F5 | Medium | fan-out gap | AD-4's "sans exception" doesn't generalize to submodule call sites in future parents (e.g. `SophisticatedCNN128Plus`), enabling silent partial mixed-precision. |
| F6 | Medium | default-type ambiguity | No AD pins whether a factory's own `compute_dtype` default (used outside `main.py`'s injection path) must be a `jnp.dtype` or may remain a string. |
