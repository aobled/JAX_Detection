# Rubric Walk — architecture-compute-dtype-hardware-2026-08-17

Reviewer: rubric walker (BMad architecture-spine Reviewer Gate)
Spine: `_bmad-output/planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md`
Driving spec: `_bmad-output/specs/spec-compute-dtype-hardware/SPEC.md`
Code checked against: `main.py` (hardware detection L31-41, model_kwargs forwarding L117-160), `model_library.py` (SeparableConv/SEBlock/SpatialAttention L18-83, SophisticatedCNN128Lite L212-333, SophisticatedCNN32Plus L335-446, ChessMoveTokenTransformer L1075-1226, MODELS dict + get_model L1751-1786, all 12 `create_*` factory signatures), `dataset_configs.py` L953-966 (CHESS_MOVE_TOKEN compute_dtype literal), CIFAR10 (L360) / FIGHTERJET_CLASSIFICATION (L111) configs.

## Overall verdict

**Not pass — one critical (blocking) contradiction, plus moderate/minor issues.** The spine is well-grounded in the real codebase (line references check out, the AD-2/AD-7 non-regression claims are accurate, the `**kwargs`-absorption trap cited in AD-3 is real) and its Deferred section correctly protects the operational envelope and the 9 untouched models. But its central mechanism (AD-3, and the paradigm itself — "Capability-Probed Injection") directly contradicts an explicit, dated decision in the governing SPEC, without acknowledging or justifying the deviation anywhere in the document. That alone should send this back for a decision (either the spine is wrong, or the spec's Decision needs to be reopened) before this ships to implementation.

## Critical finding

### F1 — AD-3 (introspection-based conditional injection) directly contradicts SPEC's explicit "unconditional forwarding, no introspection" decision

- SPEC.md Constraints (line 29): *"Le forwarding est UNCONDITIONNEL : si le backend détecté est TPU, `compute_dtype=bfloat16` s'applique quel que soit le modèle... — pas d'opt-in par registre/introspection."*
- SPEC.md Decisions, resolved by Aymeric on 2026-08-17 (line 49): *"Forwarding unconditionnel confirmé (pas de registre/introspection) — voir Constraints."*
- ARCHITECTURE-SPINE.md AD-3 (rule): *"main.py inspecte la signature de la factory cible et vérifie la présence d'un paramètre nommé explicitement `compute_dtype`... N'injecte que si ce paramètre nommé existe."* This is precisely opt-in-via-introspection — the mechanism the spec names and rejects in the same sentence. The spine's own paradigm line even calls this "Capability-Probed Injection," i.e. it's not an accidental slip, it's the spine's central design choice.
- Both CAP-1 and CAP-2 in the Capability→Architecture Map are governed partly by AD-3, so this isn't a peripheral rule — it's load-bearing for both capabilities the spec cares about.
- Practically, the spine's choice is arguably the *safer* engineering call: literal unconditional forwarding (always adding `compute_dtype` to `model_kwargs` regardless of `model_name`) would raise a `TypeError` today on `sophisticated_cnn_128_plus`, `kepler_1d_cnn`, `chess_cnn_attention_policy_value`, `chess_cnn_attention_legal_moves`, `chess_token_candidate_model`, and `chess_token_one_move_model` — none of which declare `compute_dtype` and none of which have a `**kwargs` catch-all silently absorbing it (verified: only the three `aircraft_detector_*` factories have `**kwargs`). So the letter of the SPEC's constraint, read maximally literally, would itself break the SPEC's own non-regression goal for CAP-2 ("sans régression sur les autres domaines").
- This is exactly the kind of tension a spine review should surface, not silently resolve. As written, the spine neither flags this as a deviation from the SPEC nor argues for reopening the SPEC decision — a builder reading only the spine would never know the SPEC said the opposite. **Recommendation: route back through the SPEC (or an explicit spine addendum) to either (a) revise the SPEC's "no introspection" decision now that its literal reading is shown to be unsafe against the current factory signatures, or (b) have the spine explicitly document why it deviates and get that ratified.**

## Moderate finding

### F2 — AD-4 ("apply compute_dtype to LayerNorm sans exception") is inconsistent with the very precedent AD-7 says it must match

- AD-4 Prevents clause: "un traitement spécial ad hoc et incohérent de la normalisation d'un modèle à l'autre" — i.e. its whole justification is cross-model consistency of norm-layer treatment.
- AD-4 Rule: `dtype=self.compute_dtype` applies to **all** `nn.Conv`/`nn.Dense`/`nn.BatchNorm`/`nn.LayerNorm` calls in `SophisticatedCNN32Plus`/`SophisticatedCNN128Lite`, "sans exception."
- But the existing, adopted reference implementation these two models must stay consistent with per AD-7 ("même garantie que le précédent") — `ChessMoveTokenTransformer` — explicitly does the opposite for LayerNorm: every `nn.LayerNorm()` call in that class (model_library.py L1173, 1183, 1190) takes **no** `dtype` argument at all, and its docstring (L1114-1116) gives an explicit rationale: *"`nn.Embed`/`nn.LayerNorm` volontairement laissés en `float32` par défaut ... `force_float32_reductions=True` par défaut sur LayerNorm de toute façon."*
- So after this spine ships, the codebase will have two supposedly-consistent `compute_dtype`-aware models that treat `LayerNorm` oppositely — which is the exact "traitement ad hoc et incohérent... d'un modèle à l'autre" AD-4 claims to prevent, just introduced in the other direction. AD-4's verified BatchNorm justification (flax's internal float32 stat accumulation, confirmed via web search against flax's `normalization.py`/docs) doesn't actually extend to LayerNorm's case, since flax's `force_float32_reductions` only protects the reduction step, not the surrounding elementwise scale/shift compute governed by `dtype`. **Recommendation: either AD-4 should exclude LayerNorm (matching the precedent) or explicitly state why the CNN case should diverge from the Transformer's LayerNorm treatment — right now it does neither.**

## Minor findings

### F3 — Deferred count is off by one ("10 autres modèles" vs. actually 9 named)

Deferred section states "Rollout aux **10** autres modèles de `MODELS`" then names exactly 9: `aircraft_detector_unet`, `centernet`, `centernet_lite`, `chess_cnn_attention_policy_value`, `legal_moves`, `chess_token_candidate_model`, `chess_token_one_move_model`, `kepler_1d_cnn`, `sophisticated_cnn_128_plus`. Actual MODELS dict has 12 entries (verified, model_library.py:1751-1764); minus the 2 this spine touches (`sophisticated_cnn_32_plus`, `sophisticated_cnn_128_lite`) minus the 1 already-adopted consumer (`chess_move_token_transformer`) = 9 deferred, not 10. Small but worth a copy-fix since a wrong count invites a future reader to assume a model is covered by this spine when it isn't (or vice versa).

### F4 — AD-4's "vérifié (recherche web, sources ci-dessous)" promises citations that never appear

AD-4's claim that `flax.linen.BatchNorm` always accumulates statistics in float32 internally is **true** (independently verified via web search against Flax's docs/`normalization.py` — Flax computes batch stats in float32 by default regardless of layer `dtype`, precisely to avoid over/underflow at reduced precision). But the document promises "sources ci-dessous" (sources below) and none exist anywhere in the spine — no bibliography, no links, nothing in the frontmatter `sources:` field either (that field only points back to the SPEC). The underlying fact checks out, but the citation debt should be paid before this is called "vérifié."

### F5 — Pre-existing config-driven forwarding block in `main.py` isn't explicitly retired

`main.py:154-160` currently has `if "compute_dtype" in config: model_kwargs["compute_dtype"] = config["compute_dtype"]` — the exact config-key-presence pattern AD-1/AD-3 are designed to replace. AD-1 explicitly removes the dataset-config literal that fed this branch (`dataset_configs.py:966`), which makes the branch permanently dead code once implemented, but no AD or the Structural Seed explicitly calls out removing this block from `main.py`. Low risk (dead code, not a behavioral bug) but worth naming for a clean implementation, since the Structural Seed table otherwise itemizes exactly what changes in `main.py`.

## Rubric-by-rubric summary

1. **Divergence points fixed for the level below** — Mostly yes for the two target models (AD-1/AD-2/AD-4/AD-5/AD-6 are precise enough that two builders would converge), but F1 means the central injection rule itself is not what the governing spec actually asked for — a builder following the spine faithfully produces code the spec's author didn't sign off on.
2. **Every AD's Rule enforceable and actually Prevents its Divergence** — AD-1, AD-2, AD-5, AD-6, AD-7 hold up under code inspection. AD-3 is enforceable and does prevent what it lists, but conflicts with the SPEC (F1). AD-4 is enforceable but doesn't actually deliver the cross-model consistency it claims (F2).
3. **Nothing under Deferred could let two units diverge silently** — Deferred section is sound; AD-3 guarantees the 9 untouched models keep working. No divergence risk there (see F3 for the arithmetic slip only).
4. **Named technology/library claims plausible/current** — AD-4's Flax BatchNorm claim checked out against current Flax docs/source. No other technology claims in the spine to verify. F4 flags the missing citation trail, not the fact's accuracy.
5. **Ratifies rather than contradicts the brownfield codebase** — Strong: every code line cited (main.py:31-41, main.py:117-160, model_library.py:1218-1221, dataset_configs.py:966, MODELS dict, aircraft_detector_* `**kwargs` absorption trap) was verified accurate against the actual files. F2 is the one place a spine claim (AD-4 "sans exception") doesn't match the actual precedent code (ChessMoveTokenTransformer's LayerNorm exclusion).
6. **Covers the driving spec's capabilities (CAP-1, CAP-2)** — Mapped, but F1 means the mapping is to a mechanism that contradicts the spec's explicit Constraints/Decisions section for the same capabilities.
7. **Every feature-altitude dimension decided/deferred/open** — Operational/environmental envelope explicitly deferred and reasoned (Colab TPU / local GPU-CPU, cold-start detection reused). Checkpoint compatibility, naming, and state/cross-cutting are covered in Consistency Conventions. No dimension found silently unaddressed beyond F5 (minor, implementation-level, not a whole dimension).

## Top findings (compact)

1. **Critical** — AD-3's introspection-based conditional injection directly contradicts SPEC.md's explicit, dated "unconditional forwarding, no registry/introspection" decision (2026-08-17); the spine never acknowledges the deviation even though it's the mechanism governing both CAP-1 and CAP-2.
2. **Moderate** — AD-4 mandates applying `compute_dtype` to `LayerNorm` "sans exception," but the adopted precedent it must match (`ChessMoveTokenTransformer`) explicitly excludes `LayerNorm` from `compute_dtype`, reintroducing the exact cross-model inconsistency AD-4 claims to prevent.
3. **Minor** — Deferred section says "10 autres modèles" but names only 9; MODELS dict has 12 entries total (verified), 2 touched by this spine + 1 already-adopted = 9 remaining, not 10.
4. **Minor** — AD-4 cites "sources ci-dessous" for its Flax BatchNorm claim; no sources appear anywhere in the document (the underlying claim is independently verified true, but the citation is missing).
5. **Minor** — The pre-existing `if "compute_dtype" in config` forwarding block in `main.py:154-160`, made dead by AD-1's removal of the config literal, isn't explicitly called out for removal in the Structural Seed.
