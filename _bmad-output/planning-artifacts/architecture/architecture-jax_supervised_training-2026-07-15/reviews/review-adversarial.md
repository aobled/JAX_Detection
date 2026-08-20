---
name: 'Adversarial Review — JAX Single-Pass spine'
type: reviewer-gate-finding
reviews: architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md
inherits-reviewed: architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md
created: '2026-07-15'
mandate: >
  Attack the spine as an adversary: construct two units one level down that each obey
  every AD to the letter yet still build incompatibly — clashing shared-data shapes,
  two owners of one entity, conflicting state-mutation paths.
---

# Adversarial Review — JAX Single-Pass spine

Method: read both spines in full, then cross-checked every AD's Rule text against the
actual repo (`dataset_configs.py`, `main.py`, `task_strategies.py`, `data_management.py`,
`inference_utils.py`, `model_library.py`, `loss_functions.py`) to find places where the
spine names a deliverable but does not pin the value/shape/owner that two independently
spine-compliant builders would need to agree on to interoperate. Six incompatible-pair
scenarios found, ranked most severe first.

---

## Finding 1 [CRITICAL] — `task_type` dispatch string has no pinned value and `main.py` is not in scope

**Binds:** AD-9 (`task_strategies.py`, `data_management.py`)

**The gap:** AD-9's Rule mandates a new dedicated `TaskStrategy` subclass and a new
dedicated dataset-loader class, "dispatchées par `task_type`" — ratifying the existing
pattern. But the existing pattern has **three** independent dispatch sites keyed on the
same string literal, confirmed in the live code:

- `dataset_configs.py`: `DATASET_CONFIGS[name]["task_type"]` (the source value)
- `data_management.py:get_datasets()`: `if task_type in [...]: ... elif task_type == "detection": ...` (confirmed at `data_management.py:429-463`, in AD-9's own binds list)
- `main.py`: `if task_type == "classification": ... elif task_type == "detection": ... elif task_type == "kepler": ... else: raise ValueError(...)` (confirmed at `main.py:107-143`) — **this file is dispatch site #3 and is absent from the child spine's Structural Seed and dependency diagram entirely.**

**Concrete incompatible pair:** Builder A implements `PointDetectionStrategy` in
`task_strategies.py` and, needing somewhere to wire it in, adds a branch to `main.py`
using `task_type == "detection_v2"` (their own invention, since the spine names no
string). Builder B implements the new dataset-loader class in `data_management.py`
per AD-9 and independently sets `JAX_DETECTOR`'s `task_type` to `"point_detection"`
in `dataset_configs.py` (their own invention). Both fully satisfy AD-9's letter — new
dedicated classes, dispatched by `task_type`, existing classes untouched. The result:
`main.py` raises `ValueError(f"task_type 'point_detection' non reconnu.")` at Trainer
startup, or worse — if the strings only *partially* collide (e.g. `data_management.py`
patched but `main.py` forgotten because it's outside the seed) — Trainer silently
falls through to `ClassificationStrategy`'s loss against heatmap+size targets, a
shape-mismatch crash several frames away from the actual cause.

**Fix direction:** Tighten AD-9 (or add AD-11) to (a) pin the literal `task_type` string
for the new format, and (b) add `main.py`'s dispatch `if/elif` chain to the Structural
Seed as a required touch point, explicitly listed alongside `data_management.py`'s.

---

## Finding 2 [HIGH] — heatmap+size on-disk interchange format is named but not pinned across three deliverables

**Binds:** AD-9, dependency graph nodes `NEWDSTOOL → JAXDETCFG`, `JAXDETCFG → NEWDATA`, `NEWSTRAT → NEWLOSS`

**The gap:** the Structural Seed describes `fighterjet_detection_dataset_tools_v2.py` as
producing "cibles heatmap+taille depuis raw_boxes, pas de masque intermédiaire" — so the
tool bakes the heatmap+size *targets* themselves (good, that much is pinned: no raw-box
deferral to load time). But nothing pins:

- array shape/channel semantics (`(H,W)` vs `(H,W,1)`; is `offset` — CenterNet's usual
  sub-pixel correction term — included alongside `heatmap`+`size`, or only the two named
  in the AD text?)
- key names in the saved `.npz` (`heatmap`/`size` vs `hm`/`wh`, etc.)
- size encoding units (raw pixels at heatmap resolution? at 224 resolution? log-scale,
  as CenterNet's original paper uses to stabilize regression?)
- the Gaussian-splat radius formula used to paint the heatmap around each center (CenterNet's radius-from-box-size formula is a specific, non-obvious computation — "heatmap focal loss" in `loss_functions.py` only tells the loss shape, not the target-generation formula)

**Concrete incompatible pair:** Builder A writes `fighterjet_detection_dataset_tools_v2.py`
saving `.npz` with keys `{"hm": (56,56,1) float32, "wh": (56,56,2) float32 in raw pixel units at heatmap scale}`. Builder B, working from AD-9's Rule text and `loss_functions.py`'s
existing `compute_focal_loss` naming convention, writes the new data-loader class in
`data_management.py` expecting keys `{"heatmap": (H,W), "size": (H,W,2) in 224-pixel
units}`. Both satisfy AD-9 to the letter ("own dedicated classes/tool, heatmap+size
format"); the pipeline fails on first load with a `KeyError`, or worse, silently loads
zeros via `.get()`-style defensive code and trains against an all-zero target.

**Fix direction:** Add an AD (or extend AD-9) that pins the `.npz` schema explicitly:
key names, shapes, dtype, coordinate scale (heatmap-native vs 224-native), and whether
`offset` is a third target channel — treat this the same way AD-5/AD-7 pin coordinate
spaces for the inference side.

---

## Finding 3 [HIGH] — heatmap output stride is unspecified; peak-extraction owner unclear

**Binds:** AD-1 (child), AD-5

**The gap:** AD-1's Rule says `JAX_DETECTOR` "prédit un heatmap de centres... jamais un
masque de segmentation plein format," extracted via `reduce_window` + `top_k`. It never
states the heatmap's spatial resolution relative to the 224×224 input — CenterNet-family
detectors conventionally downsample by a stride (commonly 4, giving a 56×56 heatmap for
a 224×224 input), trading localization precision for compute. AD-5's Rule then says
RESCALE "ramène les coordonnées de boîte du repère 224×224 (sortie du détecteur) vers le
repère image d'origine" — implicitly assuming detector output coordinates already live
in 224-pixel space, but never states which artifact performs the grid-index → 224-pixel
conversion (multiply by stride) or where that stride constant is defined.

**Concrete incompatible pair:** Builder A (`model_library.py`, the new detector head)
picks a stride-4 head (56×56 heatmap) — a defensible, common choice for this class of
model, fully within AD-2's "reste dans la même famille... encoder-decoder simple" and
AD-1's letter. Builder B (`inference_utils.py`, peak-extraction + RESCALE inside
`build_single_pass_predict_fn`) reads AD-1's "extraction 100% JAX-native: maxima locaux
+ Top-K" and AD-5's "repère 224×224" and takes `top_k` indices as pixel coordinates
directly (stride=1 assumption, since no stride is mentioned anywhere in either spine).
Both are spine-compliant. Result: every detected box is systematically off by a factor
of 4 in both position and implied scale — RESCALE then projects already-wrong
224-space coordinates into 1920×1080 space, silently, with no shape error to catch it
(this is a coordinate-*value* bug, not a coordinate-*shape* bug, so nothing crashes).

**Fix direction:** Tighten AD-1 to pin the heatmap output stride (or explicitly assign
it to a named config field), and tighten AD-5 to explicitly state which artifact is
responsible for the grid→224-pixel conversion prior to RESCALE proper.

---

## Finding 4 [MEDIUM] — detection-score threshold: no pinned value, and two live precedents conflict

**Binds:** AD-7

**The gap:** AD-7's Rule states `valid_mask` is "dérivé du score de détection... comparé
à un seuil" but never states the threshold's numeric value or where it is defined. The
existing codebase has **two established, mutually exclusive precedents** for exactly
this kind of constant, both explicitly ratified elsewhere in these two spines:
- module-private constant pattern (parent AD-2/AD-3: `_CLF_BATCH_SIZE`, `DETECTION_IMAGE_SIZE`, private to `inference_utils.py`, "jamais redéfinies dans un fichier consommateur")
- function-default-parameter pattern (`decode_segmentation_and_detect(..., conf_threshold=0.3, box_aera_min=225)`, the live signature in `inference_utils.py` today)

**Concrete incompatible pair:** Builder A follows the AD-2/AD-3 precedent and defines
`_DETECTION_CONF_THRESHOLD = 0.3` as a private module constant in `inference_utils.py`,
non-configurable. Builder B, noting AD-8's principle that `build_single_pass_predict_fn`
"lit `JAX_DETECTOR`... via `get_dataset_config()`," instead adds a `detection_threshold`
key to `JAX_DETECTOR`'s `DATASET_CONFIGS` entry and reads `config["detection_threshold"]`
inside the composition. Both are individually consistent with real precedent elsewhere
in the codebase and with AD-7's letter. If both patches land independently (e.g. one
per story, no cross-check), you get one of: a `KeyError` if the config key is assumed
present but Builder A never added it, or two silently-different threshold values live in
the codebase depending on which code path a given caller exercises.

**Fix direction:** Tighten AD-7 to pin both the numeric default *and* which of the two
established patterns owns it (config field vs. private constant) — do not leave a project
with two live, contradictory precedents to choose from.

---

## Finding 5 [MEDIUM] — canonical 224×224 has two independent sources of truth

**Binds:** AD-4, AD-8

**The gap:** AD-4's Rule hardcodes the composition's `RESIZE` target as a literal:
"`RESIZE` déterministe fixe vers 224×224." AD-8's Rule separately requires `JAX_DETECTOR`
to be a standard `DATASET_CONFIGS` entry, which (per `validate_config`'s `required =
["num_classes", "image_size", "model_name"]`) must carry its own `image_size` key —
almost certainly `(224, 224)` today, since AD-6 pins detector training to 224×224 chunks.
Nothing in either AD states that the `RESIZE` step in `build_single_pass_predict_fn` must
*read* `get_dataset_config("JAX_DETECTOR")["image_size"]` rather than hardcode `224`
independently.

**Concrete incompatible pair:** Builder A implements `RESIZE` in `inference_utils.py`
exactly as AD-4 literally states — a hardcoded `224` — fully compliant. Builder B, months
later, retrains `JAX_DETECTOR` at a different resolution to chase small-aircraft recall
(the exact scenario AD-2's Deferred note anticipates — "reconsidérer... si la détection
de petits avions distants s'avère insuffisante") and updates only `dataset_configs.py`'s
`image_size`. Both actions are individually spine-compliant (AD-4 was followed to the
letter when written; AD-2's Deferred note explicitly permits the retrain). The two
now-divergent 224-vs-N literals mean the composition resizes to the old value while the
checkpoint expects the new one — the input layer's shape (`jax.jit`) will error, but the
inconsistency is a design defect (two owners of one number) whether or not it crashes.

**Fix direction:** Tighten AD-4 (or AD-8) to state explicitly: `RESIZE`'s target
resolution is *derived from* `JAX_DETECTOR`'s config `image_size`, not independently
hardcoded — one source of truth, not two.

---

## Finding 6 [LOW-MEDIUM] — invalid-slot padding convention unpinned beyond `valid_mask`

**Binds:** AD-7

**The gap:** AD-7 pins that `valid_mask` — not classification confidence — is the sole
authority on slot validity. It does not state what numeric values populate `boxes`,
`classes`, `class_scores`, `detection_scores` at slots where `valid_mask` is `False`.
Given the pipeline's actual mechanics (`jax.lax.top_k` over a flattened heatmap always
returns exactly 20 indices, real or not — there is no "empty slot" in the underlying
array, only low-score ones), the natural implementation leaves **non-zero, plausible-
looking garbage coordinates** in unmasked slots rather than zeros — but this is nowhere
stated as a Rule, only implied by how `top_k` happens to behave.

**Concrete incompatible pair:** Builder A implements the composition exactly per AD-3/
AD-7 (`top_k` + `vmap`'d CROP over all 20 slots uniformly, no branching) — invalid slots
end up with real (if low-confidence) decoded box coordinates, never zeroed. Builder B,
migrating `bounding_boxes_with_classification_from_video_generation.py` to the new
composition (an explicit Deferred/Structural-Seed item — "migre vers
`build_single_pass_predict_fn` (à terme)"), carries over the old list-pipeline's mental
model where an absent detection simply isn't in the list, and filters the new fixed-size
output with `if box.sum() > 0` instead of `if valid_mask[i]`. Both builders are spine-
compliant in isolation; the migrated consumer silently displays/uses garbage detections
that `valid_mask` correctly flagged as invalid.

**Fix direction:** Tighten AD-7 to state explicitly that `boxes`/`classes`/`class_scores`
values at `valid_mask == False` slots are **unspecified/non-zero and must never be
interpreted directly** — every consumer must gate on `valid_mask`, full stop — closing
the door on zero-check-based filtering entirely rather than leaving it implicit.

---

## Summary table

| # | Severity | ADs | Clash type |
|---|----------|-----|------------|
| 1 | CRITICAL | AD-9 | Two owners of one entity (`task_type` string) + missing scope (`main.py`) |
| 2 | HIGH | AD-9 | Clashing shared-data shape (on-disk `.npz` schema) |
| 3 | HIGH | AD-1, AD-5 | Clashing shared-data shape/value (heatmap stride) |
| 4 | MEDIUM | AD-7 | Two owners of one entity (threshold: config vs. constant) |
| 5 | MEDIUM | AD-4, AD-8 | Two owners of one entity (224×224 literal) |
| 6 | LOW-MEDIUM | AD-7 | Two owners of one entity (invalid-slot padding convention) |

**Status note (2026-08-19):** Findings 1–6 above predate `AD-17` through `AD-21`. `AD-17`
closes Finding 1 (`main.py` added to `task_type` dispatch scope, single literal pinned
across three files). `AD-18` closes Finding 2 (heatmap+size `.npz` schema owned by a
single encode/decode pair). `AD-13` closes Finding 3 (stride as a single named value).
`AD-15` closes Finding 4 (threshold pinned to `JAX_DETECTOR` config, single location) and
Finding 6 (`valid_mask` as sole authority, non-zero invalid-slot values documented). `AD-12`
closes Finding 5 (224×224 literal replaced by `image_size` derived from `JAX_DETECTOR`
config). Left in place for history; not re-litigated below.

---

# Adversarial Review Pass 2 — AD-21 (`build_kwargs_from_config`)

**Date:** 2026-08-19
**Scope:** `AD-21` only (signature-introspection kwargs construction in `main.py` +
`model_library.py`), per the same mandate as Pass 1 above. Cross-checked AD-21's Rule text
against the live pre-refactor code (`main.py:128-298`, `task_strategies.py` constructors,
`model_library.py` factories `MODELS`/`get_model`) and against the inherited constraints
from the sibling spine `architecture-compute-dtype-hardware-2026-08-17` (`AD-1`, `AD-3`).

Method: for each clause of AD-21's Rule, tried to construct two separately-plausible,
independently AD-21-compliant implementations of the same clause and checked whether they
would produce different runtime behavior or diverging diagnostics with no test/error to
catch it. Four incompatible-pair scenarios found.

## Finding 7 [HIGH] — "overrides vs. config" has no rule for *which* keys are pre-computed, so two builders default-swap silently

**Binds:** AD-21 (`main.py` construction of `overrides` for the Strategy call site)

**The gap:** AD-21's Rule names exactly three keys that live in `overrides` because
they're "calculées côté `main.py`": `compute_dtype`, `dropout_rate`, `num_classes`. It says
nothing about the other kwargs the live code currently threads through the Strategy
branches — `loss_method`, `metric_method`, `report_method`, `metric_threshold`,
`label_smoothing`, `mixup_alpha` — which today are computed as **shared local variables in
`main.py`** (`main.py:193-196`, `config.get("loss_method", "cross_entropy")` etc.) and
passed explicitly into every branch, even branches whose own class default differs.
Concretely: `DetectionStrategy.__init__` defaults `loss_method="segmentation"`
(`task_strategies.py:172`), but `main.py`'s current call site always passes the
shared `loss_method` local (default `"cross_entropy"` if config omits the key),
overriding that class default. `FIGHTERJET_DETECTION`'s config happens to set
`"loss_method": "segmentation"` explicitly today (`dataset_configs.py:191`), so this never
surfaces — but nothing pins that it must for a future config.

AD-21's Rule is silent on whether this class of value belongs in `overrides` (preserving
today's "main.py-wide default always wins" behavior) or should flow through raw `config`
(letting `build_kwargs_from_config`'s "not found → omitted → target's own default applies"
path take over, which is the behavior AD-21's Prevents clause implies it wants: "son seul
besoin réel est d'exister dans la signature de la factory/classe cible"). Both readings are
individually AD-21-compliant; they are not the same reading.

**Concrete incompatible pair:** Builder A wires the `model_kwargs` call site (per the
Structural Seed, `overrides={"num_classes": ..., "dropout_rate": ..., "compute_dtype":
...}`, exactly the three named keys — faithful to AD-21's literal list). Builder B, in a
separate story, wires the Strategy-kwargs call site and — reasonably, minimal-diff — keeps
`loss_method`/`metric_method`/`report_method`/`metric_threshold` as pre-computed
`main.py` locals fed into `overrides` too, matching the *pre-AD-21* code's shape. Builder
C, later, adds a brand-new `TaskStrategy` subclass with a new optional kwarg carrying a
class-specific default that differs from whatever ad-hoc default `main.py` might
independently compute for a similarly-named key — under Builder B's "everything
pre-computed as override" convention, Builder C's class default is dead on arrival for any
config that omits the key; under a "let it flow through config" convention it would work
as the class author intended. Nothing in AD-21 tells Builder B or C which convention holds,
and the disagreement is silent — no error, just a different default.

**Fix direction:** Tighten AD-21 to state explicitly that `overrides` is reserved **only**
for values not representable as a plain `config` lookup (values sourced from a *different*
dict than `config`, e.g. `backend_config`, or genuinely computed/derived, e.g.
`compute_dtype`) — everything else, including `loss_method`/`metric_method`/
`report_method`/`metric_threshold`/`label_smoothing`/`mixup_alpha`, must be left to flow
through raw `config`, with "absent from config" meaning "let the target class's own
constructor default apply," never a `main.py`-wide substitute default.

## Finding 8 [MEDIUM-HIGH] — `overrides` keys other than `compute_dtype` have no AD-1-style protection against config-key collision

**Binds:** AD-21 (`overrides` mechanism), inherited `AD-1` (sibling spine)

**The gap:** the inherited `AD-1` guarantees `compute_dtype` specifically can never appear
in a dataset `config`, which is exactly what makes "`overrides` always wins" safe for that
one key. AD-21 generalizes the override-wins mechanism to *all* forwarded kwargs but
extends no equivalent protection to the other keys it names as overrides —
`dropout_rate` and `num_classes` — nor to any future one a builder adds. `main.py`
currently builds `dropout_rate` unconditionally for every model
(`dropout_rate = backend_config["dropout_rate"]`, `main.py:129`, fed into every
`model_kwargs`/`overrides` call), sourced from `backend_config`, not `config`.

**Concrete incompatible pair:** Builder A adds a new model factory (e.g. a follow-on to
`chess_token_candidate_model`) whose named parameter is also spelled `dropout_rate` but
whose intended semantics differ (e.g. a per-block list rather than one scalar), documented
by Builder A as a new field in that model's dataset config:
`"dropout_rate": [0.1, 0.2, 0.1]`. This is exactly the pattern AD-21's Prevents clause
promises works ("son seul besoin réel est d'exister dans la signature de la factory
cible"). But `main.py`'s pre-existing, unconditional `overrides["dropout_rate"]` (a single
float from `backend_config`) wins per AD-21's own rule, silently discarding Builder A's
config value before the target ever sees it — the collision is invisible until the model
crashes on a type mismatch (or worse, silently coerces/broadcasts and trains wrong).
`compute_dtype` is safe from this only because a *different* spine's `AD-1` separately
forbids it from ever appearing in `config`; no such rule exists for `dropout_rate` or
`num_classes`.

**Fix direction:** Either (a) tighten AD-21 to declare `dropout_rate` and `num_classes`
reserved names — no dataset config may ever declare them for a different purpose, mirroring
`AD-1`'s treatment of `compute_dtype` explicitly rather than by silent analogy — or (b)
require `overrides` to be scoped per-target rather than passed as one blanket dict built
once for every model/Strategy, so a collision would fail loudly (target sees two candidate
sources) rather than resolving silently.

## Finding 9 [MEDIUM] — `compute_dtype` "injected?" print is required to stay a hand-rolled duplicate of the helper's own introspection, with nothing pinning the two checks together

**Binds:** AD-21 (`main.py`'s compute_dtype diagnostic print vs. `build_kwargs_from_config`'s internal introspection)

**The gap:** AD-21's Rule explicitly carves the diagnostic print out of the helper: "le
print diagnostique dédié à `compute_dtype` (« injecté »/« non applicable ») reste explicite
à côté du helper, pas absorbé silencieusement." That means `main.py` must keep its own
`"compute_dtype" in inspect.signature(target_factory).parameters` check (today at
`main.py:185`, pre-refactor) purely to decide what the print says, **separate from** the
equivalent check `build_kwargs_from_config` performs internally to decide what actually
gets forwarded. AD-21 pins that both checks must exist; it does not pin that they must stay
expression-identical.

**Concrete incompatible pair:** Story A ships `build_kwargs_from_config` with a plain
`inspect.signature(target).parameters` check (matches `main.py`'s hand-rolled one — prints
and behavior agree). Story B, later, extends `build_kwargs_from_config` only — e.g. to
correctly unwrap a `functools.partial`-wrapped or decorated factory before introspecting,
a legitimate, AD-21-compliant enhancement to "introspecte la signature de `target`." Story
B has no reason to touch `main.py`'s separate print-diagnostic check (AD-21 says it stays
external to the helper), so it doesn't. Now the helper correctly forwards `compute_dtype`
through the wrapper while `main.py`'s cruder check still reports "non applicable" — a
diagnostic that actively lies about whether injection happened, for the exact value
(`compute_dtype`) the sibling spine treats as high-stakes enough to warrant its own
dedicated print in the first place.

**Fix direction:** Tighten AD-21 to require the diagnostic print derive its
injected/not-applicable verdict **from the same call** to
`build_kwargs_from_config` (e.g. by having the helper return which override keys were
actually consumed, and `main.py` print off that return value) rather than from an
independently-maintained introspection expression — "explicit, not absorbed" should mean
"a visible print statement," not "a second, separately-evolving implementation of the same
check."

## Finding 10 [LOW-MEDIUM] — `STRATEGIES` dict lookup failure mode for an unknown `task_type` is unspecified, inconsistent with the adjacent `get_model()` pattern

**Binds:** AD-21 (Strategy dispatch), AD-17 (single dispatch point, same literal)

**The gap:** the pre-refactor `if/elif` chain ends `else: raise ValueError(f"task_type
'{task_type}' non reconnu.")` — an explicit, message-bearing error. AD-21 replaces this
with `STRATEGIES = {task_type: Classe}` and says a dict "est une implémentation valide" of
AD-17's single dispatch point, but does not say how a missing key is handled. A plain
`STRATEGIES[task_type]` raises a bare `KeyError` with no explanatory message — a real
regression in debuggability, and inconsistent with the pattern immediately adjacent in the
same function: `model_kwargs` construction deliberately uses `MODELS.get(model_name)` +
`get_model()`'s own explicit `ValueError` (`model_library.py:1812-1813`) specifically
*because*, per `main.py`'s own comment at the call site, "un `model_name` invalide/mal
orthographié ne doit pas planter ici avec un `KeyError` brut" (a lesson from the 2026-08-17
Edge Case Hunter review, cited in-line). AD-21 does not require the same discipline for the
`task_type` → `STRATEGIES` lookup it introduces right below it.

**Concrete incompatible pair:** Builder A implements `STRATEGIES[task_type]` directly
(terse, matches the dict-literal spirit of "dispatch unique" in AD-21's text). Builder B,
adding a new `task_type` in a later story, writes a small config-validation helper (e.g. a
pre-flight check invoked from a `tools/` script or test) that expects to catch a
`ValueError` with the `"non reconnu"` message pattern to confirm a typo'd `task_type` is
caught before a training run starts — matching every other dispatch failure mode in this
codebase (`get_model`, the original `if/elif`). Builder B's check silently never fires
(catches `ValueError`, gets `KeyError`), reporting configs as valid when they are not.

**Fix direction:** Tighten AD-21 to require `STRATEGIES.get(task_type)` + an explicit
`raise ValueError(f"task_type '{task_type}' non reconnu. Stratégies disponibles:
{list(STRATEGIES)}")` on a `None` result — same shape as `get_model()`'s guard, stated
explicitly rather than left to be inferred from "a dict is a valid implementation."

---

## Summary table — Pass 2 (AD-21)

| # | Severity | ADs | Clash type |
|---|----------|-----|------------|
| 7 | HIGH | AD-21 | Two owners of one entity (default-value source: pre-computed override vs. raw config) |
| 8 | MEDIUM-HIGH | AD-21, AD-1 (sibling) | Two owners of one entity (`dropout_rate`/`num_classes` key, unprotected collision) |
| 9 | MEDIUM | AD-21 | Conflicting state-mutation paths (diagnostic print vs. actual injection, duplicated logic) |
| 10 | LOW-MEDIUM | AD-21, AD-17 | Two owners of one entity (missing-key failure mode: `KeyError` vs. `ValueError`) |
| 6 | LOW-MEDIUM | AD-7 | Conflicting consumption convention (padding/validity) |
