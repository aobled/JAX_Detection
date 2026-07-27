---
name: 'Reviewer Gate — Adversarial Incompatibility (one level down)'
type: review
target: architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md
lens: 'Construct two units one level down that each obey every AD to the letter yet still build incompatibly'
date: '2026-07-27'
verdict: CONDITIONAL — do not draft stories until F1 and F2 are closed by a new/tightened AD
---

# Adversarial Incompatibility Review — Epic échecs

## Method

Read the full spine (AD-21..AD-25, Structural Seed, Dependency graph). Grounded every hypothesis
against the actual parent-epic code it must slot into: `main.py:108-166` (model_kwargs assembly +
task_type dispatch), `dataset_configs.py` (`validate_dataset_config`, `required = ["num_classes",
"image_size", "model_name"]`), `task_strategies.py` (`TaskStrategy` ABC, `CenterNetDetectionStrategy`
as the nearest composite-loss precedent), `loss_functions.py:547` (`compute_centernet_loss`, bare-scalar
return), `trainer.py:245-274` (`compute_metrics` must return exactly one scalar, no way to smuggle a
second value out per-step), and `detection_target_encoding.py` (the AD-18 mirror this epic's
`chess_target_encoding.py` is explicitly modeled on). Then constructed, for each spine AD, two
one-level-down units that each satisfy the letter of the AD but disagree on a shared shape/owner/value.

## Verdict

**CONDITIONAL.** AD-21/AD-24 (isolation, zero-touch Trainer, composite loss shape) are solid — the
Trainer's `compute_metrics` single-scalar contract structurally forces AD-24's "policy accuracy gates,
value doesn't" rule, so there is no room for two builders to disagree there. But two real gaps remain
open enough for independently-compliant builders to construct incompatible units: the policy
output-space size has no declared single plumbing path into the model (F1), and the value target's
semantics (regression vs. classification) are underdetermined by AD-25's wording (F2). Both are
silent-failure classes (wrong-shape Dense head, or index-out-of-range cross-entropy) rather than
loud crashes, which is why they need an AD, not just a story-time judgment call. F3-F5 are real but
lower-severity.

## Findings

### F1 (severe) — Policy output-space size: two legitimate owners, no forced equality

**Two units:** (a) `model_library.py`'s `create_chess_cnn_attention_policy_value` factory, sized like
every existing factory in this file (`nn.Dense(self.num_classes)`, e.g. `SophisticatedCNN128Plus`
line 204) — i.e. it reads its output width from `config["num_classes"]`, because that is the *only*
plumbing `main.py` currently offers: `model_kwargs = {"num_classes": num_classes, "dropout_rate":
dropout_rate}` (main.py:118) is a hand-maintained fixed dict, with exactly one precedent for adding a
model-specific extra (`heatmap_prior`, conditionally appended). (b) `chess_target_encoding.py`, which
per AD-18/AD-22 is the sole owner of "le schéma d'encodage coup→index exact" and would naturally
define its own `POLICY_SPACE_SIZE` constant (mirroring `HEATMAP_KEY`/`SIZE_KEY` as the AD-18 source of
truth in `detection_target_encoding.py`) consumed by the PGN producer and the data loader.

**Divergence:** both units are individually AD-compliant. Unit (a) never looks at
`chess_target_encoding.py`; unit (b) never looks at `dataset_configs.py`. `dataset_configs.py`'s
`validate_dataset_config` *requires* a `num_classes` key on every entry (`required = ["num_classes",
"image_size", "model_name"]`) — a builder wiring the `CHESS` entry has to put a literal integer there
regardless. Nothing stops that literal from drifting from the encoding module's canonical size
(e.g. spine defers "gestion précise roque/promotion/en passant dans l'index" to the story — a later
edge-case fix in `chess_target_encoding.py` that changes the index count would not automatically
propagate to `dataset_configs.py["num_classes"]`). Result: model produces a policy vector of size N,
loss/decode expects size M != N — either a hard shape crash (if wired through Dense(num_classes)
directly) or, worse, a silent out-of-range index into a correctly-shaped-by-accident vector.

**AD to add/tighten:** New AD (or tighten AD-22): the chess policy head's output width MUST be
imported programmatically from `chess_target_encoding.py` (e.g. `POLICY_SPACE_SIZE` constant), never
from `dataset_configs.py["num_classes"]`. If `num_classes` must still exist on the `CHESS` config
entry to satisfy `validate_dataset_config`'s required-key check, it must be set by reference to that
same constant (`"num_classes": chess_target_encoding.POLICY_SPACE_SIZE`), never a separately
hardcoded literal — and `main.py`'s model_kwargs assembly must be called out explicitly as a
plumbing point this epic touches (it currently isn't mentioned anywhere in the Structural Seed).

### F2 (severe) — Value target: regression scalar or 3-way classification are both "compliant" readings

**Two units:** (a) a model-library builder reading AD-25's "cible value : scalaire +1/0/-1" and AD-24's
loss mirroring `compute_centernet_loss` (one classification-style term + one regression-style term,
by analogy to heatmap-focal + size-regression) as: value head = `nn.Dense(1)` + `tanh`, value_loss =
MSE against a float32 target in {-1.,0.,1.}. (b) a `chess_target_encoding.py`/loss builder reading the
same "+1/0/-1" as three discrete outcomes and implementing: value head = `nn.Dense(3)` + softmax,
value_loss = cross-entropy against a class index in {0,1,2} (shifted from -1/0/1) — a reading equally
supported by the spine's own policy precedent (AD-22 explicitly specifies cross-entropy for the
*other* head, so "loss mirrors the sibling head's classification style" is not an unreasonable
inference for a builder who only reads AD-24/AD-25, not a chess-ML paper).

**Divergence:** Dense(1) vs Dense(3) is an immediate shape crash if the mismatch is caught early, but
if `compute_chess_policy_value_loss` is written defensively (e.g. broadcasting or squeezing), it can
silently compute a nonsense loss for many epochs before anyone notices — the exact "clash of
shared-data shapes" this review lens is looking for, and exactly the class of bug the project's
existing memory note "verify numbers against logs, not comments" exists to catch after the fact.

**AD to add/tighten:** Tighten AD-25 (or add a new AD binding `chess_target_encoding.py` +
`model_library.py` + `loss_functions.py` together) to pin: value target dtype/shape *and* the value
head's exact architecture and loss form (regression+MSE+tanh, or classification+CE+3-way) — pick one,
state it as unambiguously as AD-22 already does for the policy head ("cross-entropy simple contre
l'index du coup joué, sans masque").

### F3 (moderate) — `generate_reports()`'s policy/value breakdown has no single source of truth

AD-24 says the policy/value split is "calculé et affiché uniquement... dans `generate_reports()`",
but `compute_chess_policy_value_loss` is required to mirror `compute_centernet_loss`'s contract, which
returns a **bare weighted scalar** (`loss_functions.py:556`, no component tuple). A builder
implementing `generate_reports` (per the existing pattern, e.g. `CenterNetDetectionStrategy
.generate_reports`, it re-runs the model on `val_ds` independently of the training loop) therefore
cannot get `policy_loss`/`value_loss` back from `compute_loss` — it must either reimplement the two
underlying primitives locally, or reuse a separately-exported pair (e.g.
`compute_chess_policy_loss`/`compute_chess_value_loss`, analogous to `compute_heatmap_focal_loss`/
`compute_size_regression_loss` which the CenterNet loss composes). Two builders (the loss-functions
story and the strategy/reporting story) could each pick a different one of these paths, and — since
`self.loss_params`/weight storage isn't specified either — `generate_reports` could easily display
breakdown numbers computed with different weights, or a different reduction (mean vs. sum), than what
`compute_loss` actually optimized during training.

**AD to add/tighten:** Extend AD-24 to require `loss_functions.py` export the two named component
functions as the single source of truth for both `compute_chess_policy_value_loss` and
`generate_reports` (neither may reimplement policy/value loss locally), and require
`ChessPolicyValueStrategy` to store `policy_weight`/`value_weight` as named attributes (not just an
opaque `self.loss_params` dict) so `generate_reports` is structurally forced to read the same weights
used in training.

### F4 (moderate) — Value sign convention: ownership of the "côté joueur au trait" flip is unstated

AD-25 pins the *value* — "+1/0/-1 côté joueur au trait" — but not *who* performs the perspective flip.
AlphaZero-style position planes conventionally already encode side-to-move as one of the input planes
(implied by "position encodée façon AlphaZero" in the Consistency Conventions table). Unit (a):
`chess_pgn_dataset_tools.py` computes the flip once at write time, storing an already-resolved value
target. Unit (b): a data-loader/consumer builder, seeing side-to-move is recoverable from the position
planes it already has to decode, defensively re-derives and re-applies the flip "just to be safe" —
double-flipping the sign for whichever color the convention already resolved. Both units are AD-25
compliant (it only constrains the *stored value's meaning*, not the *computation ownership*).

**AD to add/tighten:** Tighten AD-25 to state explicitly that the sign flip is computed exactly once,
at producer time, and that all consumers must treat the persisted value as final — never re-derive or
re-flip it from the position planes.

### F5 (light, pre-empted but worth closing explicitly) — K token count is safe today, fragile under one plausible extension

AD-23 correctly leaves K as a model-internal hyperparameter with no external plumbing required this
epic — so, as written, there is no incompatible pair to construct here today: a builder can hardcode K
inside `model_library.py`'s factory default and nothing else needs to know it exists. The residual
risk is inherited from the same gap as F1: `main.py`'s `model_kwargs` dict is a hand-maintained fixed
set (`num_classes`, `dropout_rate`, +1 hardcoded special case for `heatmap_prior`). If a future story
(plausibly still inside this epic's tuning phase) makes K configurable per `dataset_configs.py` entry
— a natural move, matching the `heatmap_prior` precedent — nothing in the spine says that plumbing
point must be touched, and `get_model(model_name, **kwargs)`'s permissive `**kwargs` means a silently
dropped or silently-defaulted K would not raise.

**AD to add/tighten:** Optional/light — add one sentence to AD-23: "K remains an internal-only default
inside `model_library.py` for this epic; it is never read from `dataset_configs.py` or threaded through
`main.py`'s `model_kwargs`." This forecloses the drift preemptively rather than relying on a future
builder noticing the gap unaided.

## Non-findings (checked, resolved by the spine as written)

- **AD-24 policy-accuracy-gates-only-metric:** structurally enforced by `trainer.py`'s
  `compute_metrics` contract (exactly one scalar returned per step, `trainer.py:250,265`) — no way for
  two builders to smuggle a second metric through this path, so AD-24's rule is not just a convention,
  it's the only thing that fits the existing Trainer shape. Low risk.
- **`policy_weight`/`value_weight` plumbing mechanism itself:** the existing `loss_params` dict
  convention (`dataset_configs.py["loss_params"]` → `main.py:123` → `Strategy(loss_params=...)` →
  `compute_loss(outputs, targets, **self.loss_params)`, identical to `CenterNetDetectionStrategy`) is
  unambiguous and precedented — the risk is downstream in `generate_reports` (F3), not in this
  plumbing itself.
