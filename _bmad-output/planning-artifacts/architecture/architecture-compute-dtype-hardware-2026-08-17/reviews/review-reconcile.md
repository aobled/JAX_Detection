---
type: reconciliation-review
target: architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md
against: specs/spec-compute-dtype-hardware/SPEC.md + .memlog.md
date: 2026-08-17
---

# Reconciliation Review — compute-dtype-hardware architecture spine vs SPEC

## Verdict

**Minor gaps** (with one deviation, correctly and explicitly superseded in the architecture's own memlog, that deserves to be surfaced more prominently — the rendered SPINE.md never flags that it reverses a decision the SPEC recorded as already "resolved by Aymeric").

## Item-by-item trace

### Capabilities

| SPEC item | Landed? | Where |
|---|---|---|
| CAP-1 (auto derivation, no config string) | Yes | AD-1, AD-2, AD-3; Capability→Architecture Map |
| CAP-2 (generic mechanism, validated CIFAR10+FIGHTERJET) | Yes | AD-3–AD-6; Capability→Architecture Map |

Nuance: SPEC's CAP-1 success text says "**n'importe quelle config** sur TPU applique bfloat16 au calcul interne des couches concernées ... sans qu'aucune clé compute_dtype ne soit présente." Under the spine's introspection design, this is only true for the two adapted models (CIFAR10, FIGHTERJET); a TPU run of, say, `aircraft_detector_unet` will *not* get bfloat16 applied anywhere, it will just fail to receive the kwarg silently-safely. The spine's own Deferred section acknowledges the 10 other models are untouched, so this isn't a hidden gap in substance, but the literal SPEC wording ("n'importe quelle config") is no longer literally true and the spine doesn't call out that qualification explicitly — it only surfaces via the Deferred list. Minor.

### Constraints

| SPEC constraint | Landed? | Where |
|---|---|---|
| Non-régression CHESS_MOVE_TOKEN, checkpoint-compatible | Yes | AD-7 (behavior), AD-6 ("même garantie que le précédent ChessMoveTokenTransformer" for checkpoint) |
| Poids maîtres restent float32 | Yes | AD-6 |
| **Forwarding UNCONDITIONNEL, pas d'opt-in par registre/introspection** | **Reversed** | AD-3 ("Forwarding par introspection stricte du paramètre nommé") — see Finding 1 below |
| GPU float32 = prudence projet, pas limite matérielle; bfloat16/float16 GPU hors scope | Yes | AD-1 rule text + Deferred item 3 |

### Non-goals

| SPEC non-goal | Landed? | Where |
|---|---|---|
| No bfloat16/float16 on GPU | Yes | Deferred item 3 |
| Don't touch separate INPUT dtype cast mechanism (`main.py:31-41`, float16 casting of data) | Implicit only | Not stated as its own line; AD-1 only says the *backend-detection* part of that block is reused, and the Structural Seed's `main.py` entry doesn't mention the input-cast block at all. Respected in substance (nothing in the spine touches it) but never explicitly called out the way the SPEC calls it out — worth a one-line mention in Deferred or a convention row so a future reader doesn't conflate "reuses detection" with "also reuses/modifies the input-cast logic." |
| Don't extend to all models; CIFAR10+FIGHTERJET suffice for this spec; full rollout is a distinct future decision | Yes | Deferred item 1, reinforced by architecture-memlog decision "Q1" |
| No new domain/dataset | Yes (trivially, by scope) | Not restated explicitly, but nothing in the spine creates one — fine to leave implicit |

### Success signal

SPEC calls for real-hardware validation (CIFAR10 local GPU/CPU, FIGHTERJET Colab TPU) held to "même niveau de rigueur que AD-29" (inspect actual dtype/values post-pipeline, not just code reading). The spine is a structural document and doesn't restate a test plan — that's appropriate for an architecture spine (this content belongs in stories/dev), so not a gap, but flagging that the AD-29-rigor callback doesn't appear anywhere in the spine or its memlog. If it's expected to resurface in the eventual epic/story breakdown, nothing in the spine currently guarantees that link forward.

### Assumptions

SPEC assumption that CIFAR10/FIGHTERJET declare no `compute_dtype` field today and must be adapted — confirmed and acted on: AD-4/AD-5 add the field to `SophisticatedCNN32Plus`, `SophisticatedCNN128Lite`, and their shared submodules (`SeparableConv`, `SEBlock`, `SpatialAttention`), and the Structural Seed lists both classes. Fully landed.

### SPEC's own "Decisions" section (already resolved by Aymeric before architecture ran)

| SPEC decision | Landed? | Where |
|---|---|---|
| Forwarding unconditionnel confirmed | **Superseded** | See Finding 1 |
| Hardcoded `"compute_dtype": "bfloat16"` string removed from CHESS_MOVE_TOKEN config | Yes | AD-1 explicitly: "le littéral ... est supprimé" |
| Injection-point design routed to bmad-architecture; sequencing CIFAR10 first, full generalization discussed only after | Yes, followed correctly | Architecture memlog shows CIFAR10 investigated first, FIGHTERJET added only after an explicit Aymeric decision, full rollout deferred (Deferred item 1) |

## Findings

**1. AD-3 reverses a constraint the SPEC recorded as already decided, and the rendered spine never says so.**
SPEC Constraints and SPEC Decisions both state, in Aymeric's own confirmed wording, "Forwarding unconditionnel confirmé ... pas de registre/introspection." AD-3 in the spine does the opposite: it forwards `compute_dtype` only to factories whose signature explicitly names the parameter (`inspect.signature`-based introspection). This is not a silent drop — the architecture run's own memlog documents a real re-litigation with Aymeric (a brownfield audit found that true unconditional forwarding would crash 8 of the 12 `MODELS` entries and, worse, be silently swallowed by 3 of them — `aircraft_detector_unet`/`centernet`/`centernet_lite` — which already drop unrecognized kwargs), and records a fresh decision reversing the SPEC's stance ("Résolution Q1 (Aymeric, 2026-08-17) : introspection automatique ... retenue"). Per the review's ground rules this counts as a legitimate supersession, correctly reasoned and explicitly attributed to Aymeric. The gap is presentational, not substantive: ARCHITECTURE-SPINE.md's AD-3 reads as a fresh decision with no cross-reference back to the SPEC constraint it overturns, so a reader of the spine alone (without the memlog) would not know this is a reversal of an already-"resolved" SPEC decision, and the SPEC.md file itself is now stale on this point until the planned `bmad-spec` update pass (noted in the architecture memlog's `purpose` line) actually runs.

**2. The underlying intent behind the reversed constraint is still honored, just via the opposite tactic.** The real goal behind "unconditional forwarding" was "no model should ever crash from an unexpected kwarg" (SPEC: "chaque classe modèle touchée doit structurellement déclarer/consommer le champ pour éviter un TypeError"). AD-3's introspection approach satisfies that same goal more robustly (zero crash risk for untouched models, and it explicitly avoids the `**kwargs`-swallowing trap that unconditional forwarding would have hit on 3 models). This is a case where the *mechanism* changed but the *invariant it protects* did not — worth noting so the reversal in Finding 1 doesn't read as more alarming than it is.

**3. SPEC's "trajectoire cible" (target trajectory) of an eventual unconditional mechanism is left ambiguous by the pivot.** SPEC's non-goal section explicitly frames unconditional forwarding as the intended end-state ("la trajectoire cible est bien un mécanisme unconditionnel qui couvre in fine tous les modèles"), with only the rollout *timing* deferred. The spine's Deferred section ("Rollout aux 10 autres modèles ... décision à reprendre modèle par modèle") no longer says whether that future rollout will eventually still converge on unconditional forwarding or has permanently adopted introspection as the standing design. Not a defect (it's honestly marked as a future decision), but a reader could reasonably wonder whether the SPEC's stated north star still holds.

**4. The near-miss on FIGHTERJET_CLASSIFICATION scope was caught and closed correctly — flagging as a process observation, not a defect.** The architecture memlog shows an intermediate state where only CIFAR10 was being scoped, which would have silently failed to honor the SPEC's explicit CAP-2 success signal and FIGHTERJET assumption (both already required FIGHTERJET_CLASSIFICATION / TPU validation). The memlog records the gap being noticed and escalated to Aymeric, who confirmed `sophisticated_cnn_128_lite` in scope alongside CIFAR10. The final spine (AD-4, AD-5, Structural Seed, Capability Map) correctly includes both models — no residual gap in the delivered artifact, but worth naming since it's exactly the kind of scope-narrowing that could have shipped unnoticed.

**5. Minor: the SPEC's non-goal about not touching the separate INPUT dtype-cast mechanism (`main.py:31-41` float16 casting of data, as opposed to backend detection) is respected in substance but never gets its own explicit line in the spine** — it's only inferable from what the Structural Seed *doesn't* mention. A one-line addition to Deferred or the Conventions table would make this non-goal traceable in the spine itself rather than only by absence.

## Non-findings (explicitly checked, clean)

- Checkpoint/weights invariant (float32 masters, `compute_dtype` as a non-pytree dataclass field): fully preserved, AD-6.
- BatchNorm/LayerNorm uniform treatment: not a SPEC item, but a well-justified addition (AD-4) backed by real research cited in the memlog, doesn't contradict anything in the SPEC.
- CHESS_MOVE_TOKEN non-regression + hardcoded-string removal: both landed cleanly (AD-1, AD-7).
- No new domain/dataset: respected by scope.
- GPU bfloat16/float16 out of scope: respected, stated in both AD-1 rule text and Deferred.
