---
title: Reconciliation — PRD vs ARCHITECTURE-SPINE (Epic échecs)
type: review
scope: architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md vs prd-jax_supervised_training-2026-07-27/prd.md
created: 2026-07-27
---

# Reconciliation: PRD ↔ Architecture Spine (Epic échecs)

Sources:
- PRD: `_bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-07-27/prd.md`
- Spine: `_bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md`

## 1. FR/NFR binding coverage

The spine's frontmatter `binds:` claims: `FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, NFR-1, NFR-2, NFR-3` plus 4 inherited parent ADs. Checked each against the body (AD Rules, Deferred, Consistency Conventions, Structural Seed) via full-text grep for `PRD`, `NFR`, `FR-\d`, `moteur|Stockfish|engine`.

| Req | Status | Where addressed |
|---|---|---|
| FR-1 (PGN extraction → dataset) | Partial | Structural seed names `chess_pgn_dataset_tools.py` as the producer; AD-25 fixes the *output format* (3 fields/example). The extraction mechanics themselves (1 game → N ply-examples, no result-based filtering needed for policy) are not stated as a Rule anywhere — see Gap 2. |
| FR-2 (AlphaZero-style planes + 5-move history) | Covered | AD-18 (inherited, "Binds here" row) — dedicated `chess_target_encoding.py` module fixes the exchange format; Consistency Conventions table restates it. |
| FR-3 (policy = move played, no result-filtering; value = ±1/0 from side-to-move) | Partial | Consistency Conventions table fixes the value sign convention ("côté joueur au trait"). The specific rule "no policy example excluded because the side that played it lost" has no corresponding AD/Rule — see Gap 2. |
| FR-4 (double-head TaskStrategy) | Covered | AD-24 — `ChessPolicyValueStrategy`, composite loss, both losses trained. |
| FR-5 (new model: CNN 8×8 → bottleneck → attention → policy/value heads) | Covered | AD-23 (bottleneck) + AD-22 (policy head) + Structural Seed model entry. Value head internals aren't spelled out, but that's consistent with the deliberate under-specification pattern used elsewhere (e.g. K token count) and isn't PRD-mandated detail. |
| FR-6 (no structural Trainer modification) | Covered | AD-24 Rule: "Aucune modification de `trainer.py`" is explicit and cites the exact log-loop lines that would otherwise need changing (trainer.py:245-274,493-499). |
| FR-7 (real-execution non-regression story) | Covered | AD-21 — explicit baseline-diff methodology, same precedent as AD-20 (parent). |
| NFR-1 (JAX_DETECTOR non-regression, hard constraint) | Covered | AD-21 — substance matches ("contrainte dure, non négociable" in PRD ↔ "sans modification fonctionnelle" + isolated dependency graph in spine), though the literal string "NFR-1" never appears. |
| NFR-2 (no external chess-engine dependency for labels, incl. as fallback) | **Gap** | Not addressed anywhere. Grep for `moteur`, `Stockfish`, `engine` returns zero hits in the spine body. No AD, no Deferred entry, no Consistency Convention rules this out — see Gap 1. |
| NFR-3 (no unnecessary complexification of generic pipeline) | Covered | Design Paradigm section + inherited AD-17 row + AD-24 (Trainer untouched) collectively cover this. |
| AD-3/AD-14/AD-17/AD-18 (parent, inherited) | Covered | All 4 appear in the Inherited Invariants table with a concrete, chess-specific "Binds here" description — not just carried by reference. |

## 2. Gaps found

**Gap 1 — NFR-2 claimed but not addressed (most significant).**
NFR-2 ("no dependency on an external chess engine to generate labels, neither for policy nor value, not even as a fallback") is listed in the spine's `binds:` frontmatter, but no AD, Rule, Consistency Convention, or Deferred entry ever states or enforces it. This is a real PRD constraint (explicitly called out in Non-Goals §5 as a hard exclusion "even as a fallback if the value head turns out noisy") that a future story-writer or implementer would have no architectural anchor for — nothing in the spine would flag a design that quietly pulls in Stockfish for eval refinement, for instance. Recommend adding either a one-line Rule (e.g. folded into AD-25's dataset-format decision, since it already governs `chess_pgn_dataset_tools.py`) or an explicit Deferred/Prevents note.

**Gap 2 — FR-1/FR-3's "no result-based filtering" rule has no AD anchor.**
Both FR-1 ("aucune détermination de gagnant n'est nécessaire pour extraire la policy") and FR-3 ("aucun exemple de policy n'est exclu... au motif que le camp qui l'a joué a perdu") state a specific labeling rule: policy examples are never filtered by game outcome. AD-25 fixes the *shape* of a dataset example (3 fields, no metadata) but is silent on this filtering rule specifically. It's implicit in "N ply → N examples, no filtering description," but never stated as a Rule/Prevents the way AD-22 does for the analogous "no move-masking" decision. Lower severity than Gap 1 (it's arguably a data-pipeline detail rather than an architectural invariant), but worth a one-line addition to AD-25 or a new bullet for consistency with how the spine treats every other "we deliberately do X and not Y" decision.

**Gap 3 — FR-1's batch/epoch-sizing note has no Deferred entry.**
FR-1's conséquences explicitly flag "ordre de grandeur à prendre en compte pour calibrer la taille de batch et le nombre d'epochs en architecture" (tens of thousands of games per pgnmentor archive). The spine never revisits this — no AD, no Deferred bullet. Compare to how AD-23 explicitly defers "K token count" to the story with a named Deferred entry; batch/epoch sizing gets no equivalent placeholder. Minor — likely fine to leave as a story-level hyperparameter — but since the PRD explicitly asked for it to be considered "en architecture," a one-line Deferred entry acknowledging it (mirroring the K-token treatment) would close the loop cleanly.

## 3. Open Questions (PRD §8) — resolution check

All 5 are accounted for, either resolved by an AD or explicitly carried into Deferred (none silently dropped):

1. **Token construction (pooling vs. learned queries)** — Resolved: AD-23 picks learned-query bottleneck, explicitly rejects pooling.
2. **Policy output-space encoding (fixed space + masking vs. alternative)** — Resolved: AD-22 picks fixed AlphaZero-style space, no masking during training; masking itself is explicitly pushed to Deferred ("repoussé au futur projet d'intégration chess_game.py").
3. **Explicit geometric bias in attention (Maia-3-style) vs. naked attention** — Resolved: AD-23 picks standard attention, no bias; explicitly listed in Deferred as a future iteration path, not abandoned.
4. **Detailed double-head TaskStrategy design** — Resolved: AD-24 (composite loss, PolicyAccuracy as primary metric, value head trained but non-gating).
5. **Dataset risks (blunder noise, blitz/classical mixing, opening imbalance, Maia's non-monotone predictability-by-strength finding)** — Carried into Deferred by explicit reference: "si les risques documentés au PRD (§8 Open Questions item 5, bruit/déséquilibre) s'avèrent bloquants en pratique, re-parsing du PGN brut nécessaire." This is a compressed reference rather than a per-risk treatment (it doesn't individually address blunder noise vs. cadence-mixing vs. the Maia predictability paradox), but it does explicitly name the PRD item and carry it forward rather than dropping it — satisfies the "not silently dropped" bar, if thinly.

## 4. Success Metrics / SM-C1 counter-metric consistency check

No contradiction found. Checked specifically for anything implying move-selection/inference work (PRD Non-Goals exclude this):

- AD-22's policy head produces training-time predictions only (cross-entropy vs. played move, top-1 accuracy as a training metric) — not move selection for play.
- AD-24's `PolicyAccuracy` gating the best-model checkpoint is a training-quality signal for the generalization test (SM-1), not a game-strength/Elo proxy — consistent with SM-C1 ("no Elo, no move-correctness benchmark as a closure criterion").
- Move-illegality masking, geometric attention bias, and `chess_game.py` integration — all three are explicitly named as Deferred/out-of-scope, each citing PRD Non-Goals, keeping any move-selection/inference work outside this epic's boundary as required.
- No AD, Rule, or Structural Seed entry touches `chess/chess_game.py` beyond marking it "INTOUCHÉ."

SM-C1 is respected; the spine does not accidentally scope in game-quality or move-selection work.

## 5. Summary

- 8 of 10 bound FR/NFRs are cleanly covered by a concrete AD.
- 1 clear gap: **NFR-2 is claimed but never addressed anywhere in the spine body** (Gap 1).
- 2 minor/soft gaps: FR-1/FR-3's "no result-based filtering" rule lacks an explicit AD anchor (Gap 2); FR-1's batch/epoch sizing note has no Deferred placeholder (Gap 3).
- All 5 PRD Open Questions are resolved or explicitly carried into Deferred — none silently dropped.
- No contradiction of SM-C1 or the game-quality/move-selection Non-Goals found anywhere in the spine.
