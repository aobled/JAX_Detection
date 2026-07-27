---
title: "Reconciliation — Brief/Addendum vs PRD (jax_supervised_training, chess epic)"
created: 2026-07-27
---

# Reconciliation: brief.md + addendum.md → prd.md

Sources:
- `_bmad-output/planning-artifacts/briefs/brief-jax_supervised_training-2026-07-27/brief.md`
- `_bmad-output/planning-artifacts/briefs/brief-jax_supervised_training-2026-07-27/addendum.md`
- `_bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-07-27/prd.md`

## Method

Read all three documents in full. Mapped every brief/addendum section to its PRD counterpart (Vision, Glossary, FR-1..7, Non-Goals, NFR-1..3, Success Metrics, Open Questions, Assumptions Index). Per the task brief, the addendum's "État de l'art" and "Design de label rejeté" sections are appendix-style background — not expected to be reproduced, only flagged if the PRD loses something load-bearing by not referencing them at all.

## Coverage confirmed (no gap)

- Executive summary / objective (temps 1, non-complexification, quality-of-play deferred) → PRD §1 Vision. Faithful.
- Proven-domains table → PRD §1, copied verbatim.
- Architecture envisagée (CNN 8×8 → bottleneck tokens → attention → policy/value) → PRD FR-5. Justification (relational reasoning, not receptive-field coverage) compressed into FR-5 Notes with pointer to addendum — matches the "appendix, reference is enough" allowance.
- 4 "points ouverts" from brief's Architecture section → PRD §8 Open Questions items 1, 2, 3, 4 — all present.
- Dataset construction (PGN source, python-chess replay, AlphaZero-style planes + 5-move history, policy = move played unfiltered, value = game result signed to side-to-move, draws = 0) → PRD FR-1/FR-2/FR-3. Faithful.
- No external engine for labels → NFR-2, reinforced in Non-Goals with the specific nuance that this holds even as a fallback for a noisy value head (this nuance actually originates in the addendum's risk paragraph, and the PRD correctly pulls it forward — good synthesis, not a gap).
- Rejected "winner's moves = True" labeling design → PRD FR-1 references it by name and cites the addendum for the reasoning — consistent with appendix treatment.
- Hors scope (quality of play, chess_game.py integration, no external engine dependency) → PRD §5 Non-Goals bullets 1–3. Faithful, and Non-Goal 2 correctly carries forward the addendum's chess_game.py detail (Stockfish optional advantage bar, graceful fallback to material eval).
- AD-20 / JAX Single-Pass precedent, Stories 2.4/8.9 methodology reference → PRD FR-7 and §4.3. Named references preserved.
- Success criteria → PRD §7 Success Metrics (SM-1/2/3 + counter-metric SM-C1). Faithful, including the "don't sacrifice pipeline simplicity for play strength" counter-metric framing.

## Gaps found

1. **Dataset scale figure dropped.** Brief's Dataset section states archives are "quelques dizaines de milliers de parties" per player (order-of-magnitude scale) and "victoires/défaites/nulles confondues" at the source-archive level. Neither appears in PRD FR-1 or anywhere else. Low stakes (FR-3 already states no result-based filtering at the example level), but it's a concrete, quantifiable detail from the brief that got silently dropped rather than carried into an Open Question or Assumption.

2. **One addendum risk item didn't make it into Open Questions item 5.** The addendum's "Pièges connus" list has ~5 items: (a) human move noise/blunders, (b) blitz/classical mixing, (c) **non-monotone predictability by player skill — Maia found mid-level players are most predictable, weak and strong players less so**, (d) opening/ECO class imbalance, (e) noisy value-head credit assignment (this last one *is* captured, via the Stockfish-fallback-rejection language in Non-Goals/NFR-2). PRD Open Questions item 5 only lists (a), (b), (d) — mirroring the brief's own shorter risk list, not the addendum's fuller one. Item (c) is a specific named research finding (about Maia) that never surfaces anywhere in the PRD — not in Open Questions, not in Non-Goals, not in the glossary. It's arguably load-bearing for a later architecture decision about whether to bucket training data by player strength (as Maia does) vs. pooling all GM games as this brief proposes — a tension the brief itself doesn't resolve either, but the addendum's finding is directly relevant to it.

3. **Non-Goals bullet 4 is new content not sourced from brief/addendum, and it's not flagged as an assumption.** PRD §5 adds: "Nouveau travail sur les domaines existants (CIFAR10, FIGHTERJET_*, KEPLER) — cette epic ne les modifie pas..." This statement has no equivalent in the brief or addendum — it's a reasonable inference, but the PRD is inconsistent with its own assumption-flagging discipline: the JTBD reformulation in §2 is explicitly flagged `[ASSUMPTION: ...]` and listed in §9 Assumptions Index, while this Non-Goal is presented as if sourced, with no flag and no entry in §9. Not necessarily wrong, but it's the one place the PRD silently adds scope-defining content instead of tracing it back to the source or marking it as inferred.

## Not a gap (checked, ruled out)

- No contradictions found between brief/addendum and PRD on any FR/NFR — all directional statements (non-regression is hard constraint, no external engine, quality-of-play excluded, no unnecessary Trainer complexification) are consistent word-for-word in substance across all three documents.
- The addendum's DeepMind/Maia/AlphaVile "État de l'art" detail (architectures, parameter counts, Elo figures, arXiv IDs) is correctly treated as appendix-only — PRD FR-5 references it by name (Maia-3, AlphaVile, DeepMind searchless) without reproducing figures, which satisfies the "reference is enough" bar since no Open Question or FR depends on the specific numbers.
