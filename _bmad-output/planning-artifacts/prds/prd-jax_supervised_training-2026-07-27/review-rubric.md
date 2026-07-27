# PRD Quality Review — prd-jax_supervised_training-2026-07-27

## Overall verdict

This PRD is a faithful, tightly-scoped translation of the validated brief into a capability spec for a solo technical epic — the thesis (test `Trainer`/`TaskStrategy` genericity against a structurally novel domain) is stated once and every FR, NFR, and SM traces back to it without padding. The strongest asset is honesty under pressure: FR-6 names the exact place where the "no `Trainer` change" success criterion could break (policy output-space encoding, §8 item 2) instead of hiding it, and NFR-2's non-negotiable "no external chess engine" constraint is held even against a documented mitigation (Stockfish-as-value-label fallback) that the brief's addendum shows was actively considered and rejected. The only real softness is in done-ness: a few FRs lean on undefined comparison criteria ("résultats identiques" in FR-7, "s'avère nécessaire" in FR-6) that an architecture/story session will have to operationalize rather than inherit ready-made.

## Decision-readiness — strong

Trade-offs are named, not smoothed. FR-6's Out of Scope note states plainly that the policy output-space encoding choice (§8 item 2) "impacte directement la faisabilité de ce FR" — the PRD does not pretend the "no structural `Trainer` change" bet is safe; it flags the exact fork where it could fail and defers the call to architecture rather than asserting an answer. NFR-2 ("aucune dépendance à un moteur d'échecs externe... y compris comme repli en cas de value head bruitée") is a real constraint held under a real counter-pressure — the brief's addendum shows the Stockfish-fallback option was seriously considered (DeepMind's approach) and explicitly rejected, and the PRD carries that rejection forward as non-negotiable rather than softening it to "preferred."

Open Questions (§8) are genuinely open — deferred to the architecture session with the specific decision named (token construction, output-space encoding, attention bias, dual-head `TaskStrategy` design, dataset risk mitigation), not rhetorical questions answered in the next clause.

### Findings
- **low** Self-graded success condition (FR-6) — "Si une modification de `Trainer` s'avère nécessaire... elle est documentée explicitement comme un écart" has no named arbiter for what counts as "necessary" vs. merely convenient. Low severity because the epic is solo/internal (Aymeric is both author and judge), but worth a one-line note in architecture framing so the criterion doesn't drift during implementation. *Fix:* add a sentence in FR-6 or NFR-3 naming the test ("necessary" = no alternative encapsulation exists within `TaskStrategy`'s current interface).

## Substance over theater — strong

No personas, no Vision boilerplate. The Vision (§1) is domain-specific and non-swappable — it names the exact prior domains, their `task_type`s and models in a table, and states precisely what's new (input is a structured game state, not a fixed-geometry tensor; output is policy+value, not a class or spatial map). NFRs are not boilerplate: NFR-1/2/3 each name a specific, falsifiable constraint (no `JAX_DETECTOR` impact, no external engine dependency, no structural `Trainer` change) rather than "must be reliable/scalable." The `[ASSUMPTION: ...]` tag on the JTBD in §2 is honest about being a reformulation rather than a direct quote — this is the opposite of theater.

No findings.

## Strategic coherence — strong

Thesis is explicit in §1 ("valide que le pipeline générique reste utilisable... sans être complexifié inutilement") and every feature group (4.1 dataset, 4.2 pipeline integration, 4.3 non-regression) serves it. SM-1/SM-2/SM-3 measure genericity and non-regression, not activity — there is no vanity metric here (no dataset size targets, no "number of positions processed"). SM-C1 is a real counter-metric, not decorative: it explicitly forbids trading pipeline simplicity for play-strength gains within this cycle, which is the one place a technically-minded implementer might be tempted to over-invest. MVP scope kind is "platform/technical capability" and the scope logic (Non-Goals in §5) matches that framing consistently.

No findings.

## Done-ness clarity — adequate

Most FRs carry testable consequences: FR-1 (N plies → N examples, no external engine), FR-3 (exactly one policy + one value label per example, no result-based exclusion), FR-4 (two losses visible separately in logs), FR-5 (registered as a distinct `model_name`, two distinct consumable outputs) are all verifiable without interpretation. This is the dimension where the PRD is strongest overall, but three spots would leave an engineer guessing at the acceptance bar:

### Findings
- **medium** FR-7's "identique" is undefined (§4.3, FR-7) — "un entraînement et/ou une inférence `JAX_DETECTOR` lancés après l'epic produisent des résultats identiques à une baseline capturée avant l'epic" does not say what is compared: bit-exact weights, loss/metric values within a tolerance, or output predictions (boxes/heatmaps) unchanged. This is the operationalization of the epic's one hard non-negotiable constraint (NFR-1), so ambiguity here is higher-stakes than elsewhere. Partially mitigated by the reference to "Stories 2.4/8.9 des epics précédentes" as methodological precedent, which likely already fixes a comparison method — but the PRD doesn't restate it, so a reader without that history has no bar. *Fix:* either inline the comparison method from Stories 2.4/8.9 (metric, tolerance, artifact compared) or explicitly note "method identical to Story 2.4/8.9, see [link]" so the cross-reference resolves without tribal knowledge.
- **medium** FR-6's "s'avère nécessaire" (see Decision-readiness finding above) is also a done-ness gap from the story-writer's side — there's no acceptance test for whether a proposed `Trainer` change was "necessary," only a process commitment to document it if it happens. *Fix:* same as above, define the test once and reference it from both FR-6 and NFR-3.
- **low** FR-2's consequence — "sans étape de transformation manuelle supplémentaire côté `TaskStrategy`" — uses "exploitable directement," which is close to an adjective claim ("reasonable," "user-friendly" family) rather than a bound. It's rescued by being paired with a concrete negative claim (no manual transform step), but the positive claim ("exploitable directement") itself isn't independently testable. *Fix:* drop the adjective clause and keep only the falsifiable one, or specify the expected tensor shape/dtype contract FR-5's model expects.

## Scope honesty — strong

§5 Non-Goals does real work — four explicit exclusions, each with a one-line reason, including the counter-intuitive one (no external chess engine "y compris comme option de repli si la value head s'avère bruitée," directly closing off a mitigation path that the brief's addendum shows was tempting). Both `[ASSUMPTION: ...]` tags are substantive, not filler, and both are indexed in §9. Open-items density (5 Open Questions + 2 Assumptions) is proportionate for a PRD that explicitly hands architecture-level decisions to a dedicated Winston session — this isn't a green-light-to-build document pretending to be complete; §0 states outright that it "sert de base directe à la session d'architecture."

### Findings
- **low** `[NOTE FOR PM]` convention not used — FR-4 and FR-5 each carry a **Notes:** callout pointing at real open tensions (dual-head `TaskStrategy` design, token/attention choices), which is the right instinct, but they use plain "**Notes:**" rather than the `[NOTE FOR PM]` bracket tag the rubric (and presumably the PM skill's own convention) expects. Content is fine; a reader grep-ing for `[NOTE FOR PM]` to find deferred decisions would miss these. *Fix:* retag as `[NOTE FOR PM]` for consistency with the convention, or confirm this project intentionally uses a lighter "Notes:" style throughout (in which case no fix needed, just consistency).

## Downstream usability — strong

Glossary (§3) is used consistently: `task_type`, `TaskStrategy`, policy/value heads, Position, ply, "planes façon AlphaZero," "bottleneck de tokens" all appear with identical casing/backticking across FRs and Open Questions — no drift observed. FR/NFR/SM numbering is contiguous and unique (FR-1..7, NFR-1..3, SM-1..3 + SM-C1) and every cross-reference resolves: SM-1 cites FR-4/5/6/NFR-3, SM-2 cites FR-7/NFR-1, SM-3 cites FR-1/2/3/NFR-2 — full traceability, no orphaned FR or NFR. §8 Open Question items are individually numbered and referenced by number from FR-4/5/6, which is the right pattern for a document meant to be pulled apart section-by-section by an architecture session.

No findings.

## Shape fit — strong

Correctly shaped as a capability spec for a single-operator internal tool: no UJs, no persona section, no MVP-Scope section (folded into Non-Goals per §5) — consistent with the instruction that this PRD deliberately omits both, and the omission doesn't cost decision-readiness since §2 ("Contexte d'usage") carries the JTBD framing that would otherwise live in a UJ. SMs are operational/binary ("branché sans modification de `Trainer`," "résultat identique") rather than user-facing, matching the internal/epic stakes explicitly claimed in §7's italic note. Brownfield references are accurate and verified against the actual repo: `./chess/chess_game.py` is described as "240 lignes" with Stockfish-or-material-eval fallback — confirmed against the actual file (240 lines, `stockfish_path` fallback logic at lines 29-38).

No findings.

## Mechanical notes

- **Assumptions Index roundtrip — partial gap.** §9 lists two entries, but only the first ("JTBD reformulé...") has a corresponding inline `[ASSUMPTION: ...]` tag in the body (§2). The second §9 entry ("Portée de ce PRD limitée au contenu du brief... aucun élément supplémentaire n'a été apporté en brain dump") is a meta-assumption about the PRD-writing session itself and has no inline tag anywhere in the document body. Low-impact (it's a scope-of-session note, not a domain assumption that downstream work would misread), but technically breaks the index↔inline roundtrip the rubric checks for.
- **ID continuity** — clean. FR-1 through FR-7, NFR-1 through NFR-3, SM-1 through SM-3 plus SM-C1: no gaps, no duplicates.
- **Glossary drift** — none observed. Bilingual French-prose/English-ML-term style (e.g., "tête policy," "value head") is used consistently throughout rather than drifting between synonyms.
- **UJ protagonist naming** — N/A, no UJs present (correctly, per shape fit).
- **Required sections for stakes** — present: Vision, Contexte d'usage (JTBD), Glossaire, Fonctionnalités (FR), Non-Goals, NFR, Success Metrics, Open Questions, Assumptions Index. Matches a solo/internal capability-spec PRD; the deliberately-omitted UJ and MVP-Scope sections are not gaps for this shape.
