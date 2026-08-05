---
baseline_commit: f16e5251964ae9780d5ebaf1ca0e23054c6dbbbd
---

# Story 10.2: Validation par exécution réelle et non-régression

Status: in-progress

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline,
I want valider par exécution réelle que `CHESS_SEARCH_TEACHER` s'entraîne de bout en bout via `Trainer` et que les domaines échecs/détection existants ne régressent pas, une fois les chunks `chess_search_teacher` disponibles,
so that l'epic se clôture sur preuve, pas sur lecture de code.

## 🛑 GARDE-FOU BLOQUANT — LIRE AVANT TOUTE EXÉCUTION

**Aymeric a explicitement demandé à valider lui-même le lancement de tout entraînement réel** (préférence confirmée en session : "je te laisse avancer [en autonomie], je validerai au moment de lancer le training" — voir aussi la préférence déjà connue "prévenir avant tout job d'entraînement/dataset complet, une exécution a déjà planté sa machine"). **Task 1 ci-dessous est un point d'arrêt obligatoire** : préparer le plan de run (chunks disponibles, durée/epochs proposés, GPU/TPU, estimation de temps) et **attendre le feu vert explicite d'Aymeric avant d'invoquer `main.py`/`Trainer`**, même pour un run "court". Ne jamais lancer d'entraînement en autonomie silencieuse sur cette story.

## Acceptance Criteria

1. **Given** les chunks `chess_search_teacher` générés à l'échelle côté `chess_ai` (précondition vérifiée avant de démarrer, pas découverte en cours de route), **When** un run réel est lancé sur `CHESS_SEARCH_TEACHER` via `Trainer`, **Then** il complète au moins une epoch sans exception — FR6
2. **Given** ce run, **When** `PolicyAccuracy` est loguée en validation, **Then** elle progresse de façon mesurable au fil des steps (pas un plateau constant proche du hasard, ~1/4672) — FR6
3. **Given** `CHESS_NO_HISTORY` (24.43% val `PolicyAccuracy` déjà mesuré, `docs/contract-chess-ai-training-interface.md`), **When** le run `CHESS_SEARCH_TEACHER` est rapporté, **Then** sa `PolicyAccuracy` finale est comparée, même approximativement, à cette référence dans le rapport de fin d'entraînement — point de comparaison informatif, pas un seuil de blocage : le critère de succès reste qualitatif (PRD §7 SM-C1 — jugement en jouant contre le modèle, comparé au comportement actuel ; pas de "rollback" formel si ça déçoit, `CHESS_SEARCH_TEACHER` est une config additive isolée qu'il suffit de ne plus utiliser)
4. **Given** `JAX_DETECTOR`/`CHESS_NO_HISTORY`/`CHESS_LEGAL_MOVES`, **When** au moins un est ré-exécuté après cette epic, **Then** son comportement est identique à avant (comparaison à une baseline) — FR5, AD-21 hérité
5. **Given** `docs/contract-chess-ai-training-interface.md`, **When** cette story conclut l'epic, **Then** il documente `CHESS_SEARCH_TEACHER` (nom de config, réutilisation `chess_policy_value`, statut `value_head_trained`) et la copie est reportée côté `chess_ai/docs/` — FR7
6. **Given** les Stories 10.1 et 10.2 complétées, **When** cette story conclut l'Epic 10, **Then** FR1 à FR7 sont confirmés couverts et `CHESS_SEARCH_TEACHER` est entraînable de bout en bout via `Trainer`

## Tasks / Subtasks

- [x] **Task 1 (BLOQUANT — décision Aymeric requise, AC: 1)** : Préparer et faire valider le plan de run avant tout lancement
  - [x] Vérifier l'état réel des chunks `chess_search_teacher` (`{DATA_ROOT}/chunks/chess_search_teacher/*.npz`) — au 2026-08-04 : **2 chunks réels, 10 000 positions au total** (voir Dev Notes).
  - [x] Présenter à Aymeric : nombre de chunks/positions disponibles, proposition de run (court vs complet), GPU vs TPU, estimation de durée
  - [x] **Confirmation explicite obtenue (2026-08-04)** : procéder avec les 10 000 positions actuelles (pas d'attente de génération supplémentaire côté `chess_ai`), run complet `epochs=15` (pas de run court), GPU local en premier choix — bascule vers Colab/TPU si le GPU local plante. Décision documentée dans Dev Agent Record.

- [x] Task 2: Exécuter le run réel et vérifier AC1/AC2 (AC: 1, 2)
  - [x] Lancé `python3 main.py CHESS_SEARCH_TEACHER` (GPU local, backend auto-détecté)
  - [x] 15/15 epochs complétées sans exception (~1 min au total, modèle 382K paramètres, 19 steps/epoch sur 5000 exemples train)
  - [x] `PolicyAccuracy` (val) progresse mesurablement : 0.0000 (epoch 1) → best 0.0183 (epoch 10), très au-dessus du hasard (~1/4672 = 0.000214, soit ~85x). Train continue de monter jusqu'à 0.0382 (epoch 15) pendant que val plafonne/redescend légèrement à partir de l'epoch 10 — signal de sur-apprentissage classique sur un train set de seulement 5000 exemples (voir Dev Notes, split forcé 50/50 par chunk), pas une anomalie du code.

- [x] Task 3: Rapporter la comparaison informative à `CHESS_NO_HISTORY` (AC: 3)
  - [x] `CHESS_SEARCH_TEACHER` : best val `PolicyAccuracy` = **1.83%**. `CHESS_NO_HISTORY` (référence contrat) : **24.43%**. Écart important — explication la plus probable : volume de données (5000 exemples train ici vs un jeu PGN bien plus large pour `CHESS_NO_HISTORY`), pas nécessairement un problème d'approche. Ce chiffre est rapporté à titre informatif, voir Dev Agent Record pour le détail.
  - [x] Chiffre présenté comme point de comparaison, pas comme critère de blocage — conforme au PRD §7 SM-C1 (jugement qualitatif)

- [ ] Task 4: Non-régression par exécution réelle (AC: 4)
  - [ ] Choisir un domaine existant à ré-exécuter (`CHESS_NO_HISTORY` le plus proche techniquement, sinon `CHESS_LEGAL_MOVES`/`JAX_DETECTOR`) — **également soumis au garde-fou d'exécution lourde : proposer le choix et la portée (inférence rapide vs entraînement complet) à Aymeric avant de lancer**, même hors du run principal de Task 1/2
  - [ ] Comparer le comportement obtenu à un état de référence antérieur à cette epic (baseline déjà établie lors des epics précédentes si disponible, sinon comportement documenté dans la story/rétro de l'epic d'origine du domaine choisi)
  - [ ] Documenter tout écart — aucun écart attendu (Story 10.1 n'a touché que `ChessPolicyValueDataset`/`dataset_configs.py`, confirmé par `git diff` en revue de code)

- [ ] Task 5: Synchroniser le contrat d'interface (AC: 5)
  - [ ] Mettre à jour `docs/contract-chess-ai-training-interface.md` (statut, date) pour documenter `CHESS_SEARCH_TEACHER` : nom de config, réutilisation de `chess_policy_value`/`chess_cnn_attention_policy_value`, statut `value_head_trained` (FR4, Story 10.1)
  - [ ] Reporter la copie identique côté `chess_ai/docs/contract-chess-ai-training-interface.md` (process déjà établi §4 du contrat — pas une nouveauté de cette story)

- [ ] Task 6: Clôture de l'Epic 10 (AC: 6)
  - [ ] Confirmer FR1 à FR7 couverts (déjà vérifié individuellement par Story 10.1 pour FR1-FR4, par cette story pour FR5-FR7)
  - [ ] Mettre à jour `epics.md`/`sprint-status.yaml` en conséquence (`epic-10: done` une fois cette story `done`)

## Dev Notes

- **Chunks réels au 2026-08-04** : `/home/aobled/Documents/data/chunks/chess_search_teacher/` contient `chess_search_teacher_chunk0.npz` et `chess_search_teacher_chunk1.npz`, 5000 positions chacun (10 000 au total), clés `position`(8,8,29)/`policy` uniquement — conforme au contrat §2.6. **Ce nombre était probablement un test/génération manuelle plutôt qu'une génération "à l'échelle" planifiée** — à reconfirmer avec Aymeric à Task 1, ne pas supposer que c'est suffisant sans son avis (l'epic/le PRD ne chiffrent volontairement pas "à l'échelle").
- **Pas de flag CLI pour un run court** : `main.py` ne prend que `dataset_name` en argument (`sys.argv[1]`, lignes 250-256) — `epochs`/`patience` viennent uniquement de `dataset_configs.py`. Si Task 1 aboutit à la décision de faire un run plus court que la config actuelle (`epochs=15`, héritée telle quelle de `CHESS_NO_HISTORY`, Story 10.1), la manière la plus propre est de modifier temporairement `epochs` dans l'entrée `CHESS_SEARCH_TEACHER` (`dataset_configs.py`) pour ce run de validation, puis de la remettre à sa valeur de production — décision et méthode exacte à discuter avec Aymeric à Task 1, ne pas décider seul.
- **`patience=8`** (early stopping) signifie qu'un run peut s'arrêter avant `epochs=15` de toute façon si `PolicyAccuracy` plafonne — ça peut suffire à couvrir AC1/AC2 sans modification de config, à évaluer selon le temps que Task 1 alloue.
- **Aucune préférence GPU/TPU tranchée pour ce dataset** — mémoire projet existante : GPU a légèrement surpassé TPU pour des runs de validation rapide type CIFAR10, mais ce n'est pas transposé automatiquement ici (dataset/modèle différents) ; laisser Aymeric trancher à Task 1.
- **Story 10.1 (précédente, terminée)** : `CHESS_SEARCH_TEACHER` (config), tolérance `ChessPolicyValueDataset`, et `value_head_trained` sont tous en place et testés (5 smoke-tests, `tests/test_chess_search_teacher_loader.py`) — cette story n'a **aucune** dépendance de code restante sur 10.1, seulement une dépendance de données/exécution (chunks + lancement réel).
- **Non-régression (Task 4)** : la revue de code de Story 10.1 a confirmé par `git diff` qu'aucune ligne des `TaskStrategy`/loaders autres que `ChessPolicyValueDataset.create_tf_dataset` n'a changé — le risque de régression réelle est donc faible, mais AC4 exige une preuve par exécution, pas seulement cette lecture de code.
- **Portée stricte** : ne pas toucher à `ChessPolicyValueStrategy`/au modèle — rien à modifier côté code pour cette story, seulement exécuter, observer, documenter.

### Project Structure Notes

- Aucun nouveau fichier de code attendu. Fichiers modifiés potentiels : `docs/contract-chess-ai-training-interface.md` (Task 5), `chess_ai/docs/contract-chess-ai-training-interface.md` (copie, autre repo), `_bmad-output/planning-artifacts/epics.md`/`sprint-status.yaml` (Task 6, tracking).
- Checkpoints produits (`checkpoints_chess_search_teacher/`, `save_dir` de la config) — nouveaux artefacts binaires, pas du code.

### References

- [Source: main.py:53, 250-256 (point d'entrée, pas de flag CLI epochs)]
- [Source: dataset_configs.py — entrée CHESS_SEARCH_TEACHER (Story 10.1), epochs=15/patience=8]
- [Source: task_strategies.py:469-538 (ChessPolicyValueStrategy, generate_reports/primary_metric_name)]
- [Source: docs/contract-chess-ai-training-interface.md §2.3/§2.6/§4 (process de sync déjà établi)]
- [Source: _bmad-output/implementation-artifacts/10-1-configuration-et-tolerance-du-chargeur-pour-le-dataset-professeur.md (story précédente, status done, Dev Agent Record)]
- [Source: _bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-08-04/prd.md §7 (SM-C1, critère qualitatif)]
- [Source: _bmad-output/planning-artifacts/epics.md, section "### Epic 10" (Story 10.2, Guardrails identifiés pré-mortem)]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

- 2026-08-04 : Décision Aymeric (Task 1) : run complet `epochs=15` sur les 10 000 positions actuelles (2 chunks `chess_search_teacher`), GPU local en premier choix, bascule Colab/TPU en cas de plantage. Lancement de `python3 main.py CHESS_SEARCH_TEACHER` en arrière-plan, logs suivis.
- 2026-08-04 : Run terminé avec succès, GPU local (pas eu besoin de basculer sur Colab). 15/15 epochs, ~1 minute. `ChessPolicyValueDataset` avec `val_split=0.1` sur seulement 2 chunks force un split 50/50 par chunk (1 train/1 val, 5000 exemples chacun) — comportement documenté du loader (Story 9.3), pas un bug de cette story, mais ça explique en grande partie le score modeste (peu de données train).
- 2026-08-04 : `decay_steps` automatique a échoué à trouver les chunks ("aucun chunk .npz trouvé"), repli sur la valeur de config (36700) — sans impact sur ce run court (15 epochs très en dessous de ce nombre de steps), mais pré-existant et hors scope de cette story (pas introduit par Story 10.1/10.2).
- 2026-08-04 : Résultat chiffré : best val `PolicyAccuracy`=1.83% (`CHESS_SEARCH_TEACHER`) vs 24.43% (`CHESS_NO_HISTORY`, référence contrat) — écart attribué au volume de données (5000 exemples train ici), pas encore à l'approche elle-même. Signal de sur-apprentissage visible dès l'epoch 10-11 (train continue de monter, val plafonne/redescend) — cohérent avec un train set restreint.
- **Task 4 (non-régression) : en attente de la décision d'Aymeric sur le domaine et la portée** (garde-fou d'exécution lourde de la story) — pas lancé automatiquement.

### File List
