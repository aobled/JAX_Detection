---
id: SPEC-chess-bottleneck-genetic-poc
companions: []
sources: []
---

> **Canonical contract.** This SPEC is the complete, preservation-validated contract for what to build, test, and validate. Source documents are for traceability only.

# POC — algorithme génétique sur les bottleneck_queries du modèle échecs (chess_ai)

## Why

**Vision à explorer, cheap avant d'investir.** Le modèle `ChessCnnAttentionPolicyValue` (`jax_supervised_training/model_library.py:837`, checkpoint `best_model_chess_search_teacher.pkl`, 28.00% val PolicyAccuracy) contient un paramètre appris de 3072 floats (`bottleneck_queries`, (16, 192)) qui fonctionne comme un jeu de 16 "requêtes" interrogeant l'échiquier par cross-attention — la seule partie du modèle dont l'ordre interne est arbitraire (permutation-invariant en aval) et qui pourrait raisonnablement porter quelque chose comme un "style de jeu" appris. Aymeric veut tester si figer tout le reste du modèle (backbone, projections d'attention, têtes) et ne faire évoluer que ces 3072 floats par algorithme génétique — tournoi entre variantes, sélection, mutation/croisement — est une approche exploitable, après des tentatives précédentes d'algo génétique sur l'espace complet des paramètres (~1,48M) qui avaient toutes échoué. Réduire drastiquement la dimensionnalité du génome est le levier central de cette relance.

Ce spec documente uniquement la conception du POC (couche par couche : représentation d'un individu, opérateurs, évaluation, critère d'arrêt) — décidée en session avec Winston le 2026-08-09 — pas encore son implémentation. Il vit temporairement dans `jax_supervised_training` (convention de session) mais concerne exclusivement `chess_ai` ; Aymeric le déplacera lui-même vers ce repo.

## Capabilities

- **CAP-1**
  - **intent:** Représenter un individu de la population comme un checkpoint standard, identique au checkpoint de base sauf `params["bottleneck_queries"]`.
  - **success:** Le checkpoint généré se charge et s'exécute sans aucune modification via `load_chess_model`/`select_model_move` (`chess_model_inference.py`, chess_ai).

- **CAP-2**
  - **intent:** Identifier chaque individu par un hash de contenu déterministe de son `bottleneck_queries`.
  - **success:** `sha256(bytes)[:12]` sur le tableau casté en `float32` C-contiguous ; deux individus au même génome produisent le même ID, permettant de sauter la réévaluation d'un génome déjà rencontré dans une génération antérieure.

- **CAP-3**
  - **intent:** Initialiser la génération 0 à partir du checkpoint entraîné actuel plutôt que d'une population aléatoire.
  - **success:** Population de 20 = le checkpoint actuel intact + 19 mutations gaussiennes de celui-ci ; les 20 se chargent sans erreur.

- **CAP-4**
  - **intent:** Faire varier `bottleneck_queries` par mutation gaussienne additive dont l'amplitude est un paramètre ajustable.
  - **success:** `mutation_sigma` par défaut = 0.01 (≈11% de l'écart-type réel mesuré, 0.0877, dans `best_model_chess_search_teacher.pkl`) ; changer sa valeur ne nécessite aucune modification de code.

- **CAP-5**
  - **intent:** Classer les individus d'une génération par performance de jeu réelle plutôt que par heuristique proxy.
  - **success:** Score cumulé au système points échecs (victoire=1, nulle=0.5, défaite=0) sur les résultats du round-robin ; les 10 meilleurs scores sont conservés.

- **CAP-6**
  - **intent:** Générer de nouveaux individus par recombinaison de deux parents survivants.
  - **success:** Crossover uniforme ligne-à-ligne — chaque ligne du tableau (16, 192) tirée 50/50 chez l'un des deux parents, indépendamment par ligne ; jamais de découpage par segment contigu ni par float individuel.

- **CAP-7**
  - **intent:** Évaluer la fitness de la population par confrontation directe entre toutes les paires d'individus.
  - **success:** Round-robin N-way construit au-dessus de `tournament_model_vs_model.py` (`play_one_game`/`random_opening_moves` réutilisés sans modification) ; `games_per_pair`=4 par défaut (2 ouvertures × 2 couleurs) × 190 paires (C(20,2)) = 760 parties/génération, `games_per_pair` paramétrable.

- **CAP-8**
  - **intent:** Départager les individus quand une majorité des parties d'une génération sont nulles.
  - **success:** Score Sonneborn-Berger (pondéré par la force des adversaires rencontrés) calculé en post-traitement des résultats bruts du round-robin, sans infrastructure supplémentaire.

- **CAP-9**
  - **intent:** Valider, avant d'investir dans la boucle complète, l'hypothèse que `bottleneck_queries` porte assez le "style de jeu" pour qu'une évolution ait un effet mesurable.
  - **success:** Un mini-pilote de quelques générations mesure le taux de parties décisives à `mutation_sigma`=0.01 ; un taux proche de 0% déclenche soit une augmentation de sigma, soit la remise en cause de l'approche — trancher avant tout run à 20 individus sur plusieurs générations.

- **CAP-10**
  - **intent:** Arrêter la boucle évolutive selon un critère explicite plutôt qu'indéfiniment.
  - **success:** `n_generations` paramétrable ; chaque génération journalise les scores, les IDs (hash) et le meilleur individu.

## Constraints

- Ne jamais modifier `chess_model_inference.py::select_model_move` ni `tournament_model_vs_model.py::play_one_game`/`random_opening_moves` — réutilisation stricte (AD-18 chess_ai, source unique de vérité pour légalité/sélection de coup).
- `games_per_pair` doit rester un nombre pair (contrainte héritée : `n_pairs = games // 2` dans le script de tournoi existant).
- L'ID d'un individu est un hash de contenu (sha256 tronqué), jamais un compteur ou UUID arbitraire — condition de la dédup CAP-2.
- La sélection utilise le score cumulé au système points échecs, jamais un compte brut de victoires — sensible aux nulles qui domineront probablement les premières générations.
- La tête value du checkpoint ne doit jamais servir de signal de départage : `loss_params.value_weight=0.0` et `value_head_trained=False` dans `best_model_chess_search_teacher.pkl` (vérifié 2026-08-09) — ses sorties sont du bruit d'initialisation jamais entraîné sur ce dataset.
- `mutation_sigma`, `games_per_pair` et `n_generations` doivent tous être des paramètres ajustables, jamais des constantes en dur.
- Le croisement (CAP-6) est ligne-à-ligne, jamais par segment contigu : le modèle est invariant à la permutation des 16 lignes (auto-attention équivariante + mean pool invariant en aval), donc aucune localité entre lignes voisines à préserver — un découpage contigu respecterait une structure qui n'existe pas.

## Non-goals

- Ne touche à aucun code de `jax_supervised_training` — training générique, hors scope.
- Ne modifie pas l'infrastructure de tournoi/inférence existante de `chess_ai` (`select_model_move`, `play_one_game`, `random_opening_moves`).
- N'entraîne pas la tête value.
- N'implémente ni ne lance le POC — ce spec documente uniquement la conception ; l'implémentation et l'exécution sont une étape ultérieure séparée, décidée par Aymeric après revue.
- Pas de round-robin complet à 20 individus avant que le mini-pilote (CAP-9) n'ait validé la décisivité au `mutation_sigma` choisi.

## Success signal

Le mini-pilote (CAP-9) produit, pour `mutation_sigma`=0.01 sur quelques générations légères, un taux de parties décisives mesurable et non nul — donnée suffisante pour décider si la boucle complète (20 individus, 760 parties/génération) mérite d'être implémentée côté `chess_ai`, ou si `mutation_sigma`/l'approche elle-même doivent être révisés avant d'aller plus loin.

## Assumptions

- La tête value du checkpoint actuel n'est pas entraînée sur `CHESS_SEARCH_TEACHER` (voir Constraints) — écartée comme signal de départage plutôt qu'adaptée.
- L'écart-type réel de `bottleneck_queries` (0.0877, mesuré directement dans le checkpoint le 2026-08-09) sert de base au choix de `mutation_sigma`=0.01, plutôt qu'une valeur arbitraire non ancrée dans les données réelles.

## Open Questions

- Format exact de journalisation par génération (JSON par génération ? CSV cumulatif ? autre ?) — non tranché, à décider à l'implémentation.
- Si Sonneborn-Berger (CAP-8) s'avère insuffisant pour départager les nulles : option de repli différée, réutiliser `centipawn_eval`/`resolve_stockfish_path` (déjà présents dans `evaluate_model_vs_stockfish.py`, chess_ai) pour évaluer qui poussait un avantage dans les parties nulles — non implémentée dans ce POC, à réévaluer seulement si le besoin se confirme.
