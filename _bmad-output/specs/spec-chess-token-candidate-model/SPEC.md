---
id: SPEC-chess-token-candidate-model
companions:
  - ../spec-chess-token-candidate-poc/token-candidate-dataset-schema.md
  - ../../planning-artifacts/chess-cnn-attention-policy-value.mmd
sources:
  - ../spec-chess-token-candidate-poc/SPEC.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability only — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Tronc auto-attention + tête de scoring candidats — spike temps 1 (modèle & entraînement)

## Why

`chess_ai` a spécifié et livré côté données (`spec-chess-token-candidate-poc`) le "temps 1" d'une refonte architecture échecs : encodage tokenisé de la position (64 cases) et dataset de candidats légaux, en réponse à deux constats mesurés (78% des 382k paramètres du modèle actuel enfouis dans une tête `Dense(4672)` non masquée à l'entraînement ; tronc CNN à champ réceptif local 11×11 pour un jeu où les menaces sont souvent distantes). `chess_ai` exclut explicitement de son propre périmètre tout ce qui touche au modèle, à sa configuration et à sa loss (`contract-chess-ai-training-interface.md` §4) — ce sont les décisions que cette épic `jax_supervised_training` tranche : construire le tronc auto-attention et la tête de scoring candidats, les intégrer à `dataset_configs.py`, formaliser la loss, et mesurer les résultats. Décision explicite d'Aymeric (2026-08-13) : ce spike repart à neuf sur la config bottleneck par défaut (K=8/token_dim=64) plutôt que de chercher une comparabilité stricte à variable unique avec la baseline actuelle — les mesures face à la baseline restent utiles comme référence, pas comme verdict pass/fail.

## Capabilities

- **CAP-1**
  - **intent:** Un modèle dont le tronc est de l'auto-attention pure sur les 64 tokens-case (remplace le tronc CNN existant), dont le bottleneck (cross-attention + auto-attention, `K=8`/`token_dim=64`) est repris tel quel de `model_library.py`, et dont la tête de sortie score les 50 candidats légaux fournis par le dataset au lieu d'une `Dense(4672)` fixe. Les embeddings de token (`nn.Embed(13, 64)`) et de position (`nn.Embed(64, 64)`) alimentent directement le tronc à `token_dim=64`, sans projection intermédiaire vers le bottleneck.
  - **success:** Le modèle s'entraîne et converge sans erreur sur le dataset spike (`chess_token_candidate_spike.npz`).

- **CAP-2**
  - **intent:** Une loss qui score les candidats plutôt que de classifier sur l'espace fixe 4672 — cross-entropy restreinte aux 50 slots candidats réels (`candidate_mask=1`), jamais aux slots de padding, label = `candidate_label` (index de slot dans `[0,50)`, pas un index `move_to_index`).
  - **success:** Sur un batch réel, la loss se calcule sans NaN/Inf ; sur un exemple synthétique à `candidate_mask` et label connus, la valeur calculée correspond à la valeur attendue à la main.

- **CAP-3**
  - **intent:** Une entrée `dataset_configs.py` nommée `CHESS_TOKEN` qui charge le dataset spike (`token_position`, `global_flags`, `candidate_moves`, `candidate_mask`, `candidate_label`, voir companion) et l'assemble avec le modèle de CAP-1/CAP-2.
  - **success:** La config se charge, et un premier batch passe dans le `Trainer` sans erreur.

- **CAP-4**
  - **intent:** Mesurer la `PolicyAccuracy` top-1 du nouveau modèle comme référence, à côté du checkpoint professeur actuel (`best_model_chess_search_teacher.pkl`, 28,00%) — mesure informationnelle, pas une comparaison contrôlée (voir Why : le bottleneck diffère, K=8/token_dim=64 vs K=16/token_dim=196).
  - **success:** Chiffre mesuré et documenté, à parité de volume de dataset et d'epochs avec le run baseline.

- **CAP-5**
  - **intent:** Mesurer le nombre de paramètres du nouveau modèle, avec la part imputable à la tête de scoring candidats (CAP-1) isolée explicitement — mesure de référence, pas une comparaison contrôlée contre la baseline (même réserve que CAP-4).
  - **success:** Compte de paramètres mesuré fraîchement (pas recopié d'une documentation) et documenté.

- **CAP-6**
  - **intent:** Surveiller la convergence train/val du tronc auto-attention pure, avec arrêt anticipé standard (patience sur le nombre d'epochs configuré, même mécanisme que les autres configs échecs, ex. `patience=8`) — pas de détection sur mesure d'un décrochage contre une baseline.
  - **success:** Courbes train/val documentées ; l'entraînement s'arrête au budget patience/epochs configuré comme tout autre run du domaine échecs.

## Constraints

- `chess_ai` ne définit ni n'entraîne jamais de modèle (`contract-chess-ai-training-interface.md` §4) — cette épic possède entièrement le tronc, le bottleneck, la tête de scoring, la config `dataset_configs.py` et la loss ; `chess_ai` ne fournit que le dataset figé.
- Dataset consommé tel quel (companion adopté, statut SPIKE/PROVISOIRE) : fichier unique `.npz`, `token_position` relatif au joueur au trait, `candidate_moves` jamais réordonné (ordre `board.legal_moves` naturel), padding sentinel `-1`, `MAX_CANDIDATES=50` (positions au-delà déjà filtrées côté `chess_ai`) — le masquage/loss côté modèle respecte cet ordre et ce padding tels quels, sans réordonnancement supplémentaire.
- Représentation d'un coup candidat **décomposée**, jamais via une table `nn.Embed(NUM_MOVES=4672, -)` : `move_to_index` encode `index = from_square × 73 + move_type` (`chess_ai/chess_target_encoding.py:243`, `move_type ∈ [0,73)` — 0-63 type direction/distance dame, 64-72 sous-promotions). `from_square = index // 73` réutilise la même table d'embedding positionnel `(64, 64)` déjà dans le tronc (poids partagés, 0 paramètre supplémentaire). `move_type = index % 73` utilise une petite table dédiée `nn.Embed(73, 32)` (2 336 paramètres), indépendante de 4672. Une table indexée sur l'espace complet des coups réintroduirait exactement la structure que CAP-5 cherche à éviter, déplacée côté entrée.
- Vocabulaire de coup = `move_to_index`/`index_to_move` existant (`NUM_MOVES=4672`, AD-18 hérité) — aucun nouveau schéma d'encodage de coup, aucune réimplémentation indépendante ; seule sa décomposition structurelle (`from_square`, `move_type`) est exploitée, jamais recalculée autrement que par division/modulo sur l'index existant.
- Dataset spike hors contrat stable `contract-chess-ai-training-interface.md` §2.1 tant que non concluant — la nouvelle entrée `dataset_configs.py` (`CHESS_TOKEN`) n'est pas committée comme config stable au même titre que `CHESS_SEARCH_TEACHER` avant verdict CAP-4/5/6.
- Aucune compatibilité de checkpoint avec l'outillage self-play existant de `chess_ai` (`tournament_model_vs_model.py`) n'est requise pour ce spike — le format de sortie (scores sur 50 candidats) diffère fondamentalement du vecteur policy fixe `(B,4672)` actuel, aucune rétrocompatibilité avec les `.pkl` existants n'est demandée.

## Non-goals

- Modification ou régénération du dataset chess_ai — propriété de `chess_ai`, cette épic consomme le `.npz` existant tel quel.
- Biais géométrique explicite dans l'embedding positionnel — réserve future, pas ce spike.
- Comparaison contrôlée à variable unique contre la baseline CNN actuelle (28,00%, K=16/token_dim=196) — décision explicite de repartir sur la config par défaut K=8/token_dim=64 (voir Why), CAP-4/CAP-5 restent des mesures de référence.
- Détection automatique de décrochage vs baseline avec bascule vers un tronc conv-stem léger (piste du SPEC.md chess_ai source) — remplacée par l'arrêt anticipé standard de CAP-6 ; reste une piste de repli documentée, non implémentée ce spike.
- Adaptation de l'outillage self-play `chess_ai` (`tournament_model_vs_model.py`) au nouveau format de checkpoint — travail futur séparé côté `chess_ai`, hors périmètre de cette épic.
- Entraînement génétique/self-play avec sanction sur la répétition (temps 2) — hors scope, une pénalité de répétition en fitness a déjà été rejetée deux fois dans un POC précédent, à réexaminer plus tard sans reconduire automatiquement.
- Intégration au contrat stable `contract-chess-ai-training-interface.md` §2.1 à ce stade.

## Success signal

Le spike (temps 1) produit un modèle qui s'entraîne sans erreur (CAP-1), une loss correcte (CAP-2), une config intégrée (CAP-3), et des mesures documentées d'accuracy (CAP-4) et de paramètres (CAP-5) — lues comme référence, pas comme verdict pass/fail contre la baseline 28,00% (comparaison non contrôlée, voir Why). L'entraînement respecte le budget patience/epochs standard (CAP-6). Le taux de répétition en self-play (mesure de référence côté `chess_ai`, hors exécution de cette épic) n'est pas un critère de succès ou d'échec.

## Assumptions

- Le risque de data-hunger connu de l'attention pure sans biais de localité (ViT vs CNN à petite échelle) est accepté sans garde-fou de comparaison contrôlée (CAP-6 utilise l'arrêt anticipé standard, pas une détection de décrochage vs baseline).
- Le dataset `.npz` déjà généré et vérifié exhaustivement côté `chess_ai` (412 574 positions, run du 2026-08-13, vérifié directement : `token_position` (412574,64) int32 ∈[0,12], `candidate_moves` (412574,50) int32 ∈[-1,4660], `candidate_label` (412574,) int32 ∈[0,49]) est la source de vérité pour cette épic — pas de nouvelle génération prévue.
