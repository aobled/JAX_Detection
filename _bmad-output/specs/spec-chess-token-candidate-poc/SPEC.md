---
id: SPEC-chess-token-candidate-poc
companions:
  - token-candidate-dataset-schema.md
sources:
  - ../../../docs/chess-architecture-v2-hypotheses.md
  - ../../../docs/chess_ai_global_conclusions.md
  - ../../../docs/ChessCnnAttentionPolicyValue.md
  - ../../../docs/contract-chess-ai-training-interface.md
  - ../../../chess_target_encoding.py
  - ../spec-chess-move-token-poc/SPEC.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability only — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Input tokenisé + sortie candidats légaux — spike temps 1

## Why

**Un postulat commun aux deux architectures échecs déjà testées (teacher CNN, move_token transformer) a été isolé début août 2026 (`chess_ai_global_conclusions.md`) : imitation pure d'un oracle par-position, sans signal lié à l'issue de partie — cause du blocage stratégique/répétition.** Orthogonalement à ce diagnostic (paradigme d'entraînement), une discussion du 2026-08-13 (`chess-architecture-v2-hypotheses.md`) a isolé deux pistes d'architecture concrètes, mesurées et documentées, mais jamais testées : (1) 78% des 382k paramètres du modèle actuel vivent dans une seule tête `Dense(4672)` qui doit apprendre à ignorer ~4640 classes toujours illégales par position (20-40 coups légaux réels en moyenne) — un espace de sortie fixe non masqué à l'entraînement (AD-22) ; (2) le tronc CNN actuel propage l'information localement (champ réceptif 11×11) avant toute interaction globale, un biais de localité dont la pertinence est discutable aux échecs (une pièce menace des cases distantes). Ce spec cadre le côté `chess_ai` (encodage + dataset) de la première étape ("temps 1" : apprendre au modèle à prédire les coups légaux) d'une refonte architecture qui adresse ces deux points, en gardant explicitement le paradigme d'entraînement (temps 2, hors scope) et le bottleneck K=8 existant (en pause) inchangés.

## Capabilities

- **CAP-1**
  - **intent:** (`chess_ai`) Encoder une position comme 64 tokens entiers (un par case, bibliothèque fermée de 13 valeurs : ami/ennemi × Roi/Dame/Tour/Fou/Cavalier/Pion + case vide, encodage relatif au joueur au trait) plutôt que des plans binaires 8×8×C, plus un petit vecteur de drapeaux globaux (trait, 4 droits de roque, répétition — inchangés dans leur sémantique, juste plus tokenisés en plans spatiaux).
  - **success:** Pour un échantillon de positions réelles, l'encodage tokenisé et le décodage inverse (reconstruction de la position depuis les tokens + drapeaux) sont bijectifs — round-trip vérifié.

- **CAP-2**
  - **intent:** (`chess_ai`) Générer un dataset spike **frais et seedé** (script autonome, même patron que `chess_decisive_teacher_dataset_tools.py`) contenant, par position : l'encodage tokenisé (CAP-1) et la liste des coups candidats légaux de cette position (indices `move_to_index` existants, AD-18 — aucun nouveau schéma d'encodage de coup).
  - **success:** Script exécuté de bout en bout sur un lot d'auto-jeu, `.npz` produits conformes au schéma du companion, format non intégré au pipeline `.npz` stable.

- **CAP-3**
  - **intent:** (`jax_supervised_training`, épic et session séparées — voir Constraints) Un modèle qui remplace le tronc CNN par de l'auto-attention pure sur les 64 tokens dès le premier étage, conserve le bottleneck `K=8` existant inchangé en aval, et remplace la tête `Dense(4672)` par un score sur les candidats légaux fournis par le dataset.
  - **success:** Le modèle s'entraîne et converge sans erreur sur le dataset spike (CAP-2).

- **CAP-4**
  - **intent:** Comparer la `PolicyAccuracy` top-1 du nouveau modèle au checkpoint professeur actuel (`best_model_chess_search_teacher.pkl`, 28,00%), à parité de volume de dataset et d'epochs.
  - **success:** Chiffre mesuré et documenté ; comparable ou meilleur que 28,00% constitue un signal positif (voir Success signal).

- **CAP-5**
  - **intent:** Vérifier la réduction réelle du nombre de paramètres du modèle par rapport à la baseline actuelle (382 017 paramètres, dont 299 008 dans `Dense(4672)`).
  - **success:** Compte de paramètres mesuré et documenté, avec la part imputable à l'ancienne tête `Dense(4672)` explicitement retirée.

- **CAP-6**
  - **intent:** Surveiller la convergence train/val de l'auto-attention pure pour détecter un décrochage par rapport à la baseline CNN (écart croissant façon move_token dès l'epoch 9, ou accuracy plafonnant significativement sous 28%).
  - **success:** Courbes train/val documentées ; si le décrochage est confirmé, la condition d'arrêt anticipé (Constraints) est déclenchée et documentée comme telle, pas silencieusement ignorée.

- **CAP-7**
  - **intent:** Mesurer, comme référence et non comme critère de succès/échec, le taux de parties/nulles en répétition d'un self-play du nouveau modèle contre lui-même, avec le protocole exact déjà utilisé pour les baselines existantes.
  - **success:** 100 parties, `tournament_model_vs_model.py`, ouvertures aléatoires rejouées 2× (annulation du biais de couleur) — chiffre consigné dans le tableau comparatif de `chess_ai_global_conclusions.md` (à côté de 78,0% teacher, 94,0% move_token), explicitement marqué comme point de référence, pas comme verdict de ce spike.

## Constraints

- `chess_ai` ne définit ni n'entraîne jamais de modèle (`contract-chess-ai-training-interface.md` §4) — le tronc auto-attention, le bottleneck et la tête de scoring (CAP-3) sont définis et entraînés côté `jax_supervised_training`, dans une épic et une session séparées ; confirmé explicitement avec l'utilisateur le 2026-08-13. `chess_ai` ne produit que le dataset (CAP-1/CAP-2).
- Vocabulaire de coup candidat = `move_to_index`/`index_to_move` existant (`NUM_MOVES=4672`, AD-18, "seule conversion, jamais réimplémentée indépendamment") — aucun nouveau schéma d'encodage de coup.
- Encodage token relatif au joueur au trait (ami/ennemi), jamais couleur absolue (blanc/noir) — préserve la normalisation déjà en place dans `chess_target_encoding.py`.
- Drapeaux d'état globaux (trait, 4 droits de roque, répétition) restent des scalaires hors-token — pas assez d'entrées discrètes pour justifier un embedding dédié.
- Aucune règle dynamique/dérivée (ex. roque interdit par échec courant) n'est encodée dans l'input — toujours déléguée à `python-chess` via la génération de candidats (AD-18), jamais réimplémentée côté encodage.
- `MAX_CANDIDATES=50` par position ; positions à plus de 50 coups légaux (rares, jusqu'à 218 théorique) **filtrées** (non écrites dans le dataset), jamais tronquées avec réordonnancement — un ordre truqué (ex. coup professeur toujours en tête) laisserait le modèle deviner le label par sa position plutôt que par son contenu. Voir companion pour le détail exact.
- Professeur = **Stockfish** profondeur 12 (comme `CHESS_SEARCH_TEACHER` depuis le 2026-08-07 — pas `chess_search.py`, abandonné pour ce rôle : 2,44 pos/s mono-processus, ~84h à l'échelle), pour rester comparable aux baselines déjà mesurées (28,00% accuracy, 78,0%/94,0% répétition self-play).
- Dataset spike hors du contrat stable `contract-chess-ai-training-interface.md` §2.1 tant que non concluant — tooling autonome, même patron que le spike move-token et le POC génétique bottleneck.
- Condition d'arrêt anticipé côté `jax_supervised_training` (CAP-6) : si la convergence de l'auto-attention pure décroche nettement de la baseline CNN, basculer sur un tronc convolutif léger ("conv stem") devant l'attention plutôt que d'insister sur le tout-attention — décision à documenter dans l'épic séparée, pas à ignorer silencieusement.

## Non-goals

- Définition ou entraînement du modèle — épic `jax_supervised_training` séparée.
- Intégration au contrat stable `contract-chess-ai-training-interface.md` §2.1 à ce stade.
- Régénération du dataset à pleine échelle.
- Biais géométrique explicite dans l'embedding positionnel (réserve future si l'embedding appris `(64, D)` s'avère insuffisant — pas ce spike).
- Training génétique/self-play avec sanction sur la répétition (temps 2). Rappel pour cette étape future : une pénalité de répétition en fitness a déjà été rejetée deux fois dans le POC génétique précédent (`chess_ai_global_conclusions.md` §4) — à réexaminer dans le nouveau contexte, pas reconduire automatiquement.
- Refonte du bottleneck `K=8` — en pause, dépend du résultat de ce spike.

## Success signal

Le spike (temps 1) est concluant sur ses propres termes s'il satisfait CAP-4 (accuracy comparable ou meilleure que 28,00%) et CAP-5 (réduction mesurée des paramètres) sans déclencher la condition d'arrêt anticipé de CAP-6. **Le taux de répétition en self-play (CAP-7) n'est explicitement pas un critère de succès ou d'échec de ce spike** — le modèle n'a reçu aucun signal lié à l'issue de partie à ce stade ; il est mesuré uniquement comme point de référence pour isoler, plus tard, l'effet propre d'un temps 2 (génétique/self-play) sur la répétition, sans confondre ce qui vient de l'architecture et ce qui vient du paradigme d'entraînement.

## Assumptions

- Le risque de data-hunger connu de l'attention pure sans biais de localité (ViT vs CNN à petite échelle, cf. ImageNet-1k) est accepté avec un garde-fou explicite (condition d'arrêt anticipé) plutôt qu'écarté par précaution — le plateau (64 cases) est nettement plus petit qu'une image naturelle, donc pas d'obstacle de principe démontré pour ce cas précis, seulement un risque à surveiller.
- L'embedding positionnel appris `(64, D)` libre est retenu comme option la plus simple à tester en premier ; un biais géométrique explicite reste une option de repli si le résultat déçoit, non testée ici.
- Le bottleneck `K=8` (cross-attention + auto-attention) reste structurellement inchangé pour ce spike — seule sa source d'alimentation change (tokens attention au lieu de features CNN).

## Open Questions

- Nom exact de la nouvelle entrée `dataset_configs.py` côté `jax_supervised_training` (ex. `CHESS_TOKEN_CANDIDATE`) — à fixer dans l'épic séparée, pas dans ce spec.
