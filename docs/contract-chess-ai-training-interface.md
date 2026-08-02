# Contrat d'interface : `jax_supervised_training` ↔ `chess_ai`

**Statut : mis à jour (2026-08-01), à faire évoluer des deux côtés.** Ce document n'est pas un
PRD ni une epic — c'est la surface d'interface entre deux repos indépendants qui doit
rester identique des deux côtés, contrairement au reste du code échecs qui, lui, est
dupliqué et volontairement laissé libre de diverger (voir `chess_ai/HANDOFF.md`).

**Copie canonique : TBD, mais les deux copies existent désormais** (`chess_ai/docs/`
depuis l'installation de son instance BMAD) **et doivent être maintenues manuellement en
synchro** — pas de lien symbolique, pas de génération automatique. Toute modification
de ce fichier d'un côté doit être reportée dans l'autre avant de démarrer une epic qui
en dépend.

**Correction 2026-08-01 (spec `spec-chess-npz-boundary-cleanup`, `jax_supervised_training`)** :
la répartition ci-dessous a changé depuis l'ébauche du 2026-07-31. À l'origine,
`jax_supervised_training` générait encore les `.npz` échecs et gardait une copie de
`chess_target_encoding.py` en plus de celle de `chess_ai`. Ce n'est plus le cas — voir §1.

## 1. Répartition

- **`jax_supervised_training`** : définit les modèles, les entraîne (`Trainer`/
  `TaskStrategy`/`main.py`). **Ne génère plus les `.npz` échecs**, ne compose jamais, ne
  joue jamais. Ne connaît le format `.npz` que par sa forme (shape, noms de clés) —
  aucune dépendance à `python-chess`, même indirecte, depuis le 2026-08-01.
- **`chess_ai`** : génère les `.npz` échecs (`chess_target_encoding.py` +
  `dataset_builder/chess_pgn_dataset_tools.py`, propriétaire exclusif désormais) et
  compose des checkpoints déjà entraînés (jeu, tournoi, évaluation). Ne réentraîne
  jamais, ne définit jamais d'architecture de modèle.

**Flux concret** : `chess_ai` produit des `.npz` → `jax_supervised_training` les lit,
entraîne, produit des `.pkl` (checkpoints) → `chess_ai` les charge et les compose. Si un
futur modèle (§3) change de nature (ex. sortie différente d'un policy+value combiné),
`chess_ai` peut avoir besoin d'adapter son côté consommateur (`chess_model_inference.py`
notamment) pour charger le nouveau format de checkpoint — attendu, pas une anomalie.

## 2. Ce qui est déjà stable (ne pas changer d'un côté sans changer l'autre)

### 2.1 Encodage de la position

`chess_target_encoding.py::encode_position` (**désormais côté `chess_ai` uniquement** —
`jax_supervised_training` a supprimé sa copie le 2026-08-01, voir §1) — plans 8×8×19
(position seule) ou 8×8×29 (avec historique, `include_history=True`). Le **format
produit**, lui, n'est pas négociable indépendamment : toute évolution du schéma de plans
doit être portée dans `chess_ai` (seul propriétaire du code d'encodage) **et** dans les
littéraux `dataset_configs.py` de `jax_supervised_training` (`num_classes`, `num_channels`,
`input_shape` des entrées `CHESS*`), sous peine de désynchronisation silencieuse — c'est
exactement le risque qu'AD-18 de ce repo décrit pour d'autres formats d'échange, reporté
ici au niveau des valeurs littérales plutôt que du code partagé.

### 2.2 Espace de sortie policy

`NUM_MOVES = 4672` (64 cases source × 73 types de coup, schéma AlphaZero),
`move_to_index`/`index_to_move` comme seule conversion (jamais réimplémentée
indépendamment) — **vivent uniquement côté `chess_ai`** ; `jax_supervised_training` ne
connaît que la valeur `4672` (littéral `num_classes`), jamais la logique d'encodage/décodage
d'un coup. Vaut pour le modèle combiné actuel ; à confirmer que les futurs modèles
séparés (§3) le réutilisent tel quel ou en dérivent un sous-ensemble.

### 2.3 Contrat de checkpoint

`chess_model_inference.py::load_chess_model` (côté `chess_ai`) détecte automatiquement la
variante d'entrée via un champ `num_channels` embarqué dans la config sauvegardée avec le
checkpoint — pas de nom de fichier ou de convention externe. **Tout nouveau modèle
entraîné côté `jax_supervised_training` doit suivre ce même principe** : la config
sauvegardée doit porter tout ce qui est nécessaire pour que `chess_ai` charge et utilise
le checkpoint sans supposition externe.

### 2.4 Patron de composition

`JAX_DETECTOR` (`build_single_pass_predict_fn`, détection + classification composées en
une seule passe) est le précédent architectural explicitement visé par HANDOFF.md pour
composer les 2 futurs modèles échecs. À réutiliser comme référence de conception, pas à
réinventer.

## 3. Ce qui reste ouvert (à trancher côté PRD `chess_ai`, puis à reporter ici)

- **Modèle 1 "candidats de coups"** (équivalent "detection") : `model_name`/`task_type`
  exacts, forme de sortie (distribution sur un sous-ensemble de coups ? score par coup
  légal ? top-K ?) — non tranché. **Le rôle exact du modèle lui-même est encore en débat**
  (discuté le 2026-08-01, non tranché) : `board.legal_moves` (`python-chess`) donne déjà
  gratuitement l'ensemble des coups légaux, un modèle entraîné dessus n'apporterait rien
  si "candidats" veut dire "légaux". La piste jugée plus intéressante par Aymeric :
  apprendre au modèle à *respecter les règles* comme étape avant de choisir un coup
  (le modèle "comprend" ce qu'il fait, pas juste un signal légalité redondant) — à
  pressure-tester côté `chess_ai` (`bmad-forge-idea` recommandé) avant de spécifier un
  format de dataset ici.
- **Modèle 2 "stratégie/évaluateur"** (équivalent "classification") : `model_name`/
  `task_type` exacts, forme de sortie (évaluation d'une position candidate ? classement
  de plusieurs candidats ?) — non tranché.
- **Labels/dataset requis pour chacun** : le dataset actuel (`chess_pgn_dataset_tools.py`)
  produit policy+value pour le modèle combiné existant ; les 2 nouveaux modèles
  pourraient nécessiter un format de label différent (ex. paires de positions à comparer
  pour le modèle 2) — à spécifier avant toute nouvelle entrée `dataset_configs.py`.
- **Nommage `dataset_configs.py`** : whatever convention suit `CHESS_NO_HISTORY` existant,
  à fixer une fois les 2 modèles nommés.

## 4. Process

**Le/les modèles sont gérés exclusivement ici** (`jax_supervised_training`) : définition
d'architecture (`model_library.py`), entraînement (`Trainer`/`TaskStrategy`/`main.py`).
Toute epic `chess_ai` qui nécessite une nouvelle capacité d'entraînement se traduit par
une epic **dans `jax_supervised_training`** (nouveau modèle/config/TaskStrategy), sur le
même mode que l'Epic 9 — pas par du travail direct dans ce repo depuis une session
`chess_ai`.

**Côté `chess_ai`, seuls des `.npz` sont générés et fournis en entrée** ; en retour,
`chess_ai` reçoit des `.pkl` (checkpoints), avec potentiellement des adaptations
nécessaires côté `chess_ai` (ex. `chess_model_inference.py`) pour consommer un nouveau
format de sortie — normal si le modèle change de nature (modèle unique → modèles
composés), pas un signe d'erreur de contrat.

Ce contrat est l'unique document que les deux PRD ont besoin de référencer en
commun ; il n'y a pas de PRD ou d'epic partagée entre les deux repos.

À chaque décision prise côté `chess_ai` sur §3, reporter le résultat ici (ou dans la
copie côté `chess_ai` si elle existe déjà) avant de démarrer l'epic côté
`jax_supervised_training`.
