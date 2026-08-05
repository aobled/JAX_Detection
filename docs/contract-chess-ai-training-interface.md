# Contrat d'interface : `jax_supervised_training` ↔ `chess_ai`

**Statut : mis à jour (2026-08-04), à faire évoluer des deux côtés.** Ce document n'est pas un
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

### 2.5 Dataset dédié "compréhension des coups légaux" (Modèle 1, Epic 2 `chess_ai`)

**Tranché et implémenté côté `chess_ai` (Story 2.2, 2026-08-02)** — voir §3 pour la
résolution du rôle du modèle lui-même ; ce qui suit est le contrat de données concret.

- **Clé `.npz` et label** : `LEGAL_MASK_KEY = "legal_mask"` (constante dans
  `chess_target_encoding.py`, à réutiliser telle quelle, jamais un littéral choisi
  localement) — masque multi-label `int8`, shape `(NUM_MOVES,)` = `(4672,)`, 1 par coup
  légal de la position, 0 sinon. Réutilise l'espace d'action §2.2 tel quel (pas un espace
  réduit ou différent). Aucun `VALUE_KEY` n'est écrit pour ce dataset.
- **Encodage position** : `encode_position(..., include_legal_hint=False)` — variante qui
  remplace le plan "cases de destination des coups légaux" (plan 18) par un plan dédié
  "case cible de la prise en passant", pour ne pas donner la réponse en entrée d'un modèle
  censé apprendre la légalité. **Les constantes de plan pour cette variante,
  `NUM_POSITION_PLANES_NO_HINT` et `NUM_PLANES_NO_HINT`, valent 19 et 29 —
  numériquement identiques aux `NUM_POSITION_PLANES`/`NUM_PLANES` existants (§2.1), ce
  n'est pas une erreur ni un shape réduit.** Les deux variantes ont donc le même
  `num_channels` que le dataset policy+value existant à `include_history` égal — ce qui
  les distingue, c'est le **contenu** sémantique du plan 18 (indice légalité vs. case de
  prise en passant), jamais sa position ni la forme du tableau. Ne pas supposer un
  `num_channels` différent côté `jax_supervised_training` pour ce dataset.
- **Sortie de fichiers** : préfixe dédié (`chess_legal_moves`), jamais le préfixe `chess`
  existant — aucune collision avec le dataset policy+value (AD-4 côté `chess_ai`).

### 2.6 Dataset dédié "distillation depuis la recherche classique" (Epic 3 `chess_ai`)

**Implémenté côté `chess_ai` (Story 3.1, 2026-08-04)** — voir §3 pour le contexte de
décision (pourquoi cette direction plutôt que la pipeline composée §2.4/§3) ; ce qui suit
est le contrat de données concret.

- **Clé `.npz` et label** : `POLICY_KEY = "policy"` (constante existante, réutilisée
  telle quelle — pas un nouveau littéral) — `int32`, un index dans l'espace d'action
  existant §2.2 (`NUM_MOVES=4672`), le coup choisi par `chess_search.py::select_search_move`
  (recherche alpha-bêta + évaluateur matériel, profondeur 4, aucun réseau de neurones,
  jamais appelé à l'inférence — seulement à la génération de ce dataset). Aucun
  `VALUE_KEY` n'est écrit pour ce dataset (positions d'auto-jeu, pas d'issue de partie
  complétée naturellement à leur associer).
- **Encodage position** : chemin par défaut **inchangé** (`include_legal_hint=True`,
  `NUM_POSITION_PLANES`/`NUM_PLANES` existants, §2.1) — **aucune nouvelle variante de
  plans pour ce dataset**, contrairement à §2.5.
- **Sortie de fichiers** : préfixe dédié (`chess_search_teacher`), distinct des deux
  préfixes existants (`chess`, `chess_legal_moves`) — aucune collision (AD-4 côté
  `chess_ai`).

## 3. Ce qui reste ouvert (à trancher côté PRD `chess_ai`, puis à reporter ici)

**Direction opérationnelle actuelle (Epic 3 `chess_ai`, 2026-08-03/04) — supersède la
pipeline composée ci-dessous pour le court terme :** plutôt que de composer Modèle 1
(légalité) + Modèle 2 (stratégie), la piste retenue est un **modèle unique**, entraîné par
distillation depuis `chess_search.py` (recherche classique, professeur à l'entraînement
uniquement, jamais à l'inférence — principe AlphaZero) — voir §2.6 pour le contrat de
données, et `chess_ai/_bmad-output/planning-artifacts/brief-model-without-search-2026-08-03.md`
pour le raisonnement complet (pourquoi pas de composition avec Modèle 1 — légalité déjà
native via `python-chess`/`include_legal_hint`, §2.1/§2.5 — et pourquoi pas un algorithme
génétique). La pipeline composée (Modèle 1 + Modèle 2) ci-dessous reste **suspendue, pas
abandonnée** — à reconsidérer si les résultats du modèle unique déçoivent. Les bullets
Modèle 1/Modèle 2 qui suivent restent factuellement exacts (le rôle de Modèle 1 reste
tranché tel que documenté ; celui de Modèle 2 reste réellement non tranché) — seule la
priorité/direction change.

- **Modèle 1 "compréhension des coups légaux"** (anciennement "candidats de coups") :
  **rôle tranché (PRD `chess_ai` du 2026-08-02, epic "Legal-Moves-Understanding
  Dataset")** — ce n'est pas un modèle de "candidats" (qui serait redondant avec
  `board.legal_moves`), c'est un modèle entraîné à *apprendre* la légalité d'un coup à
  partir d'une position, sans recevoir l'indice légal en entrée (voir §2.5) — la brique de
  base d'un futur pipeline composé (position → modèle légalité → modèle stratégie),
  toujours pas scopée elle-même ici (Temps 3, non commencé). Forme de sortie : un
  multi-label `(4672,)` (une probabilité de légalité par coup de l'espace d'action §2.2),
  pas un top-K ni une distribution normalisée — c'est directement dérivé du format
  `legal_mask` §2.5. `model_name`/`task_type` exacts côté `dataset_configs.py` restent à
  fixer (voir bullet nommage ci-dessous), mais la nature du modèle et son dataset ne sont
  plus en débat.
- **Modèle 2 "stratégie/évaluateur"** (équivalent "classification") : `model_name`/
  `task_type` exacts, forme de sortie (évaluation d'une position candidate ? classement
  de plusieurs candidats ?) — **toujours non tranché**, hors scope de l'epic qui a résolu
  le Modèle 1.
- **Labels/dataset requis pour le Modèle 2** : le dataset actuel
  (`chess_pgn_dataset_tools.py`) produit policy+value pour le modèle combiné existant ; le
  Modèle 1 a désormais son propre dataset (§2.5) ; le Modèle 2 pourrait nécessiter un
  format de label différent (ex. paires de positions à comparer) — toujours à spécifier
  avant toute nouvelle entrée `dataset_configs.py` pour ce modèle.
- **Nommage `dataset_configs.py`** : whatever convention suit `CHESS`/`CHESS_NO_HISTORY`/
  `CHESS_NAKAMURA_NO_HISTORY` existants, à fixer une fois les 2 modèles nommés (le
  Modèle 1 a maintenant un dataset stable à référencer ; le nom d'entrée
  `dataset_configs.py` lui-même reste une décision côté `jax_supervised_training`).

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
copie côté `jax_supervised_training` si elle existe déjà) avant de démarrer l'epic côté
`jax_supervised_training`.
