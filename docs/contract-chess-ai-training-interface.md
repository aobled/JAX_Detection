# Contrat d'interface : `jax_supervised_training` ↔ `chess_ai`

**Statut : mis à jour (2026-08-16), à faire évoluer des deux côtés.** Ce document n'est pas un
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
  (recherche alpha-bêta + évaluateur matériel, aucun réseau de neurones, jamais appelé à
  l'inférence — seulement à la génération de ce dataset). Aucun `VALUE_KEY` n'est écrit
  pour ce dataset (positions d'auto-jeu, pas d'issue de partie complétée naturellement à
  leur associer).
  - **Profondeur de recherche du professeur : pas une valeur figée par ce contrat, côté
    `chess_ai` uniquement.** Documentée à "profondeur 4" à l'écriture initiale de cette
    section (2026-08-04) ; passée à **profondeur 12** depuis (2026-08-07, décision
    `chess_ai` en revue de code — une profondeur trop proche de celle de l'auto-jeu
    rendrait le professeur redondant avec le coup déjà joué, videtant le signal de
    distillation). N'affecte que la *qualité* du label, jamais le schéma `.npz`
    (clé/shape/dtype inchangés) — ne pas supposer une profondeur particulière côté
    `jax_supervised_training`, qui n'en a de toute façon aucune connaissance directe.
- **Encodage position** : chemin par défaut **inchangé** (`include_legal_hint=True`,
  `NUM_POSITION_PLANES`/`NUM_PLANES` existants, §2.1) — **aucune nouvelle variante de
  plans pour ce dataset**, contrairement à §2.5.
- **Sortie de fichiers** : préfixe dédié (`chess_search_teacher`), distinct des deux
  préfixes existants (`chess`, `chess_legal_moves`) — aucune collision (AD-4 côté
  `chess_ai`).
- **Statut côté `jax_supervised_training` (Epic 10, confirmé par exécution réelle,
  2026-08-04 à 2026-08-09)** : entrée `CHESS_SEARCH_TEACHER` dans `dataset_configs.py`,
  réutilise à l'identique `task_type="chess_policy_value"` et
  `model_name="chess_cnn_attention_policy_value"` (Epic 9, aucun nouveau modèle) — seule
  la tête value est neutralisée par pondération (`loss_params.value_weight=0.0`, jamais
  supprimée), `value_head_trained=False` dérivé automatiquement de cette valeur. Entraîné
  de bout en bout via `Trainer` sans erreur, `PolicyAccuracy` progresse mesurablement.
  Meilleur résultat obtenu (10 000 parties/1 402 252 positions, profondeur 12,
  `token_dim=192`/`dropout=0.35`/`label_smoothing=0.2`) : **28.00% val `PolicyAccuracy`**
  — dépasse la référence `CHESS_NO_HISTORY` (24.43%, §2.1). Détail complet de la campagne
  de tuning : `_bmad-output/implementation-artifacts/chess-search-teacher-strategy.md`
  (`jax_supervised_training`).

### 2.7 Dataset dédié "professeur décisif" (spec `spec-decisive-teacher-dataset`, `chess_ai`)

**Implémenté côté `chess_ai` (2026-08-11/12)**, dataset généré mais **pas encore
entraîné côté `jax_supervised_training`** au moment de l'écriture de cette section — voir
`_bmad-output/implementation-artifacts/spec-decisive-teacher-dataset.md` pour le contexte
de décision complet (variante de §2.6, ne garde qu'un sous-ensemble filtré des positions).

- **Clé `.npz` et label** : `POLICY_KEY` seul (comme §2.6), coup choisi par un second
  moteur Stockfish "professeur" (`teacher_depth=12`, même profondeur que §2.6) — mais une
  position n'est **gardée** que si ce coup produit un vrai saut de valeur par rapport à
  `chess_search.py::material_balance` (delta ≥ `min_delta_cp=150` centipawns, ou mat
  délivré en faveur du joueur au trait) ; la majorité des positions d'auto-jeu visitées
  sont donc filtrées, pas écrites. Aucun `VALUE_KEY` (même raisonnement que §2.6).
- **Auto-jeu source volontairement asymétrique** (`weak_depth_range=(2, 5)` par défaut,
  ajouté à `generate_selfplay_positions` de §2.5 — additif, `None` = comportement
  §2.5/§2.6 inchangé) : une couleur plus faible par partie produit davantage de vraies
  erreurs/coups punitifs, matière première du filtre delta. N'affecte que la génération
  côté `chess_ai`, aucune conséquence sur le schéma `.npz`.
- **Encodage position** : chemin par défaut inchangé, comme §2.6 (aucune nouvelle
  variante de plans pour ce dataset).
- **Sortie de fichiers** : préfixe dédié (`chess_decisive_teacher`), distinct des trois
  préfixes existants (`chess`, `chess_legal_moves`, `chess_search_teacher`) — aucune
  collision (AD-4 côté `chess_ai`).
- **État du dataset (mesuré, pas supposé)** : run à pleine échelle 2026-08-12, 10 000
  parties/7 workers, **286 107 positions gardées** (~2,9% des positions d'auto-jeu
  visitées — filtre delta volontairement sélectif), 29 chunks. Smoke test préalable
  (30 parties mono-processus) : taux de rétention ~26% par-position-non-terminale,
  confirmé non-dégénéré avant de lancer le run complet.
- **Statut côté `jax_supervised_training`** : aucune entrée `dataset_configs.py` créée à
  ce jour — reste à faire côté training (probablement `CHESS_DECISIVE_TEACHER`, même
  patron que `CHESS_SEARCH_TEACHER` §2.6 : `task_type="chess_policy_value"`,
  `model_name="chess_cnn_attention_policy_value"`, tête value neutralisée par
  `loss_params.value_weight=0.0`). Comparer l'accuracy et surtout le taux de
  nulles-par-répétition en self-play contre le checkpoint `CHESS_SEARCH_TEACHER` existant
  (28.00%, §2.6 ci-dessus) est la question ouverte que ce dataset doit trancher.

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

### CHESS_TOKEN_1_MOVE (spec `spec-chess-token-1-move`, `chess_ai`, 2026-08-15)

Suite de `spec-chess-token-candidate-poc` (non documenté ici, reste hors contrat stable,
spike non concluant à ce jour) — diagnostic : (a) `Dense(4672)` est structurellement
gaspillé sur un espace de sortie qui n'a jamais 4672 coups réellement possibles par
position ; (b) le scorer à 50 candidats de `chess-token-candidate-poc` ne peut, par
construction, jamais produire de coup illégal (candidats pré-filtrés par `python-chess`
avant même d'atteindre le modèle) — ce qui empêche de mesurer si le réseau "connaît" les
règles. Spec complète (rationale, capabilities, contraintes) :
`chess_ai/_bmad-output/specs/spec-chess-token-1-move/SPEC.md` + son companion
`architecture-diagram.md`. **Rapport d'état pour lancer l'epic `jax_supervised_training`
(§4), pas une promotion vers le contrat stable §2** — reste hors de §2 tant que non
concluant, même discipline que `move_token`/`chess-token-candidate-poc`.

- **Dataset : aucun changement côté `chess_ai` — mais PAS `chess_search_teacher`.**
  **Correction 2026-08-15** (défaut trouvé côté `jax_supervised_training`, spec initiale
  fausse) : `chess_search_teacher` (§2.6) stocke des plans binaires (`encode_position`,
  8×8×29), incompatibles avec le tronc `token_embed`/`pos_embed` de ce modèle, qui a
  besoin de `token_position` (`encode_position_tokenized`). Le bon dataset, déjà généré,
  déjà au bon format, est celui de `chess-token-candidate-poc` :
  `/home/aobled/Documents/data/chunks/chess_token_candidate_spike/chess_token_candidate_spike.npz`
  (1 244 231 positions, 2026-08-13, professeur Stockfish profondeur 12 — même professeur
  que `chess_search_teacher`, échantillon d'auto-jeu différent). Clés `token_position`
  `(N,64)` et `global_flags` `(N,6)` utilisées directement comme entrée modèle ; le
  label unique attendu par ce modèle se dérive de `candidate_moves[i, candidate_label[i]]`
  (schéma complet :
  `chess_ai/_bmad-output/specs/spec-chess-token-candidate-poc/token-candidate-dataset-schema.md`)
  — les colonnes `candidate_moves`/`candidate_mask` restantes ne sont pas utilisées par
  cette architecture à tête unique.
- **Impact sur le calcul de la loss (côté `jax_supervised_training` uniquement)** : le
  label dérivé ci-dessus (index unique dans `[0, NUM_MOVES)`, §2.2) doit être décomposé
  en `(from_square, move_type)` au moment du calcul de la loss — nouvelle fonction
  `chess_target_encoding.py::decompose_move_index(index) -> (from_square, move_type)`
  (`chess_ai`, ajoutée 2026-08-15, inverse arithmétique de `move_to_index` existant §2.2,
  `divmod(index, 73)`, aucune validation de légalité). Loss jointe sur 2 têtes
  indépendantes (`from_square` 64 classes, `move_type` 73 classes, **prédites
  indépendamment l'une de l'autre, sans conditionnement**) au lieu d'une seule loss
  `Dense(4672)`.
- **Modèle cible** : tronc auto-attention pure sur tokens-case (`token_embed`/`pos_embed`,
  sans CNN — même principe d'entrée que `chess-token-candidate-poc`), **volontairement
  allégé** (moins/plus étroit de couches d'attention) pour que `token_embed`/`pos_embed`
  représentent une part significative des paramètres totaux (objectif explicite : rendre
  `chess_bottleneck_genetic.py` capable d'un impact comportemental mesurable — 5
  tentatives précédentes négatives sur un sous-espace mutable trop petit, voir
  `chess_ai/docs/chess_ai_global_conclusions.md` §4). **Pas d'étage bottleneck**
  (pas de cross-/auto-attention vers K tokens latents) — le tronc alimente directement
  les 2 têtes de sortie.
- **Comparaison** : accuracy jointe (`from_square` ET `move_type` corrects
  simultanément) face à `CHESS_SEARCH_TEACHER` (28.00% val `PolicyAccuracy`, §2.6).
- **Config `dataset_configs.py`** : nom de travail `CHESS_TOKEN_1_MOVE` — nommage
  exact/`task_type`/`model_name` restent une décision côté `jax_supervised_training`
  (même précédent que le nommage Modèle 1/2 ci-dessus).

**Résultat 2026-08-15 (entraînement réel, `jax_supervised_training`) : version éliminée,
pas concluante.** Training lancé (GPU, 50 epochs prévues), arrêté manuellement à
l'epoch 40/50 — plateau net de `JointMoveAccuracy` depuis l'epoch ~34 (0.0500→0.0510,
quasi stable sur 6 epochs). **Train ≈ Val** (Train ~4.5-4.6%, Val 5.06-5.10%, aucun
écart) : signature d'un plafond de capacité/structurel, pas d'overfitting (même
diagnostic que `chess-search-teacher-strategy.md`, §2.6). Très en-deçà des 28.00%
`CHESS_SEARCH_TEACHER`, mais pas strictement comparable (accuracy jointe sur 2 têtes non
masquées 64×73 vs top-1 sur 4672 déjà filtré). Checkpoint conservé (19 977 paramètres,
`best_model_chess_token_1_move.pkl`) mais non retenu comme base de travail. Détail complet :
`_bmad-output/implementation-artifacts/spec-chess-token-1-move.md` (section "Training
Outcome").

**Piste v2 à ÉTUDIER, non retenue/non planifiée à ce jour (diagnostic Winston,
2026-08-15)** — suspect principal : l'absence de conditionnement entre les 2 têtes
(contrainte actuelle du §3 ci-dessus, "prédites indépendamment") oblige le modèle à
marginaliser sur l'identité de la pièce en case de départ pour prédire `move_type`, alors
que ce type de coup en dépend presque entièrement aux échecs — un vrai verdict
nécessiterait de lever cette contrainte, ce qui reviendrait sur l'objectif "illégalité
mesurable" du diagnostic initial de ce spec (à rediscuter si cette piste est reprise, pas
un simple réglage d'hyperparamètre). Options identifiées, par ordre d'impact attendu si
cette piste est un jour reprise :
1. Conditionner légèrement `move_type` sur `from_square` (ex. `[pooled ; embedding(from_square)]`
   en entrée de la tête `move_type`, teacher-forcing à l'entraînement) — lève le verrou
   structurel principal, sans réintroduire un conditionnement total (le `from_square`
   prédit reste libre, donc un coup illégal reste possible).
2. Remplacer le mean-pool actuel par un token de lecture appris, inséré comme 65e token
   dans la même séquence auto-attention (pas un étage de cross-attention séparé façon
   bottleneck — resterait conforme à la contrainte "pas d'étage bottleneck" ci-dessus).
3. Élargir `token_dim`/`num_trunk_layers` (le tronc actuel, `token_dim=32`/1 couche, est
   délibérément allégé) — levier confirmé dans ce domaine (`CHESS_SEARCH_TEACHER`,
   64→128 = +6pts) mais probablement insuffisant seul vu le plafond train≈val observé.

**Reprise actée 2026-08-16** — les 3 options ci-dessus testées en UN SEUL run combiné
(pas 3 runs isolés) : `spec-chess-token-1-move-v2.md` (`jax_supervised_training`), édité
EN PLACE sur `CHESS_TOKEN_1_MOVE` (pas de nouveau domaine). Détail complet (rationale,
Boundaries & Constraints, résultat) : `_bmad-output/implementation-artifacts/
spec-chess-token-1-move-v2.md` côté `jax_supervised_training`.

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
