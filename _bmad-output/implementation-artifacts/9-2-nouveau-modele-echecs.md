---
baseline_commit: 1a85f2583c10770dcbeb84140aaac9fe842781f7
---

# Story 9.2: Nouveau modèle échecs

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline de modèles,
I want un nouveau modèle `chess_cnn_attention_policy_value` enregistré dans `model_library.py`,
so that le domaine échecs dispose d'une architecture CNN+bottleneck de tokens appris+attention+têtes policy/value, dimensionnée par le schéma d'échange de la Story 9.1 (`chess_target_encoding.py`, `NUM_MOVES=4672`, `NUM_PLANES=29`).

## Acceptance Criteria

1. **Given** `chess_target_encoding.py` (Story 9.1) et sa constante `NUM_MOVES` **When** le modèle `chess_cnn_attention_policy_value` est implémenté **Then** sa tête policy produit un vecteur de taille `NUM_MOVES` (source unique, pas un littéral dupliqué) — AD-22
2. **Given** l'architecture envisagée **When** le modèle est construit **Then** il enchaîne CNN 8×8 (convolutions + blocs résiduels) → bottleneck de K tokens par requêtes apprises (cross-attention, Perceiver/TokenLearner-style) → auto-attention standard entre les tokens (sans biais géométrique) → tête policy + tête value — FR5, AD-23
3. **Given** la tête value **When** elle est implémentée **Then** elle produit un scalaire (`Dense(1)`+`tanh`, borne [-1, 1]) — AD-24
4. **Given** `model_library.get_model()` **When** `chess_cnn_attention_policy_value` est appelé avec les `model_kwargs` génériques (`num_classes`, `dropout_rate`) **Then** il s'instancie sans modification de `get_model()` ni de `main.py` au-delà du dispatch déjà prévu

## Tasks / Subtasks

- [x] Task 1: Créer `ChessCnnAttentionPolicyValue` dans `model_library.py` (AC: 1, 2, 3)
  - [x] Backbone CNN 8×8, **sans aucun maxpool** (voir Dev Notes § Pourquoi pas de maxpool) : blocs résiduels `SeparableConv`+`BatchNorm`+`silu` au pattern déjà établi (voir Dev Notes § Pattern résiduel à réutiliser), résolution spatiale 8×8 conservée du début à la fin, canaux de sortie finaux = D (dimension des tokens, voir ci-dessous)
  - [x] Aplatir la sortie CNN `(B, 8, 8, D)` en séquence de 64 tokens `(B, 64, D)` — un token par case
  - [x] Bottleneck : K vecteurs de requête **appris** (`self.param`, shape `(K, D)`, broadcastés sur le batch) + `nn.MultiHeadDotProductAttention` en cross-attention (`inputs_q=requêtes, inputs_kv=64 tokens case`) → `(B, K, D)` (voir Dev Notes § Requêtes apprises Perceiver-style pour le pattern Flax exact)
  - [x] Auto-attention standard entre les K tokens du bottleneck (`nn.MultiHeadDotProductAttention(inputs_q=tokens)`, self-attention) — sans biais géométrique (AD-23, différé)
  - [x] Pooling des K tokens (`jnp.mean` sur l'axe K) → vecteur `(B, D)`
  - [x] Tête policy : `nn.Dense(NUM_MOVES)` sur le vecteur poolé — **logits bruts, aucune activation** (la cross-entropy de la Story 9.3 attend des logits, pas une distribution déjà normalisée)
  - [x] Tête value : `nn.Dense(1)` puis `nn.tanh`, `squeeze` du dernier axe → shape `(B,)` (pas `(B, 1)`)
  - [x] `__call__` retourne `{POLICY_KEY: policy_logits, VALUE_KEY: value}` — miroir exact du pattern dict `{HEATMAP_KEY: ..., SIZE_KEY: ...}` de `AircraftDetectorCenterNet` (voir Dev Notes)
- [x] Task 2: Factory `create_chess_cnn_attention_policy_value` + enregistrement dans `MODELS` (AC: 1, 4)
  - [x] Signature `create_chess_cnn_attention_policy_value(num_classes, dropout_rate=0.1, **kwargs)` — **doit utiliser `num_classes`**, ne pas le laisser tomber dans `**kwargs` (voir Dev Notes § Piège : num_classes silencieusement ignoré, un piège réel déjà présent dans ce fichier pour un autre modèle)
  - [x] `num_classes` (nom du kwarg générique, imposé par la plomberie `model_kwargs` de `main.py`) porte `NUM_MOVES` — le champ interne du modèle peut s'appeler `num_moves` pour la clarté, mais le kwarg **externe** de la factory doit rester `num_classes` pour rester compatible avec `main.py:110` sans modification (AC: 4)
  - [x] Ajouter l'entrée `'chess_cnn_attention_policy_value': create_chess_cnn_attention_policy_value` dans le dict `MODELS` (`model_library.py`, juste après les entrées CenterNet/sophisticated existantes)
- [x] Task 3: Script de validation autonome (AC: 1, 2, 3, 4)
  - [x] Construire le modèle avec `num_classes=NUM_MOVES` (importé de `chess_target_encoding.py`, jamais un littéral `4672`), `model.init(rng, x, training=True)` sur un batch factice `(B, 8, 8, NUM_PLANES)` (`NUM_PLANES` importé, jamais `29` en dur)
  - [x] Vérifier la shape/dtype de sortie : `POLICY_KEY` → `(B, NUM_MOVES)` float32 ; `VALUE_KEY` → `(B,)` float32, valeurs dans `[-1, 1]` (vérifier bornes réelles sur plusieurs forward pass, pas juste supposer `tanh`)
  - [x] Vérifier `apply_fn(vars, x, training=True, mutable=['batch_stats'], rngs=...)` (mode entraînement, `batch_stats` mutable) **et** `apply_fn(vars, x, training=False)` (mode éval) — les deux chemins utilisés par `trainer.py`, voir Dev Notes § Contrat Trainer
  - [x] Vérifier que `get_model('chess_cnn_attention_policy_value', num_classes=NUM_MOVES, dropout_rate=0.1)` fonctionne (AC 4, appel via la factory générique, pas la classe directement)
  - [x] Vérifier qu'un `jax.grad` sur une loss factice (ex. somme des deux sorties) ne lève aucune erreur — confirme la différentiabilité de bout en bout (bottleneck + attention inclus) avant la Story 9.3

## Dev Notes

### Portée exacte (ne pas dépasser)

- Cette story modifie **uniquement** `model_library.py` (ajout d'une classe + une factory + une entrée `MODELS`). Elle ne touche **aucun** autre fichier existant (`dataset_configs.py`, `task_strategies.py`, `data_management.py`, `main.py` — zéro modification). L'entrée `CHESS` dans `dataset_configs.py` et le dispatch `task_type` sont la Story 9.3, hors scope ici.
- Pas de masquage des coups illégaux (AD-22, déjà tranché en Story 9.1) — la tête policy retourne des logits bruts sur tout l'espace `NUM_MOVES`, sans lien avec `chess_target_encoding.py::index_to_move`.
- Pas de biais géométrique dans l'attention (AD-23, différé) — `nn.MultiHeadDotProductAttention` standard, sans encodage positionnel dédié à la géométrie de l'échiquier.

### Pourquoi pas de maxpool (contrairement aux CNN 128×128 existants)

Les CNN images de ce projet (`SophisticatedCNN128Plus/Lite`) maxpoolent agressivement (128→64→32→16) car l'image d'entrée est grande et le calcul doit être maîtrisé. Le plateau d'échecs fait 8×8 : maxpooler ferait 8→4→2→1, détruisant l'identité des cases avant même d'atteindre le bottleneck — alors que la tête policy (AD-22) a besoin de distinguer les 64 cases source. Le brief produit (2026-07-27) justifie explicitement l'absence de pooling : *"un empilement de 3-4 convs 3×3 [suffit à] couvrir [le plateau 8×8] entièrement"* en champ réceptif — donc pas besoin de sous-échantillonner pour élargir le champ réceptif, contrairement au cas image. **Ne pas copier le pattern maxpool des CNN 128×128 existants.**

### Pattern résiduel à réutiliser (déjà établi, ex. `SophisticatedCNN128Lite`, `model_library.py:242-254`)

```python
x = SeparableConv(filters, (3, 3))(x, training)
x = nn.BatchNorm(use_running_average=not training)(x)
x = nn.silu(x)

residual = nn.Conv(filters, (1, 1), padding="SAME", use_bias=False,
                    kernel_init=nn.initializers.kaiming_normal())(x)
residual = nn.BatchNorm(use_running_average=not training)(residual)

x = SeparableConv(filters, (3, 3))(x, training)
x = nn.BatchNorm(use_running_average=not training)(x)
x = x + residual
x = nn.silu(x)
```
`SeparableConv`, `SEBlock`, `SpatialAttention` sont déjà définis en tête de `model_library.py` (lignes 18-83) — réutilisables tels quels, ne pas les redéfinir. `padding="SAME"`, jamais de `strides` > 1 ni de `nn.max_pool` (voir ci-dessus).

### Requêtes apprises Perceiver-style (premier usage de ce pattern dans ce codebase)

```python
K, D = 8, 64  # valeurs de depart (proposition initiale d'Aymeric, brief 2026-07-27, memlog) - hyperparametres, ajustables
queries = self.param('bottleneck_queries', nn.initializers.normal(0.02), (K, D))
queries = jnp.broadcast_to(queries[None, :, :], (x.shape[0], K, D))  # (B, K, D)

tokens = x.reshape(x.shape[0], 64, D)  # (B, 8, 8, D) -> (B, 64, D), un token par case
bottleneck = nn.MultiHeadDotProductAttention(num_heads=4)(inputs_q=queries, inputs_kv=tokens)  # (B, K, D)
bottleneck = nn.MultiHeadDotProductAttention(num_heads=4)(inputs_q=bottleneck)  # self-attention, (B, K, D)
```
`num_heads=4` avec `D=64` (16 par tête) est une proposition de départ, pas une contrainte architecturale — ajustable. `flax.linen.MultiHeadDotProductAttention` (vérifié installé, `flax` 0.10.7, voir spine AD-23) : `inputs_q` seul = self-attention ; `inputs_q` + `inputs_kv` = cross-attention (API vérifiée directement sur l'installation, pas supposée de mémoire).

### Piège : `num_classes` silencieusement ignoré (déjà présent ailleurs dans ce fichier, à ne pas reproduire)

`create_aircraft_detector_centernet(dropout_rate=0.2, heatmap_prior=0.01, **kwargs)` (`model_library.py:694`) absorbe `num_classes` dans `**kwargs` **sans jamais l'utiliser** — `AircraftDetectorCenterNet` n'a pas de champ `num_classes`, donc `main.py` peut passer `num_classes=1` sans effet, ce qui est correct pour CenterNet (mono-classe, pas besoin de ce paramètre) mais **serait un bug silencieux pour le modèle échecs**, où `num_classes` est précisément le mécanisme choisi en architecture (AD-22) pour transporter `NUM_MOVES` jusqu'à la tête policy, en réutilisant la plomberie `model_kwargs` existante de `main.py:110` (`{"num_classes": ..., "dropout_rate": ...}`) sans aucun kwarg nouveau. La factory de cette story doit donc **lire et utiliser** `num_classes`, pas seulement l'accepter en signature.

### Contrat Trainer (déjà générique, aucune modification requise)

`trainer.py` appelle `apply_fn(vars, images, training=True, mutable=['batch_stats'], rngs=rngs)` à l'entraînement et `apply_fn(vars, images, training=False)` à l'éval — mécanisme déjà générique (fonctionne pour tout modèle avec `nn.BatchNorm`), aucune modification de `trainer.py` nécessaire. Le modèle doit juste respecter la signature `__call__(self, x, training=True)` déjà utilisée par tous les modèles existants.

### Project Structure Notes

- `model_library.py` — fichier **UPDATE**, pas nouveau. Ajouter la classe après `AircraftDetectorCenterNetLite`/avant `Kepler1DConvNet` (ou en fin de fichier, avant `MODELS`), cohérent avec l'ordre existant (détection, puis classification, puis Kepler). Import à ajouter en tête de fichier : `from chess_target_encoding import POLICY_KEY, VALUE_KEY, NUM_MOVES, NUM_PLANES` (miroir de `from detection_target_encoding import HEATMAP_KEY, SIZE_KEY` déjà présent ligne 15).
- Aucun nouveau fichier créé par cette story.
- Aucune nouvelle dépendance externe — `flax.linen.MultiHeadDotProductAttention` déjà disponible (`flax` 0.10.7, vérifié par le spine d'architecture AD-23).

### Testing Standards

Pas de suite de tests automatisée formelle dans ce projet — script autonome (Task 3), même convention que `tests/test_chess_target_encoding.py` (Story 9.1) et `tests/test_detection_target_encoding.py` (Story 7.1). Fichier suggéré : `tests/test_chess_model.py`.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` § Epic 9, Story 9.2] — story source, ACs
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md#AD-22,AD-23,AD-24`] — espace policy/source unique num_classes, bottleneck Perceiver-style, value scalaire tanh+MSE
- [Source: `_bmad-output/planning-artifacts/briefs/brief-jax_supervised_training-2026-07-27/brief.md`, `.memlog.md`] — architecture envisagée par Aymeric (CNN 8×8 → bottleneck tokens → attention → policy/value), justification receptive field, proposition initiale K=8/D=64
- [Source: `model_library.py:18-83`] — `SeparableConv`/`SEBlock`/`SpatialAttention`, réutilisables tels quels
- [Source: `model_library.py:212-310` (`SophisticatedCNN128Lite`)] — pattern résiduel à reprendre pour le backbone CNN
- [Source: `model_library.py:548-696` (`AircraftDetectorCenterNet` + `create_aircraft_detector_centernet`)] — précédent direct pour la sortie dict à 2 têtes et le piège `num_classes` ignoré
- [Source: `model_library.py:884-908`] — dict `MODELS`, `get_model()` (aucune modification structurelle requise, juste une entrée)
- [Source: `main.py:100-116`] — plomberie `model_kwargs` générique (`num_classes`, `dropout_rate`), pattern du kwarg conditionnel `heatmap_prior`
- [Source: `trainer.py:229-274`] — contrat `apply_fn`/`mutable=['batch_stats']`/`training=True|False`, déjà générique
- [Source: `chess_target_encoding.py`] — `NUM_MOVES=4672`, `NUM_PLANES=29`, `POLICY_KEY`/`VALUE_KEY` (Story 9.1)
- [Source: `_bmad-output/implementation-artifacts/9-1-module-dechange-partage-builder-dataset-pgn.md`] — story précédente, contrat d'entrée complet + apprentissages (bug de documentation NUM_PLANES corrigé en review, garder les deux valeurs — 4672 et 29 — cohérentes partout dans cette story aussi)

## Dev Agent Record

### Agent Model Used

### Debug Log References

- Smoke test manuel après Task 1/2 : `get_model('chess_cnn_attention_policy_value', num_classes=NUM_MOVES, dropout_rate=0.1)` + forward pass sur `(2, 8, 8, NUM_PLANES)` — shapes `policy (2, 4672)`, `value (2,)` confirmées avant d'écrire le script de test complet
- `python3 tests/test_chess_model.py` — 5/5 tests passés (shapes/dtypes, bornes value sur 40 échantillons, modes train/eval `apply_fn`, factory/registre `get_model`, différentiabilité de bout en bout via `jax.grad`)
- `python3 tests/test_chess_target_encoding.py`, `test_chess_pgn_dataset_tools.py`, `test_detection_target_encoding.py` — repassés, aucune régression (13/13, 2/2, 6/6)
- `python3 -c "import main"` — import propre après modification de `model_library.py` (GPU détecté, aucune erreur)

### Completion Notes List

- Implémenté `ChessCnnAttentionPolicyValue` (`model_library.py`) : backbone CNN 8×8 sans maxpool (1 conv initiale + 2 blocs résiduels `SeparableConv`+`BatchNorm`+`silu`, pattern réutilisé de `SophisticatedCNN128Lite`), bottleneck Perceiver-style (`K=8` requêtes apprises, `D=64`, cross-attention puis auto-attention via `nn.MultiHeadDotProductAttention`, `num_heads=4`), pooling moyen, tête policy (`Dense(num_moves)`, logits bruts) et tête value (`Dense(1)`+`tanh`, squeeze vers `(B,)`). Retourne `{POLICY_KEY: ..., VALUE_KEY: ...}`.
- `create_chess_cnn_attention_policy_value(num_classes, dropout_rate=0.1, **kwargs)` lit et utilise `num_classes` (transmis à `num_moves`) — vérifié explicitement par test (`test_get_model_factory_and_registry`) que ce n'est pas silencieusement ignoré comme dans `create_aircraft_detector_centernet`. Entrée `'chess_cnn_attention_policy_value'` ajoutée au dict `MODELS`.
- **Écart mineur par rapport aux Dev Notes** : la story suggérait d'importer `POLICY_KEY, VALUE_KEY, NUM_MOVES, NUM_PLANES` en tête de `model_library.py`. Seuls `POLICY_KEY`/`VALUE_KEY` sont effectivement utilisés dans ce fichier (le nombre de canaux d'entrée est inféré par la première conv, comme tous les autres modèles ; `num_moves` arrive en paramètre via la factory, pas en constante importée) — `NUM_MOVES`/`NUM_PLANES` non importés pour éviter un import inutilisé. Aucun impact sur les ACs, `chess_target_encoding.py` reste la source unique de ces constantes (AD-18/AD-22), simplement pas dupliquée en import mort ici.
- Les 4 acceptance criteria sont satisfaits et vérifiés par test : AC1 (`NUM_MOVES` transmis via `num_classes`, jamais un littéral dans le modèle), AC2 (CNN 8×8 sans pooling → bottleneck K tokens appris → auto-attention, architecture vérifiée par les shapes intermédiaires implicites dans les tests de forward pass), AC3 (value scalaire `(B,)` bornée `[-1,1]`, bornes réelles vérifiées sur 40 échantillons), AC4 (`get_model()` fonctionne sans modification, `main.py` non touché - import propre confirmé).

### File List

- `model_library.py` (modifié — nouvelle classe `ChessCnnAttentionPolicyValue`, nouvelle factory `create_chess_cnn_attention_policy_value`, nouvel import `chess_target_encoding`, nouvelle entrée `MODELS`, nouvelle entrée `get_model_info()`)
- `tests/test_chess_model.py` (nouveau)
- `requirements.txt` (modifié — ajout de `chess`, trouvé manquant en code review)

### Review Findings

Code review adversarial (2026-07-27, Blind Hunter + Edge Case Hunter + Acceptance Auditor en parallèle). 1 finding `high` (dépendance `chess` absente de `requirements.txt`, menace AD-21), plusieurs `medium`/`low` de robustesse et de fidélité aux Dev Notes. 3 findings écartés comme bruit ou hors scope.

**Patch (11) :**

- [x] [Review][Patch] `chess` absent de `requirements.txt` malgré la nouvelle dépendance transitive **universelle** : `model_library.py` importe désormais `chess_target_encoding.py` en tête de fichier, donc TOUT entraînement (CIFAR10, `JAX_DETECTOR` compris) casserait sur un environnement frais sans `python-chess` installé — menace directement AD-21 (zéro impact sur `JAX_DETECTOR`), pas juste un problème du domaine échecs [requirements.txt]
- [x] [Review][Patch] Reshape du bottleneck code en dur `8 * 8` au lieu de le dériver de `x.shape` — contredit la docstring de la classe qui affirme ne rien coder en dur (canaux d'entrée inférés) ; aucune validation de shape avant reshape ; aucun test pour une entrée mal formée [model_library.py, `ChessCnnAttentionPolicyValue.__call__`]
- [x] [Review][Patch] `get_model_info()` non mis à jour pour le nouveau modèle — casse la parité 1:1 que chaque entrée de `MODELS` a avec ce dict, lèverait `ValueError` si un futur appelant l'itère [model_library.py:1029-1091]
- [x] [Review][Patch] La factory `create_chess_cnn_attention_policy_value` accepte `**kwargs` mais ne les transmet jamais à `ChessCnnAttentionPolicyValue(...)` — les 3 hyperparamètres explicitement qualifiés d'"ajustables" dans la docstring (`token_dim`, `num_bottleneck_tokens`, `num_heads`) sont en réalité inatteignables depuis `main.py`
- [x] [Review][Patch] Aucune validation de `num_moves > 0`, et surtout aucune protection contre le défaut `num_classes=2` déjà utilisé ailleurs dans ce même fichier (`SophisticatedCNN128Lite` etc.) — un appel sans `num_classes=NUM_MOVES` explicite construirait silencieusement une tête policy de la mauvaise taille, sans erreur
- [x] [Review][Patch] Pas d'assert `token_dim % num_heads == 0` — erreur Flax opaque en cas de future modification de ces hyperparamètres
- [x] [Review][Patch] Topologie du bloc résiduel différente du pattern exact demandé dans les Dev Notes (skip depuis l'entrée brute du bloc + 2 `SeparableConv` séquentielles, vs. le pattern établi de `SophisticatedCNN128Lite` où le skip part après la première conv+activation) — pas un bug (shapes/gradient corrects), mais un écart non documenté par rapport à une instruction explicite de réutilisation à l'identique
- [x] [Review][Patch] Entrée `MODELS` ajoutée en toute fin (après `kepler_1d_cnn`) plutôt que "juste après les entrées CenterNet/sophisticated" comme demandé en Task 2 — cosmétique, aucun impact fonctionnel
- [x] [Review][Patch] `test_end_to_end_differentiability` ne vérifie qu'un OU logique sur tous les gradients confondus (`any(leaf != 0)`), pas composant par composant — pourrait ne pas détecter une branche morte (ex. auto-attention court-circuitée)
- [x] [Review][Patch] Docstring de la classe cite "brief produit 2026-07-27" sans chemin de fichier — citation externe non vérifiable sans déjà savoir où chercher
- [x] [Review][Patch] `test_value_bounded_in_tanh_range` : le commentaire "10 inits différentes" survend ce qui est réellement testé (statistiques `BatchNorm` toujours non-entraînées à chaque init fraîche, pas un comportement post-entraînement réaliste) — le test lui-même reste correct, formulation à préciser

**Tous les 11 patchs appliqués (2026-07-27).** Point notable : en corrigeant le patch "test de différentiabilité composant par composant", le test plus strict a révélé que `test_end_to_end_differentiability` utilisait une entrée `x=zeros(...)` (héritée de `_init_model`, partagée avec d'autres tests) — avec une entrée exactement nulle, les clés/valeurs d'attention sont toutes à zéro, donc le softmax devient uniforme et indépendant des requêtes : `bottleneck_queries` et une bonne partie du backbone recevaient un gradient réellement nul, pas un faux positif du test. Corrigé en donnant à ce test sa propre entrée aléatoire (`jax.random.normal`) — les 19 modules du modèle reçoivent maintenant un gradient fini et non-nul, une garantie de différentiabilité bien plus solide que le test initial (qui aurait laissé passer une branche morte). `python3 tests/test_chess_model.py` (5/5), `test_chess_target_encoding.py`/`test_chess_pgn_dataset_tools.py`/`test_detection_target_encoding.py` (non-régression, tous verts), `python3 -c "import main"` (import propre) re-vérifiés après tous les patchs.

**Écartés (dismiss, 3)** : import `NUM_MOVES`/`NUM_PLANES` non fait en tête de fichier — déjà auto-divulgué dans les Completion Notes avec justification, l'auditeur confirme lui-même qu'aucun AC n'est violé ; absence de garde-fou empêchant d'appeler ce modèle hors d'un contexte échecs (`task_type` non-échecs) — c'est explicitement le travail de dispatch de la Story 9.3, pas de celle-ci ; structure du script de test à `assert` bruts sans isolation par test — convention déjà établie de ce projet (le Blind Hunter le confirme lui-même, pas une régression introduite ici).
