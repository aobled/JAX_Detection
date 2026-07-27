---
baseline_commit: 1a85f2583c10770dcbeb84140aaac9fe842781f7
---

# Story 9.3: `TaskStrategy` échecs, loss composite et intégration au pipeline

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'entraînement,
I want une nouvelle `ChessPolicyValueStrategy` et une entrée `CHESS`/`task_type="chess_policy_value"` dispatchée dans `main.py`/`data_management.py`,
so that le domaine échecs s'entraîne de bout en bout via `Trainer`, sans modification structurelle de celui-ci (FR4/FR6, AD-17/AD-24).

## Acceptance Criteria

1. **Given** le modèle (Story 9.2) et le dataset (Story 9.1) **When** `compute_loss` est appelé **Then** il retourne une loss unique `policy_weight * policy_loss + value_weight * value_loss` (poids dans `loss_params`, exactement sur le modèle de `compute_centernet_loss`) — AD-24, FR4
2. ~~**Given** `trainer.py` **When** cette story est complétée **Then** aucune ligne de `trainer.py` n'est modifiée — FR6~~ **AMENDÉ (2026-07-27, code review, arbitré par Aymeric)** : `trainer.py` porte exactement 1 ligne modifiée (`self.num_channels`, `Trainer.__init__`) — écart nécessaire trouvé en review (le domaine échecs a 29 canaux d'entrée, hors de l'hypothèse grayscale/RGB binaire de `Trainer`), documenté et autorisé explicitement plutôt qu'absorbé silencieusement (voir PRD FR-6 "Out of Scope", ce scénario était anticipé). Vérifié par test que c'est **exactement** ce changement, rien de plus (`test_trainer_change_is_the_single_authorized_deviation`).
3. **Given** `generate_reports()` **When** l'entraînement se termine **Then** le détail `policy_loss`/`value_loss` est affiché séparément, en réutilisant `self.loss_params` et les mêmes sous-fonctions de `loss_functions.py` (pas de duplication) — AD-24
4. **Given** `primary_metric_name`/`optimization_mode` **When** le meilleur modèle est sauvegardé **Then** c'est la policy accuracy (top-1) qui gate la sauvegarde, `optimization_mode="max"` — AD-24
5. **Given** `task_type="chess_policy_value"` et l'entrée `CHESS` (`dataset_configs.py`) **When** `main.py` et `data_management.py` sont mis à jour **Then** le dispatch suit exactement le pattern des 2 points de dispatch réels existants (AD-17 hérité), sans branche conditionnelle sur une classe existante
6. **Given** AD-17 **When** cette story est complétée **Then** aucune modification n'est faite à `ClassificationStrategy`/`DetectionStrategy`/`CenterNetDetectionStrategy`/`KeplerStrategy` ni à leurs classes de chargeur associées

## Tasks / Subtasks

- [x] Task 1: `loss_functions.py` — 3 fonctions, miroir de `compute_centernet_loss` (AC: 1, 3)
  - [x] `compute_chess_policy_loss(policy_logits, policy_targets)` — `optax.softmax_cross_entropy_with_integer_labels(...).mean()`, même formule que `ClassificationStrategy.compute_loss` (`task_strategies.py`) mais factorisée ici (miroir architectural de `compute_centernet_loss`, pas inline dans la strategy)
  - [x] `compute_chess_value_loss(value_pred, value_targets)` — MSE simple (`jnp.mean((value_pred - value_targets) ** 2)`), les deux déjà dans `[-1, 1]`
  - [x] `compute_chess_policy_value_loss(outputs, targets, policy_weight=1.0, value_weight=1.0)` — combine les deux ci-dessus en une loss unique pondérée, exactement sur le modèle de `compute_centernet_loss` (`loss_functions.py:547`). `outputs`/`targets` = dict `{POLICY_KEY: ..., VALUE_KEY: ...}`
  - [x] Ajouter `import optax` et `from chess_target_encoding import POLICY_KEY, VALUE_KEY` en tête de `loss_functions.py` (miroir de `from detection_target_encoding import HEATMAP_KEY, SIZE_KEY` déjà présent)
- [x] Task 2: `task_strategies.py` — `ChessPolicyValueStrategy` (AC: 1, 3, 4, 6)
  - [x] `__init__(self, loss_params: dict = None)` — même pattern que `CenterNetDetectionStrategy` (pas de `loss_method`/`metric_method`/`report_method`, une seule méthode de perte/métrique)
  - [x] `primary_metric_name` → `"PolicyAccuracy"`, `optimization_mode` → `"max"` (AC: 4)
  - [x] `_get_export_path`/`get_training_state_path` — copie identique du pattern générique dérivé de `dataset_name` (voir Dev Notes, déjà partagé par les 4 strategies existantes)
  - [x] `preprocess_batch(self, images, targets, is_training, rng=None)` — cast `targets[POLICY_KEY]` en `int32`, `targets[VALUE_KEY]` en `float32` ; pas de mixup/label smoothing ; retourne `(images, targets, False)` (miroir `CenterNetDetectionStrategy.preprocess_batch`)
  - [x] `compute_loss(self, outputs, targets, **kwargs)` → `compute_chess_policy_value_loss(outputs, targets, **self.loss_params)` (AC: 1)
  - [x] `compute_metrics(self, outputs, targets)` → policy top-1 accuracy : `(jnp.argmax(outputs[POLICY_KEY], axis=-1) == targets[POLICY_KEY]).mean()` (même formule que `ClassificationStrategy.compute_metrics`)
  - [x] `generate_reports(self, val_ds, final_state, model, config)` — prend 1 batch de `val_ds`, appelle `final_state.apply_fn(...)`, calcule `compute_chess_policy_loss`/`compute_chess_value_loss` séparément (import direct depuis `loss_functions.py`, **pas** de réimplémentation), affiche les deux valeurs + leurs poids (`self.loss_params`) (AC: 3)
- [x] Task 3: `data_management.py` — `ChessPolicyValueDataset` + dispatch `get_datasets()` (AC: 5, 6)
  - [x] Nouvelle classe `ChessPolicyValueDataset` (voir Dev Notes § Chargeur échecs pour le détail complet — **split train/val différent des autres loaders**, à lire avant de coder)
  - [x] Nouvelle branche `elif task_type == "chess_policy_value":` dans `get_datasets()` (`data_management.py:602+`), instancie `ChessPolicyValueDataset`, retourne `(train_ds, val_ds)` — même forme que les branches `detection`/`detection_centernet` existantes
  - [x] Import `POSITION_KEY, POLICY_KEY, VALUE_KEY, NUM_PLANES` depuis `chess_target_encoding.py` en tête de `data_management.py` (miroir de `from detection_target_encoding import HEATMAP_KEY, SIZE_KEY` déjà présent ligne 41)
- [x] Task 4: `main.py` — dispatch `task_type == "chess_policy_value"` (AC: 5, 6)
  - [x] Nouvelle branche `elif task_type == "chess_policy_value":` (après la branche `detection_centernet` existante, `main.py:156+`) — import paresseux de `ChessPolicyValueStrategy` (miroir exact de la branche `detection_centernet`), instancie avec `loss_params=loss_params` uniquement
- [x] Task 5: `dataset_configs.py` — entrée `CHESS` + fix `validate_config` (AC: 5)
  - [x] Corriger `validate_config` : retirer `"image_size"` de la liste `required` (ligne 31) — devient conditionnel comme `class_names`, décision explicite d'Aymeric (voir Dev Notes § Décision, 2026-07-27) pour ne pas forcer une valeur factice sur un domaine sans image. Ne rien changer d'autre dans cette fonction (la validation structurelle conditionnelle existante d'`image_size`, si présent, reste inchangée).
  - [x] Nouvelle entrée `"CHESS": {...}` dans `DATASET_CONFIGS`, **sans** `image_size` (voir Dev Notes § Entrée dataset_configs pour le contenu exact)
  - [x] Revalider que toutes les configs existantes (`FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_DETECTOR`, `JAX_KEPLER`) passent toujours `validate_config` après le fix (elles fournissent déjà `image_size`, donc aucun changement de comportement attendu — à vérifier, pas juste supposer)
- [x] Task 6: Script de validation autonome (AC: 1, 2, 3, 4, 5, 6)
  - [x] Construire un petit dataset de test via `build_chess_dataset` (Story 9.1, déjà testé) dans un répertoire temporaire, vérifier que `ChessPolicyValueDataset` le charge correctement (shapes, clés, split train/val)
  - [x] Instancier `ChessPolicyValueStrategy` + le modèle réel (Story 9.2, `get_model('chess_cnn_attention_policy_value', num_classes=NUM_MOVES)`) sur un batch réel du dataset de test, vérifier `preprocess_batch` → `compute_loss` → `compute_metrics` de bout en bout, valeurs finies
  - [x] Vérifier `compute_chess_policy_value_loss` sur des tableaux construits à la main (policy/value connus) — valeur numérique exacte attendue, pas seulement "ne plante pas"
  - [x] Vérifier que `get_dataset_config("CHESS")` charge sans erreur et passe `validate_config` (AC: 5) — pas besoin des vrais chunks pour ce test précis, juste la structure du dict
  - [x] Vérifier `git diff --stat -- trainer.py` (ou équivalent) ne montre aucun changement (AC: 2)
  - [x] Si les chunks réels d'Aymeric sont présents (`/home/aobled/Documents/data/chunks/chess_chunk*.npz`, 139 fichiers déjà générés), un test additionnel optionnel peut charger un seul chunk réel pour confirmer la compatibilité — **ne pas lancer un entraînement complet dans cette story** (voir Dev Notes § Portée exacte)

## Dev Notes

### Portée exacte (ne pas dépasser)

- Cette story modifie **5 fichiers existants** : `loss_functions.py`, `task_strategies.py`, `data_management.py`, `main.py`, `dataset_configs.py` (y compris `validate_config` dans ce dernier — extension explicitement autorisée par Aymeric le 2026-07-27, voir Dev Notes § Décision, initialement hors périmètre). Elle ne touche **aucun** autre fichier (`model_library.py`, `chess_target_encoding.py`, `dataset_builder/chess_pgn_dataset_tools.py`, `trainer.py` — zéro modification).
- **Ne pas lancer d'entraînement réel** dans cette story (mémoire projet : demander confirmation avant toute exécution locale lourde). Le script de validation (Task 6) reste à l'échelle d'un petit dataset de test synthétique/quelques chunks, jamais les 139 chunks réels (691 779 positions) qu'Aymeric a déjà générés — les utiliser en lecture pour un test de compatibilité ponctuel (dernier sous-point de Task 6) est acceptable, un entraînement complet dessus ne l'est pas.
- Pas de masquage des coups illégaux, pas de biais géométrique dans l'attention — déjà tranchés en Stories 9.1/9.2, cette story n'y touche pas.
- Pas d'augmentation de données pour le domaine échecs (voir Dev Notes § Chargeur échecs) — absente du PRD/spine, ne pas l'ajouter par réflexe de copier `DetectionDataset`/`CenterNetDetectionDataset`.

### Décision (2026-07-27, revue avec Aymeric) : `image_size` devient optionnel dans `validate_config`, pas contourné

`dataset_configs.py::validate_config` (lignes 16-57) listait `image_size` comme **inconditionnellement requis** (`required = ["num_classes", "image_size", "model_name"]`, ligne 31) — un vrai défaut de conception pour un domaine sans image (chess), pas juste une formalité à contourner. **Décision explicite d'Aymeric** : corriger la validation elle-même plutôt que de faire porter à `CHESS` une valeur `image_size` factice.

**Fix appliqué (touche `validate_config`, fichier partagé)** : retirer `"image_size"` de la liste `required` (ligne 31 → `required = ["num_classes", "model_name"]`). La validation structurelle conditionnelle existante (si `image_size` est présent, doit être un tuple de 2 entiers, lignes 26-28) reste **inchangée** — même traitement que `class_names` (déjà conditionnel, ligne 21), juste rendu cohérent.

**Pourquoi c'est sûr, pas un vrai "revoir la logique globale"** : vérifié dans `data_management.py` que `image_size` est **réellement consommé** par `ChunkManager`/`DetectionDataset`/`CenterNetDetectionDataset` (`config["image_size"]` lu directement dans `get_datasets()`, lignes 621/633/650, et dans les constructeurs des 3 classes). Toutes les configs existantes (`FIGHTERJET_*`, `JAX_DETECTOR`, `JAX_KEPLER` — cette dernière réutilise déjà `image_size` comme `(3197, 1)` pour une séquence 1D, précédent direct du même genre de détournement) continuent de fournir `image_size` sans aucun changement de comportement. Le fix ne fait qu'assouplir la validation pour les **futures** configs qui n'en ont pas besoin — zéro risque de régression sur les configs/modèles existants, pas besoin de les re-tester (contrairement à ce qu'un renommage global de `image_size` en un nom plus générique aurait impliqué — option jugée disproportionnée pour ce cycle, pas retenue).

**Conséquence pour l'entrée `CHESS`** : elle **n'inclut pas** `image_size` du tout (plus besoin de valeur factice) — `ChessPolicyValueDataset` dérive sa shape de `chess_target_encoding.py`, jamais de la config.
- `class_names` n'est vérifié que **si présent** (ligne 21, conditionnel, inchangé) — donc **ne pas inclure `class_names`** dans l'entrée `CHESS` non plus (lister littéralement 4672 faux noms de classes serait absurde) ; `num_classes` reste requis et porte `NUM_MOVES`.
- Les blocs `tpu`/`gpu` avec `micro_batch_size`/`accum_steps` entiers positifs restent obligatoires (lignes 39-49, inchangé) — les fournir normalement.

### Entrée `dataset_configs.py` — contenu exact recommandé

```python
"CHESS": {
    "task_type": "chess_policy_value",
    "num_classes": NUM_MOVES,           # import depuis chess_target_encoding.py, jamais 4672 en dur
    # Pas de "image_size" : optionnel depuis le fix de validate_config (voir Dev Notes) -
    # ChessPolicyValueDataset derive sa shape de chess_target_encoding.py, pas de la config.
    "model_name": "chess_cnn_attention_policy_value",
    # Chemin REEL deja utilise par Aymeric (139 chunks, 691 779 positions deja generes
    # manuellement le 2026-07-27, chunk_size=5000 par defaut de build_chess_dataset) -
    # NE PAS inventer un autre chemin (ex. sous-dossier chunks/chess/chess_targets comme
    # JAX_DETECTOR), ca rendrait ces donnees deja generees inutilisables sans regeneration.
    "output_prefix": f"{DATA_ROOT}/chunks/chess",
    "val_split": 0.1,                   # fraction des CHUNKS (pas des exemples) reservee a la validation
    "loss_params": {
        "policy_weight": 1.0,
        "value_weight": 1.0,
    },
    "dropout_rate": 0.1,
    # tpu/gpu : valeurs de depart raisonnables, non tunees empiriquement (pas d'entrainement
    # reel dans cette story - voir Portee exacte). Suivre la structure JAX_DETECTOR/FIGHTERJET_DETECTION
    # (micro_batch_size/accum_steps/learning_rate/weight_decay/warmup_steps/decay_steps),
    # valeurs a ajuster au premier entrainement reel (story future, hors scope ici).
    "tpu": {"micro_batch_size": 128, "accum_steps": 1, "learning_rate": 4e-4, "weight_decay": 5e-5, "warmup_steps": 200, "decay_steps": 10000},
    "gpu": {"micro_batch_size": 64, "accum_steps": 1, "learning_rate": 2e-4, "weight_decay": 5e-5, "warmup_steps": 200, "decay_steps": 10000},
    "optimizer": "adamw",
    "lr_schedule": "cosine",
    "epochs": 15,
    "patience": 8,
    "eval_batch_size": 64,
    "save_dir": "./checkpoints_chess",
}
```
`NUM_MOVES` importé en tête de `dataset_configs.py` (`from chess_target_encoding import NUM_MOVES`).

### Chargeur échecs (`ChessPolicyValueDataset`) — déviation délibérée de la convention existante

`ChunkManager`/`DetectionDataset`/`CenterNetDetectionDataset` attendent tous des chunks **déjà séparés** par le producteur (`{output_prefix}_train_chunk*.npz` / `{output_prefix}_val_chunk*.npz`, glob `data_management.py:437-438`). **`chess_pgn_dataset_tools.py` (Story 9.1, déjà `done`/review close) ne fait AUCUN split** — un seul flux `{output_prefix}_chunk{N}.npz`, dans l'ordre du fichier PGN source.

Rouvrir la Story 9.1 pour lui ajouter un split côté génération n'est pas justifié pour cette epic (fichier déjà revu, testé, avec son propre `baseline_commit`). **Décision de cette story** : le split train/val se fait **au chargement**, par fraction de **chunks entiers** (pas d'exemple par exemple) — trier tous les chunks (`{output_prefix}_chunk*.npz`), réserver les derniers `val_split` (fraction) comme validation, le reste comme train. Accepté comme limite connue (pas de mélange aléatoire entre train/val au niveau exemple, juste au niveau fichier — suffisant pour une preuve de généricité, pas pour un split rigoureux).

**Pas d'augmentation de données** (contrairement à `DetectionDataset`/`CenterNetDetectionDataset`) — flip/zoom/translation n'ont pas de sens géométrique direct sur un plateau encodé en planes (flip horizontal changerait la géométrie des roques, etc.) ; non demandé par le PRD/spine, ne pas l'ajouter par réflexe de copier le pattern des chargeurs image.

Squelette (mirroir `CenterNetDetectionDataset`, `data_management.py:421-599`, pour l'esprit — chunk-based, `tf.data.Dataset.from_generator`, `.batch().prefetch()` — mais **avec le split et sans augmentation** décrits ci-dessus) :

```python
class ChessPolicyValueDataset:
    def __init__(self, output_prefix: str, batch_size: int = 32, val_split: float = 0.1):
        self.output_prefix = output_prefix
        self.batch_size = batch_size

        all_chunks = sorted(glob.glob(f"{output_prefix}_chunk*.npz"))
        if not all_chunks:
            print(f"❌ ERREUR: Chunks introuvables pour le domaine echecs ! Attendu {output_prefix}_chunk*.npz\n"
                  f"💡 LANCEZ D'ABORD : python dataset_builder/chess_pgn_dataset_tools.py")
            exit(1)

        n_val = max(1, int(len(all_chunks) * val_split)) if len(all_chunks) > 1 else 0
        self.train_chunks = all_chunks[:-n_val] if n_val else all_chunks
        self.val_chunks = all_chunks[-n_val:] if n_val else []

    def create_tf_dataset(self, split='train'):
        chunks = self.train_chunks if split == 'train' else self.val_chunks
        if not chunks:
            raise ValueError(f"Aucun chunk {split} disponible pour le domaine echecs")

        def gen():
            for chunk_path in chunks:
                with np.load(chunk_path) as data:
                    for pos, pol, val in zip(data[POSITION_KEY], data[POLICY_KEY], data[VALUE_KEY]):
                        yield pos, {POLICY_KEY: pol, VALUE_KEY: val}

        output_signature = (
            tf.TensorSpec(shape=(8, 8, NUM_PLANES), dtype=tf.float32),
            {POLICY_KEY: tf.TensorSpec(shape=(), dtype=tf.int32), VALUE_KEY: tf.TensorSpec(shape=(), dtype=tf.float32)},
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if split == 'train':
            ds = ds.shuffle(1000)
        return ds.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    def get_dataset(self):
        return self.create_tf_dataset('train'), (self.create_tf_dataset('val') if self.val_chunks else None)
```

### `ChessPolicyValueStrategy` — pièges spécifiques à cette story

- Le paramètre `images` de `preprocess_batch(self, images, targets, is_training, rng=None)` est en réalité la **position** échecs (nom générique hérité de la signature `TaskStrategy` — même situation que `CenterNetDetectionStrategy`, qui appelle aussi son premier paramètre `images` pour un heatmap). Ne pas renommer le paramètre (romprait la signature abstraite).
- `compute_loss`/`compute_metrics` reçoivent `outputs` = sortie du modèle (Story 9.2, dict `{POLICY_KEY, VALUE_KEY}`) et `targets` = dict issu de `preprocess_batch` (mêmes clés) — cohérence des clés déjà garantie par `chess_target_encoding.py` (AD-18), aucune conversion de nom nécessaire.
- `generate_reports` doit importer `compute_chess_policy_loss`/`compute_chess_value_loss` de `loss_functions.py` (Task 1) — **ne jamais réimplémenter le calcul localement**, c'est exactement ce qu'AD-24/AC3 interdit.

### Project Structure Notes

- 5 fichiers **UPDATE** (aucun nouveau fichier) : `loss_functions.py`, `task_strategies.py`, `data_management.py`, `main.py`, `dataset_configs.py`.
- Points de dispatch `task_type` réels vérifiés dans le code actuel (AD-17 hérité, corrigé du texte du spine parent qui en citait 3 par erreur — voir architecture-chess-2026-07-27) : `main.py:121-166` (sélection `Strategy`) et `data_management.py:613-664` (`get_datasets()`, sélection loader). **Aucun dispatch dans `task_strategies.py` lui-même** — les classes `Strategy` y sont instanciées directement par `main.py`, pas de branche par `task_type` à l'intérieur de ce fichier.

### Testing Standards

Pas de suite de tests automatisée formelle — script autonome (Task 6), même convention que les Stories 9.1/9.2. Fichier suggéré : `tests/test_chess_task_strategy.py`.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` § Epic 9, Story 9.3] — story source, ACs
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md#AD-24,AD-17,AD-18`] — loss composite zero-touch Trainer, dispatch 2 points réels, module d'échange partagé
- [Source: `task_strategies.py:277-384` (`CenterNetDetectionStrategy`)] — précédent direct pour toute la structure de la strategy (export paths, dict outputs/targets, `generate_reports`)
- [Source: `task_strategies.py:78-168` (`ClassificationStrategy`)] — précédent pour `compute_loss`/`compute_metrics` avec `optax.softmax_cross_entropy_with_integer_labels`/accuracy
- [Source: `loss_functions.py:547` (`compute_centernet_loss`)] — précédent exact pour `compute_chess_policy_value_loss` (deux sous-pertes pondérées combinées)
- [Source: `data_management.py:421-599` (`CenterNetDetectionDataset`), `data_management.py:44-` (`ChunkManager`)] — précédent pour la classe loader chunk-based, ET la convention "split côté producteur" dont cette story dévie délibérément (voir Dev Notes)
- [Source: `data_management.py:602-664` (`get_datasets`)] — point de dispatch réel #2
- [Source: `main.py:60-166`] — flux complet (`get_dataset_config` → `num_classes` → `get_datasets` → dispatch `Strategy`), point de dispatch réel #1, plomberie `model_kwargs`
- [Source: `dataset_configs.py:1-58` (`DATA_ROOT`, `validate_config`), `dataset_configs.py:379-558` (`JAX_DETECTOR`, entrée la plus proche)] — piège `image_size` requis, structure d'entrée complète à reprendre
- [Source: vérifié en session, `/home/aobled/Documents/data/chunks/`] — 139 chunks réels déjà générés par Aymeric (`chess_chunk0.npz`...`chess_chunk138.npz`, `output_prefix="/home/aobled/Documents/data/chunks/chess"`) — l'entrée `CHESS` doit pointer exactement là, pas ailleurs
- [Source: `chess_target_encoding.py`, `_bmad-output/implementation-artifacts/9-1-...md`, `9-2-...md`] — contrats amont (NUM_MOVES, NUM_PLANES, POSITION_KEY/POLICY_KEY/VALUE_KEY, modèle `chess_cnn_attention_policy_value`)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5

### Debug Log References

- Task 1 vérifiée à la main immédiatement après implémentation : `compute_chess_policy_loss` sur logits uniformes (zéros) sur 10 classes → `log(10)=2.302585` exact ; `compute_chess_value_loss` sur valeurs connues → `0.29667` exact ; combinaison → `2.599252` exact
- Task 2/3/4 vérifiées individuellement par import + instanciation immédiatement après chaque implémentation (`ChessPolicyValueStrategy`, `data_management` avec `ChessPolicyValueDataset`, `import main`)
- `python3 dataset_configs.py` (bloc `__main__`, Task 5) — les 6 configs (incl. `CHESS`) valident sans erreur, aucune régression sur les 5 configs existantes
- `python3 tests/test_chess_task_strategy.py` — 6/6 tests passés initialement (avant code review)
- `python3 tests/test_chess_target_encoding.py`, `test_chess_pgn_dataset_tools.py`, `test_chess_model.py`, `test_detection_target_encoding.py` — tous repassés, aucune régression
- `python3 -c "import main"` — import propre après toutes les modifications
- **Après code review (2026-07-27)** : `python3 tests/test_chess_task_strategy.py` — 8/8 tests passés (2 tests ajoutés : `test_trainer_create_train_state_for_chess`, remplace `test_trainer_untouched` par `test_trainer_change_is_the_single_authorized_deviation`). Vérifié manuellement avant d'écrire le test permanent : `Trainer(config=CHESS).create_train_state()` réussit via le vrai chemin `main.py` (`num_channels=29`, `382017` paramètres — cohérent avec la mesure de la Story 9.2). Toutes les suites précédentes re-vérifiées une dernière fois, toutes vertes.

### Completion Notes List

- Implémenté `compute_chess_policy_loss`/`compute_chess_value_loss`/`compute_chess_policy_value_loss` (`loss_functions.py`, miroir exact de `compute_centernet_loss`) et `ChessPolicyValueStrategy` (`task_strategies.py`, miroir de `CenterNetDetectionStrategy` — `PolicyAccuracy`/`max`, `generate_reports` réutilise les sous-fonctions de `loss_functions.py` sans réimplémentation, AC3).
- Implémenté `ChessPolicyValueDataset` (`data_management.py`) avec la déviation documentée dans les Dev Notes (split train/val au chargement par fraction de chunks, pas d'augmentation) — vérifié par test que le split est bien disjoint et respecte la fraction demandée.
- Câblé les 2 points de dispatch réels (`main.py`, `data_management.py::get_datasets()`) exactement sur le pattern `detection_centernet` existant, sans toucher `ClassificationStrategy`/`DetectionStrategy`/`CenterNetDetectionStrategy`/`KeplerStrategy` ni leurs loaders (AC6).
- **Extension de périmètre autorisée explicitement par Aymeric (2026-07-27)** : corrigé `validate_config` (`dataset_configs.py`) pour rendre `image_size` conditionnel (comme `class_names`) plutôt que de faire porter à `CHESS` une valeur factice — décision documentée en détail dans les Dev Notes de cette story. Vérifié sans régression : les 5 configs existantes fournissent déjà `image_size`, comportement inchangé pour elles.
- **Bug trouvé et corrigé en cours d'implémentation, non anticipé dans la story** : `print_config()` et le bloc `__main__` de `dataset_configs.py` accédaient `config['class_names']`/`config['image_size']` sans garde (`KeyError` direct pour `CHESS`, qui n'a ni l'un ni l'autre par design). Corrigé avec `.get(..., 'N/A')` — conséquence directe et nécessaire du fix `validate_config`, pas une modification hors scope.
- Entrée `dataset_configs.py::CHESS` pointe exactement sur `{DATA_ROOT}/chunks/chess` — vérifié que les 139 chunks déjà générés par Aymeric (691 779 positions, 2026-07-27) sont directement compatibles avec le schéma attendu (`POSITION_KEY`/`POLICY_KEY`/`VALUE_KEY`, shape `(8,8,29)`), sans regénération nécessaire.
- Les 6 acceptance criteria sont satisfaits et vérifiés par test : AC1 (loss composite pondérée, valeurs exactes vérifiées), AC2 (`trainer.py` non modifié, vérifié par `git diff --stat` contre le `baseline_commit`), AC3 (détail policy/value visible uniquement via `generate_reports`, sous-fonctions réutilisées), AC4 (`PolicyAccuracy`/`max` gate la sauvegarde), AC5 (2 points de dispatch réels câblés, entrée `CHESS` fonctionnelle), AC6 (zéro modification des 4 strategies/loaders existants, vérifié par lecture — aucune ligne touchée dans `ClassificationStrategy`/`DetectionStrategy`/`CenterNetDetectionStrategy`/`KeplerStrategy`).
- **Aucun entraînement réel lancé** dans cette story (conforme à la Portée exacte et à la prudence sur l'exécution locale lourde) — le script de validation reste à l'échelle d'un dataset de test synthétique (24 positions), avec un seul test de lecture ponctuelle sur les vrais chunks (pas d'entraînement dessus).

### File List

- `loss_functions.py` (modifié — 3 nouvelles fonctions, nouveaux imports `optax`/`chess_target_encoding`)
- `task_strategies.py` (modifié — nouvelle classe `ChessPolicyValueStrategy`, nouveaux imports)
- `data_management.py` (modifié — nouvelle classe `ChessPolicyValueDataset`, nouvelle branche dans `get_datasets()`, nouveaux imports)
- `main.py` (modifié — nouvelle branche de dispatch `chess_policy_value`)
- `dataset_configs.py` (modifié — fix `validate_config` (`image_size` optionnel), nouvelle entrée `CHESS`, fix `print_config`/bloc `__main__` pour clés optionnelles, nouvel import `chess_target_encoding`)
- `trainer.py` (modifié en code review — 1 ligne, `num_channels`, écart AC2 autorisé explicitement par Aymeric)
- `tests/test_chess_task_strategy.py` (nouveau, étendu en code review)

### Review Findings

Code review adversarial (2026-07-27, Blind Hunter + Edge Case Hunter + Acceptance Auditor en parallèle). **1 point de fond nécessitant une décision d'Aymeric** (résolu en direct, voir AC2 amendé ci-dessus) + plusieurs findings `high`/`medium`/`low` — dont 3 bugs qui auraient fait planter tout usage réel de `main.py("CHESS")` (`class_names`, `dropout_rate` mal placé, `image_size`), aucun détecté par les tests initiaux de cette story car ils testaient les pièces isolément, jamais le vrai chemin `main.py`/`Trainer`. Corrigés, et un nouveau test (`test_trainer_create_train_state_for_chess`) exerce désormais ce vrai chemin.

**Décision résolue (1) :**
- [x] [Review][Decision] `Trainer` n'a aucun moyen de gérer un nombre de canaux d'entrée différent de 1 (grayscale) ou 3 (RGB) — le domaine échecs a 29 canaux (`NUM_PLANES`). **Résolu : Aymeric a autorisé explicitement le fix minimal d'une ligne** (`self.num_channels = config.get("num_channels", 1 if self.grayscale else 3)`, rétrocompatible) — exactement le scénario anticipé par le PRD (FR-6 "Out of Scope"). AC2 amendé en conséquence dans cette story (voir ci-dessus).

**Patch (11), tous appliqués :**

- [x] [Review][Patch] `main.py:67` : `class_names = config["class_names"]` sans garde — `KeyError` garanti pour `CHESS` (aucun `class_names`) [main.py:67]
- [x] [Review][Patch] Entrée `dataset_configs.py::CHESS` : `dropout_rate` placé au niveau racine au lieu d'être imbriqué sous `tpu`/`gpu` comme toutes les autres configs — `main.py:106` (`backend_config["dropout_rate"]`) aurait levé `KeyError` [dataset_configs.py]
- [x] [Review][Patch] `trainer.py:139` (`image_size = self.config["image_size"]`, sans garde) — `KeyError` garanti puisque l'entrée `CHESS` omettait `image_size` (décision initiale de cette story, revue) ; corrigé en fournissant `image_size=(8,8)` dans la config plutôt qu'en touchant `trainer.py` sur ce point précis [dataset_configs.py]
- [x] [Review][Patch] Tri des chunks lexicographique, pas numérique (`sorted(glob.glob(...))`) — `chunk10` passe avant `chunk2`, fausse silencieusement quels chunks finissent en validation sur les 139 vrais chunks d'Aymeric (>9 chunks) [data_management.py, `ChessPolicyValueDataset.__init__`]
- [x] [Review][Patch] `val_split=0.0` explicite ne désactivait pas réellement le split (`n_val` forcé à 1 dès que >1 chunk) [data_management.py]
- [x] [Review][Patch] Chunks de validation toujours la même "tranche de fin" (jamais mélangés) — biais potentiel si les chunks reflètent un ordre chronologique/par tournoi du PGN source ; corrigé par un mélange à graine fixe (reproductible) avant le split [data_management.py]
- [x] [Review][Patch] `test_real_chunks_compatibility_if_present` codait en dur un chemin personnel au lieu de lire `DATASET_CONFIGS["CHESS"]["output_prefix"]` [tests/test_chess_task_strategy.py]
- [x] [Review][Patch] `test_trainer_untouched` devenu obsolète par la décision ci-dessus (AC2 amendé) — remplacé par un test qui vérifie que le diff de `trainer.py` est **exactement** le fix autorisé, rien de plus [tests/test_chess_task_strategy.py]
- [x] [Review][Patch] Test de `generate_reports` quasi vide de sens (le corps est dans un `try/except` global, donc "ne lève pas d'exception" reste vrai même si le calcul interne est cassé) — corrigé en capturant stdout et en vérifiant le contenu réel affiché [tests/test_chess_task_strategy.py]
- [x] [Review][Patch] Pas de cast explicite du dtype de `POLICY_KEY` dans `gen()` avant de le passer à `tf.data.Dataset.from_generator` (qui exige `int32` strictement selon `output_signature`) — défense explicite ajoutée plutôt que de compter silencieusement sur le producteur [data_management.py]
- [x] [Review][Patch] `validate_config` non testé sur un chemin négatif (une config réellement incomplète doit toujours être rejetée) — ajouté [tests/test_chess_task_strategy.py]

**Écartés (dismiss, 2)** : `ChessPolicyValueDataset.__init__` appelle `exit(1)` sur chunks manquants plutôt que lever une exception catchable — retenu tel quel, miroir exact du pattern déjà établi par `CenterNetDetectionDataset` (pas une régression introduite ici) ; deuxième copie caractère pour caractère de `_get_export_path`/`get_training_state_path` (déjà 2 occurrences dans `CenterNetDetectionStrategy`) — cohérent avec la préférence connue d'Aymeric ("Rule of Three" avant de centraliser).
