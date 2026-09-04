# Audit du pipeline Loss — `jax_supervised_training`

**Date** : 2026-09-03  
**Périmètre** : traçage complet `main.py → Trainer → Strategy → loss_functions.py`, analyse de qualité, faisabilité de la généralisation.  
**Statut** : Palier 1 + Palier 2 implémentés et revus le 2026-09-04 (voir marqueurs ✅ inline). Palier 3 toujours non recommandé.

---

## 1. Cartographie du flux (call graph)

```
main.py
 ├─ config = get_dataset_config(dataset_name)        # dataset_configs.py
 ├─ strategy = STRATEGIES[task_type](**strategy_kwargs)   # task_strategies.py
 │     └─ strategy_kwargs filtrés via STRATEGY_FORWARDED_CONFIG_KEYS
 │         (loss_method, loss_params, label_smoothing…)
 └─ trainer = Trainer(model, config, backend, strategy)
       ├─ _create_train_step()  @jax.jit
       │     └─ strategy.compute_loss(outputs, targets, use_onehot_labels=…)
       └─ _create_eval_step()   @jax.jit
             └─ strategy.compute_loss(outputs, targets, use_onehot_labels=False)
```

**Responsabilités** :

| Couche | Rôle |
|---|---|
| `dataset_configs.py` | Déclare `task_type`, `loss_method` (si dispatch), `loss_params` (hyperparamètres) |
| `main.py` | Instancie la Strategy via `STRATEGIES[task_type]`, lui forwarde les clés config |
| `task_strategies.py` | Dispatch `loss_method` → fonction de loss ; déroule `**loss_params` |
| `loss_functions.py` | Fonctions de loss pures (JAX/optax, aucune dépendance Model/Config) |
| `trainer.py` | Boucle d'optimisation, délègue 100 % du calcul à `strategy.compute_loss` |

---

## 2. Inventaire de `loss_functions.py`

| Fonction / Constante | Domaine | Consommatrice(s) directe(s) |
|---|---|---|
| `compute_grid_loss` | Détection YOLO single-scale | `DetectionStrategy` (`loss_method="grid"`) |
| `compute_grid_loss_multilevel` | Détection YOLO dual-scale 14×14+7×7 | `DetectionStrategy` (`loss_method="grid_multilevel"`) |
| `compute_v7_loss` | Détection anchor-free tri-scale 28+14+7 | `DetectionStrategy` (`loss_method="v7"`) |
| `compute_segmentation_loss` | Segmentation sémantique BCE+Dice | `DetectionStrategy` (`loss_method="segmentation"`) |
| `compute_focal_loss` | Classification multiclasse focal | `ClassificationStrategy` via `CLASSIFICATION_LOSS_FUNCTIONS["focal_loss"]` (✅ 2026-09-04, import lazy supprimé) |
| `compute_classification_cross_entropy_loss` | Classification cross-entropy (wrapper `optax`) | `ClassificationStrategy` via `CLASSIFICATION_LOSS_FUNCTIONS["cross_entropy"]` (✅ 2026-09-04, nouveau) |
| `compute_heatmap_focal_loss` | CenterNet heatmap CornerNet-style | `compute_centernet_loss` (appelée en interne) |
| `compute_size_regression_loss` | CenterNet taille en log-scale | `compute_centernet_loss` (appelée en interne) |
| `compute_centernet_loss` | CenterNet composite heatmap+taille | `CenterNetDetectionStrategy` |
| `compute_chess_policy_loss` | Échecs cross-entropy policy | `ChessPolicyValueStrategy`, `ChessMoveTokenStrategy`, `compute_chess_policy_value_loss`, `compute_chess_token_1_move_loss` |
| `compute_chess_value_loss` | Échecs MSE tête value | `compute_chess_policy_value_loss`, `ChessPolicyValueStrategy.generate_reports` |
| `compute_chess_policy_value_loss` | Échecs composite policy+value | `ChessPolicyValueStrategy` |
| `compute_chess_legal_moves_loss` | Échecs BCE multi-label coups légaux | `ChessLegalMovesStrategy` |
| `compute_chess_token_candidate_loss` | Échecs cross-entropy masquée (50 slots) | `ChessTokenStrategy` |
| `compute_chess_token_1_move_loss` | Échecs 2 têtes from_square+move_type | `ChessTokenOneMoveStrategy` |
| `compute_chess_token_1_move_joint_accuracy` | Métrique jointe token 1-move | `ChessTokenOneMoveStrategy` |
| `CHESS_TOKEN_1_MOVE_NUM_MOVE_TYPES` | Constante 73 (types de coups) | `compute_chess_token_1_move_loss`, `ChessTokenOneMoveStrategy` |

---

## 3. Mapping Config → Strategy → Loss

### 3.1 Configs avec dispatch `loss_method`

Ces strategies supportent plusieurs fonctions de loss sélectionnables depuis la config.

| Config | `task_type` | `loss_method` | Fonction appelée | `loss_params` transmis |
|---|---|---|---|---|
| `FIGHTERJET_CLASSIFICATION` | `classification` | `focal_loss` | `CLASSIFICATION_LOSS_FUNCTIONS["focal_loss"]` → `compute_focal_loss` | `gamma=2.0` |
| `CIFAR10` | `classification` | `cross_entropy` | `CLASSIFICATION_LOSS_FUNCTIONS["cross_entropy"]` → `compute_classification_cross_entropy_loss` (✅ 2026-09-04, lookup dict, plus inline) | *(aucun)* |
| `FIGHTERJET_DETECTION` | `detection` | `segmentation` | `DETECTION_LOSS_FUNCTIONS["segmentation"]` → `compute_segmentation_loss` (✅ 2026-09-04, lookup dict, plus inline) | `bce_weight`, `dice_weight`, `false_positive_penalty` |

Dispatch (✅ 2026-09-04) : deux registres **séparés par domaine** dans `loss_functions.py`, `CLASSIFICATION_LOSS_FUNCTIONS` et `DETECTION_LOSS_FUNCTIONS`, plutôt qu'un seul dict partagé comme envisagé en §5.2.A. Voir §5.2 pour la raison (une revue de code a trouvé qu'un dict unique laisse une `loss_method` du mauvais domaine s'exécuter silencieusement ou échouer avec une erreur peu claire, au lieu de la `ValueError` explicite d'avant).

### 3.2 Configs sans dispatch (strategy mono-loss)

| Config | `task_type` | Fonction appelée | `loss_params` transmis |
|---|---|---|---|
| `JAX_KEPLER` | `kepler` | `optax.softmax_cross_entropy` *(inline)* | *(aucun)* |
| `JAX_DETECTOR` | `detection_centernet` | `compute_centernet_loss` | `heatmap_weight`, `size_weight`, `alpha`, `beta` |
| `CHESS_NO_HISTORY` | `chess_policy_value` | `compute_chess_policy_value_loss` | `policy_weight`, `value_weight` |
| `CHESS_SEARCH_TEACHER` | `chess_policy_value` | `compute_chess_policy_value_loss` | `policy_weight`, `value_weight`, `label_smoothing` |
| `CHESS_LEGAL_MOVES` | `chess_legal_moves` | `compute_chess_legal_moves_loss` | `pos_weight` |
| `CHESS_MOVE_TOKEN` | `chess_move_token` | `compute_chess_policy_loss` *(réutilisée)* | `label_smoothing` |
| `CHESS_TOKEN` | `chess_token` | `compute_chess_token_candidate_loss` | `label_smoothing` |
| `CHESS_TOKEN_1_MOVE` | `chess_token_1_move` | `compute_chess_token_1_move_loss` | `from_square_weight`, `move_type_weight`, `label_smoothing` |

---

## 4. Constat qualité — anomalies identifiées

### 4.1 ✅ Import mort dans `trainer.py` — corrigé le 2026-09-04

**Fichier** : `trainer.py:27`

```python
from loss_functions import compute_grid_loss   # ← jamais appelée dans ce fichier
```

`Trainer` délègue 100 % du calcul de la loss à `strategy.compute_loss()` depuis la migration Strategy. Cet import est un vestige du code pré-Strategy — il n'est **utilisé nulle part** dans `trainer.py`. Résidu sans impact fonctionnel, mais trompeur pour un lecteur qui s'attendrait à une utilisation directe.

**Correction** : ligne supprimée.

---

### 4.2 ✅ Clé `loss_method` morte dans `KeplerStrategy` — corrigé le 2026-09-04

**Fichier config** : `dataset_configs.py:310` — `"loss_method": "cross_entropy"`  
**Fichier strategy** : `task_strategies.py:402-412`

`KeplerStrategy.compute_loss` ignore `self.loss_method` : elle appelle toujours `optax.softmax_cross_entropy[_with_integer_labels]` inline, sans dispatch. La clé est pourtant forwardée via `STRATEGY_FORWARDED_CONFIG_KEYS["kepler"]` et stockée dans `self.loss_method`.

**Options** :
- Option A (cohérence) : ajouter le même dispatch if/elif que `ClassificationStrategy` — `KeplerStrategy` deviendrait capable de focal_loss comme la classification.
- Option B (simplicité) : supprimer `loss_method` de `JAX_KEPLER` config et de `STRATEGY_FORWARDED_CONFIG_KEYS["kepler"]`, enlever `self.loss_method` du `__init__` — Kepler ne supporte qu'une seule loss par conception.

Option B appliquée : `loss_method` retiré de `KeplerStrategy.__init__`, de `JAX_KEPLER` (`dataset_configs.py`) et de `STRATEGY_FORWARDED_CONFIG_KEYS["kepler"]`. Kepler reste mono-loss par conception.

---

### 4.3 ✅ Import lazy de `compute_focal_loss` — résolu le 2026-09-04

Résolu comme effet de bord du Palier 2 : le if/elif de `ClassificationStrategy.compute_loss` a disparu au profit du lookup `CLASSIFICATION_LOSS_FUNCTIONS.get(self.loss_method)`, donc l'import lazy qu'il contenait n'existe plus. `compute_focal_loss` est maintenant référencé une seule fois, au niveau module, dans `loss_functions.py` (définition de `CLASSIFICATION_LOSS_FUNCTIONS`) — `task_strategies.py` ne l'importe plus du tout directement.

---

### 4.4 ℹ️ Duplication partielle `ClassificationStrategy` ↔ `compute_chess_policy_loss`

La docstring de `compute_chess_policy_loss` (`:565`) reconnaît explicitement : *"Même formule que `ClassificationStrategy.compute_loss`"*. La duplication est **intentionnelle** (AD-24) : chess a besoin d'un primitif composable (`compute_chess_policy_value_loss`, `compute_chess_token_1_move_loss`), tandis que `ClassificationStrategy` gère aussi `mixup_alpha`/`use_onehot_labels` qui sont sans objet dans le domaine échecs. La séparation est justifiée, mais reste une obligation de synchronisation si la formule de base évolue.

---

### 4.5 ℹ️ Fonctions YOLO (`compute_grid_loss`, `multilevel`, `v7`) sans consommateur actif

Ces trois fonctions sont disponibles via `DetectionStrategy` (`loss_method="grid"`, `"grid_multilevel"`, `"v7"`, toutes trois dans `DETECTION_LOSS_FUNCTIONS` depuis le 2026-09-04) mais aucune config active (`FIGHTERJET_DETECTION`) ne les utilise — `FIGHTERJET_DETECTION` utilise `"segmentation"`. Elles restent du code dead-end côté configs. À garder tant que `FIGHTERJET_DETECTION` n'est pas archivé, ou documenter explicitement leur statut expérimental.

---

## 5. Analyse de la proposition : `loss_fn` dans `dataset_configs.py`

### 5.1 L'état actuel EST déjà partiellement ce pattern

Le mécanisme `loss_method` + `loss_params` dans `dataset_configs.py` **est** le pattern demandé, pour les stratégies qui supportent le dispatch. Comparaison :

| Dimension | `model_name` (modèle) | `loss_method` + `loss_params` (loss) |
|---|---|---|
| Déclaré dans config | ✅ `dataset_configs.py` | ✅ `dataset_configs.py` |
| Dispatch centralisé | ✅ `MODELS` dict (`model_library.py`) | ⚠️ `if/elif` inline dans chaque Strategy |
| Paramètres | ✅ `MODEL_FORWARDED_CONFIG_KEYS` + `build_kwargs_from_config` | ✅ `**self.loss_params` |
| Applicable partout | ✅ tous les domaines | ⚠️ seulement `classification`, `detection`, `kepler` |

### 5.2 Ce qui manque pour une généralisation complète

**A. Un registre `LOSS_FUNCTIONS` dans `loss_functions.py`** (équivalent `MODELS`)

Sketch initial de cette section (un seul dict partagé entre tous les `task_type`) :

```python
LOSS_FUNCTIONS = {
    "cross_entropy": None,          # inline optax, pas une fonction de ce module
    "focal_loss": compute_focal_loss,
    "segmentation": compute_segmentation_loss,
    "grid": compute_grid_loss,
    "grid_multilevel": compute_grid_loss_multilevel,
    "v7": compute_v7_loss,
    "centernet": compute_centernet_loss,
    "chess_policy_value": compute_chess_policy_value_loss,
    # etc.
}
```

Cela remplacerait les `if/elif` actuels dans `ClassificationStrategy` et `DetectionStrategy` et rendrait le dispatch testable isolément.

**✅ Correction apportée en revue (2026-09-04)** : ce sketch à un seul dict a été implémenté tel quel dans un premier temps, puis corrigé avant merge — un `LOSS_FUNCTIONS` unique partagé entre `ClassificationStrategy` et `DetectionStrategy` supprime silencieusement la validation par domaine. Concrètement : une config détection avec `loss_method="cross_entropy"` par erreur de config s'exécutait sans erreur (mauvaise loss, mauvaise forme de sortie) au lieu de lever la `ValueError` explicite d'avant ; le sens inverse levait un `TypeError` opaque au lieu du message clair nommant le domaine. Ce risque n'était pas anticipé par le tableau d'obstacles B ci-dessous. Fix retenu : deux registres séparés, `CLASSIFICATION_LOSS_FUNCTIONS` et `DETECTION_LOSS_FUNCTIONS`, chacun n'exposant que les `loss_method` valides pour son domaine — le lookup rate (et la `ValueError` se déclenche) exactement comme avant pour toute méthode du mauvais domaine. Trouvé par la couche Edge Case Hunter de `bmad-code-review`, régression couverte par `tests/test_loss_dispatch.py` (6 cas, dont les 2 cas cross-domaine).

**B. Pourquoi une généralisation *totale* (toutes les strategies) est difficile**

Les fonctions de loss ont des signatures hétérogènes qui empêchent un adaptateur `loss_fn(outputs, targets, **loss_params)` universel :

| Cas | Obstacle |
|---|---|
| `compute_focal_loss` | `use_onehot_labels` est un **signal runtime** (vient de `preprocess_batch`), pas un hyperparamètre config |
| `compute_chess_token_candidate_loss` | La target est un dict `{"candidate_label", "candidate_mask"}` — dépaquetée *dans* la Strategy avant d'appeler la loss (3 args positionnels, pas 2) |
| `compute_chess_legal_moves_loss` | `pos_weight` isolé d'un dict `loss_params` dans Strategy (convention légèrement différente) |
| `compute_centernet_loss` | Outputs et targets sont des dicts, pas des tenseurs — transparent si la strategy débalise correctement |

La plupart des cas sont gérables via `**loss_params`, **sauf** `use_onehot_labels` (signal runtime).

### 5.3 Évaluation risques/bénéfices

| Scénario | Bénéfice | Risque / Effort |
|---|---|---|
| **Rien changer** | — | Stale import + KeplerStrategy incohérente persistent |
| **Fix ciblé** (§ 4.1 + 4.2) | Nettoie 2 anomalies confirmées, zéro régression | Faible (2 lignes + 1 suppression) |
| **Registre `LOSS_FUNCTIONS`** | Dispatch testable, pas de if/elif dans les strategies, cohérent avec `MODELS` | Moyen — refactorer `ClassificationStrategy` + `DetectionStrategy`, ajouter un test |
| **Généralisation totale** (toutes les strategies + config) | Config 100 % auto-descriptive | Élevé — signatures hétérogènes, `use_onehot_labels` reste dans Strategy quoi qu'il arrive, risque de sur-abstraction |

### 5.4 Recommandation

**Palier 1 — immédiat, sans risque — ✅ implémenté le 2026-09-04** :
- Supprimer l'import mort `compute_grid_loss` dans `trainer.py`
- Corriger `KeplerStrategy` (option B : supprimer la clé morte `loss_method`)

**Palier 2 — si refactoring souhaité — ✅ implémenté le 2026-09-04 (avec correction, voir §5.2.A)** :
- Ajouter un registre de dispatch dans `loss_functions.py` — implémenté comme **deux dicts séparés par domaine** (`CLASSIFICATION_LOSS_FUNCTIONS`, `DETECTION_LOSS_FUNCTIONS`), pas un seul `LOSS_FUNCTIONS` partagé comme sketché initialement
- Remplacer les `if/elif` de `ClassificationStrategy.compute_loss` et `DetectionStrategy.compute_loss` par un lookup dict scopé à leur domaine
- `loss_method` reste dans `dataset_configs.py` (sa place naturelle) et est forwarded tel quel — aucun changement de config
- Tests de régression ajoutés : `tests/test_loss_dispatch.py`

**Palier 3 — non recommandé pour l'instant** :
- Étendre le mécanisme aux strategies mono-loss (chess/centernet) — la séparation actuelle "mono-loss sans `loss_method`" est intentionnelle et documentée (cf. commentaire JAX_DETECTOR dans dataset_configs.py:538). L'unifier apporterait une cohérence formelle mais effacerait une distinction qui rend le code plus lisible (pas de dispatch à comprendre pour chess).

---

## 6. Résumé exécutif

| Catégorie | Verdict |
|---|---|
| Architecture globale du pipeline | ✅ Solide — séparation claire loss_functions / Strategy / Trainer |
| `loss_functions.py` (fonctions pures) | ✅ Bien isolé, aucune dépendance externe (Model/Config/Trainer) |
| `loss_params` dans `dataset_configs.py` | ✅ Le pattern "hyperparamètres de loss en config" existe déjà et fonctionne |
| `trainer.py` | ✅ Import mort `compute_grid_loss` supprimé (2026-09-04) |
| `KeplerStrategy` | ✅ Clé `loss_method` morte retirée (2026-09-04) |
| Proposition généralisation | ✅ Implémentée au palier 2, corrigée en deux registres par domaine (2026-09-04, voir §5.2.A) — palier 3 toujours non justifié |

