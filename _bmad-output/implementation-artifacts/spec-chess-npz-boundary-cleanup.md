---
title: 'Retirer chess_target_encoding.py et le générateur npz dupliqué de jax_supervised_training'
type: 'refactor'
created: '2026-08-01'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/_bmad-output/specs/spec-chess-npz-boundary-cleanup/SPEC.md']
baseline_commit: 'aac08da7db5d0b281db1efdab7b26e9d7d74fa67'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `jax_supervised_training` garde `chess_target_encoding.py` et `dataset_builder/chess_pgn_dataset_tools.py` alors que `chess_ai` (split du 2026-07-30) possède désormais la génération réelle des `.npz` échecs — deux sources de vérité pour le même format, le risque de dérive silencieuse qu'AD-18 de ce projet existe pour éviter. C'est aussi la cause racine, jamais fermée, d'un `ModuleNotFoundError: No module named 'chess'` déjà survenu en production sur un run `JAX_CLASSIFICATION` (le correctif du 2026-07-31 n'a fait que retarder le problème, pas le fermer).

**Approach:** Supprimer les deux fichiers. Remplacer les 7 constantes qu'ils exportaient (`NUM_MOVES`, `NUM_PLANES`, `NUM_POSITION_PLANES`, `BOARD_SIZE`, `POSITION_KEY`, `POLICY_KEY`, `VALUE_KEY`) par leurs valeurs littérales, directement aux points d'usage réels dans `dataset_configs.py`, `task_strategies.py`, `loss_functions.py`, `data_management.py`, `model_library.py` — cohérent avec le fait qu'aucun autre domaine de `dataset_configs.py` n'importe jamais ses constantes de forme. Retirer `chess` de `requirements.txt`. Supprimer 4 tests devenus impossibles à maintenir sans le code retiré (garder `tests/test_chess_model.py`, imports corrigés).

## Boundaries & Constraints

**Always:**
- Les 7 valeurs restent strictement identiques (`NUM_MOVES=4672`, `NUM_PLANES=29`, `NUM_POSITION_PLANES=19`, `BOARD_SIZE=8`, clés `"position"`/`"policy"`/`"value"`) — seule leur forme change (littéral au lieu d'import).
- `tests/test_chess_model.py` doit continuer à passer, assertions identiques, après correction de ses imports.
- Aucun fichier de ce repo n'importe `python-chess`, même indirectement, après ce changement.
- La logique métier de `ChessPolicyValueStrategy`, `ChessPolicyValueDataset`, `ChessCnnAttentionPolicyValue` ne change pas — seuls imports et littéraux bougent.

**Ask First:** si l'un des 5 fichiers consommateurs révèle un usage de `chess_target_encoding.py` non catalogué ci-dessous (au-delà des 7 constantes), HALT et demander avant d'improviser.

**Never:** garder une version réduite de `chess_target_encoding.py` (déjà tranché : suppression complète, pas un renommage) ; recentraliser `POSITION_KEY`/`POLICY_KEY`/`VALUE_KEY` dans un nouveau module (décision explicite : littéraux en dur, voir tâche `deferred-work.md`) ; archiver les fichiers/tests supprimés (git history suffit) ; toucher `docs/contract-chess-ai-training-interface.md`, `HANDOFF.md` (chess_ai), ou le repo `chess_ai` lui-même ; relancer un entraînement ou une revalidation de modèle.

</frozen-after-approval>

## Code Map

- `chess_target_encoding.py` -- supprimé (module d'encodage + constantes, plus nécessaire côté training)
- `dataset_builder/chess_pgn_dataset_tools.py` -- supprimé (générateur npz dupliqué, canonique côté `chess_ai`)
- `dataset_configs.py:10` (import), `:593,600,604` (`CHESS`), `:695-697` (`CHESS_NO_HISTORY`), `:741-743` (`CHESS_NAKAMURA_NO_HISTORY`) -- littéraux à écrire
- `task_strategies.py:7` (import), `:474-526` (`ChessPolicyValueStrategy`, 6 usages `POLICY_KEY`/`VALUE_KEY`) -- littéraux
- `loss_functions.py:6` (import), `:588-591` (`compute_chess_policy_value_loss`) -- littéraux
- `data_management.py:42` (import), `:619,627` (défaut `num_planes`), `:660-686` (`ChessPolicyValueDataset`), `:770` (défaut `config.get`) -- littéraux
- `model_library.py:16` (import), `:845-892` (docstring + asserts `ChessCnnAttentionPolicyValue`), `:961` (retour dict) -- littéraux
- `requirements.txt:13` -- ligne `chess` à retirer
- `tests/test_chess_model.py:17` -- import à corriger (littéraux au lieu de `chess_target_encoding`)
- `tests/test_chess_target_encoding.py`, `tests/test_chess_pgn_dataset_tools.py`, `tests/test_chess_target_encoding_lazy_import.py`, `tests/test_chess_task_strategy.py` -- supprimés (testent du code retiré, ou dépendent de `build_chess_dataset` — voir Design Notes)
- `_bmad-output/implementation-artifacts/deferred-work.md` -- 2 entrées à ajouter

## Tasks & Acceptance

**Execution:**
- [x] `chess_target_encoding.py` -- supprimer le fichier -- plus aucun consommateur n'a besoin du codec, seulement de littéraux désormais inline
- [x] `dataset_builder/chess_pgn_dataset_tools.py` -- supprimer le fichier -- dupliqué côté `chess_ai`, seule source de vérité désormais
- [x] `dataset_configs.py` -- retirer l'import `chess_target_encoding`, écrire `NUM_MOVES=4672`/`NUM_PLANES=29`/`NUM_POSITION_PLANES=19` en littéraux dans les 3 entrées chess -- ferme la dernière dépendance chess au chargement du module
- [x] `task_strategies.py` -- retirer l'import, remplacer `POLICY_KEY`/`VALUE_KEY` par `"policy"`/`"value"` littéraux -- idem
- [x] `loss_functions.py` -- retirer l'import, remplacer `POLICY_KEY`/`VALUE_KEY` par littéraux -- idem
- [x] `data_management.py` -- retirer l'import, remplacer `POSITION_KEY`/`POLICY_KEY`/`VALUE_KEY`/`NUM_PLANES`/`BOARD_SIZE` par littéraux (défaut `num_planes=29`) -- idem
- [x] `model_library.py` -- retirer l'import, remplacer `POLICY_KEY`/`VALUE_KEY`/`BOARD_SIZE` par littéraux -- idem
- [x] `requirements.txt` -- retirer la ligne `chess` -- plus aucun fichier de ce repo n'importe `python-chess`
- [x] `tests/test_chess_model.py` -- remplacer l'import `chess_target_encoding` par les 4 littéraux (`NUM_MOVES=4672`, `NUM_PLANES=29`, `POLICY_KEY="policy"`, `VALUE_KEY="value"`) -- seul changement, le test reste vert à l'identique
- [x] `tests/test_chess_target_encoding.py`, `tests/test_chess_pgn_dataset_tools.py`, `tests/test_chess_target_encoding_lazy_import.py`, `tests/test_chess_task_strategy.py` -- supprimer les 4 fichiers -- testent du code qui n'existe plus ou dépendent de `build_chess_dataset` (décision explicite Aymeric : suppression pure, pas d'archive, git history suffit)
- [x] `_bmad-output/implementation-artifacts/deferred-work.md` -- ajouter : (1) les 3 clés de dict codées en dur plutôt que centralisées, à revisiter si friction réelle observée ; (2) plus de test de non-régression sur `ChessPolicyValueStrategy`/`ChessPolicyValueDataset` après cette suppression, à couvrir par la future epic à 2 modèles côté `chess_ai`

**Acceptance Criteria:**
- Given un environnement Python sans le paquet `chess` installé, when `import main` puis `import dataset_configs, model_library, loss_functions, data_management, task_strategies` sont exécutés, then aucune erreur d'import ne survient.
- Given `python3 tests/test_chess_model.py`, when exécuté après les changements, then tous les tests passent avec les mêmes assertions qu'avant.
- Given `grep -rn "chess_target_encoding\|chess_pgn_dataset_tools" --include="*.py" .`, when exécuté après les changements, then aucune occurrence ne subsiste dans le code actif.
- Given les 3 entrées chess de `dataset_configs.py`, when inspectées après les changements, then `num_classes`/`num_channels`/`input_shape` portent exactement les mêmes valeurs qu'avant.

## Spec Change Log

<!-- vide -->

## Design Notes

Deux décisions déjà arbitrées avec Aymeric à ne pas rouvrir pendant l'implémentation :
1. **Pas de module de constantes de remplacement.** Vérifié qu'aucune autre entrée de `dataset_configs.py` (CIFAR10, JAX_DETECTOR, etc.) n'importe `num_classes`/`input_shape` — toutes en littéral. Les 3 entrées chess étaient l'exception à corriger, pas un précédent à prolonger.
2. **`tests/test_chess_task_strategy.py` supprimé, pas adapté.** Il dépend de `build_chess_dataset` (supprimé) pour générer ses fixtures. Un remplacement par un générateur synthétique inline a été envisagé puis écarté par Aymeric : la surface qu'il teste (`ChessPolicyValueStrategy`/`ChessPolicyValueDataset`) sera vraisemblablement remplacée par la future architecture à 2 modèles côté `chess_ai` — investir dans son maintien maintenant serait prématuré.

## Verification

**Commands:**
- `python3 tests/test_chess_model.py` -- expected: tous les tests passent, identique à avant
- `python3 -c "import main"` -- expected: import propre
- `python3 dataset_configs.py` (bloc `__main__` existant) -- expected: les 8 configs (dont les 3 chess) valident sans erreur
- `grep -rln "chess_target_encoding\|chess_pgn_dataset_tools" --include="*.py" .` -- expected: aucune sortie
- `grep -n "^chess$" requirements.txt` -- expected: aucune sortie
- Import de `dataset_configs`, `model_library`, `loss_functions`, `data_management`, `task_strategies` (et `main`) dans un process où `sys.modules["chess"] = None` (même technique que l'ancien `tests/test_chess_target_encoding_lazy_import.py`, avant suppression) -- expected: aucune `ModuleNotFoundError`

## Suggested Review Order

**Frontière fermée : configs échecs en littéraux**

- Les 3 entrées échecs perdent leur import symbolique — mêmes valeurs (4672/29/19), écrites en dur comme tout le reste du fichier.
  [`dataset_configs.py:585`](dataset_configs.py#L585)

**Consommateurs : dict-keys en dur (`policy`/`value`/`position`)**

- Seule classe à double tête/double loss du projet — vérifie que le retrait de l'import n'a pas touché la logique.
  [`task_strategies.py:469`](task_strategies.py#L469)

- Loss composite pondérée, miroir de `compute_centernet_loss` — clés littérales, calcul inchangé.
  [`loss_functions.py:581`](loss_functions.py#L581)

- Chargeur chess, le point le plus délicat — défaut `num_planes=29` maintenant documenté au lieu de dériver silencieusement de l'import.
  [`data_management.py:774`](data_management.py#L774)

- Définition de la classe modèle — asserts `BOARD_SIZE`/clés remplacés par littéraux, docstring recentrée sur le contrat `.npz` côté chess_ai.
  [`model_library.py:837`](model_library.py#L837)

**Garde-fou de non-régression ajouté en revue**

- Remplace `test_chess_target_encoding_lazy_import.py` — garantit qu'aucun domaine non-échecs ne dépend, même indirectement, de `python-chess`.
  [`test_no_chess_dependency.py:1`](../../tests/test_no_chess_dependency.py#L1)

**Dette explicitement tracée**

- Documente le hardcode des 3 clés et la perte de couverture sur `ChessPolicyValueStrategy`/`ChessPolicyValueDataset` — décisions déjà arbitrées, pas des oublis.
  [`deferred-work.md:538`](deferred-work.md#L538)

**Peripherals**

- `tests/test_chess_model.py` -- seul test échecs restant, imports remplacés par littéraux, assertions identiques.
- `requirements.txt` -- ligne `chess` retirée.
- `chess_target_encoding.py`, `dataset_builder/chess_pgn_dataset_tools.py`, `tests/test_chess_target_encoding.py`, `tests/test_chess_pgn_dataset_tools.py`, `tests/test_chess_target_encoding_lazy_import.py`, `tests/test_chess_task_strategy.py` -- supprimés.
