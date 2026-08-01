---
id: SPEC-chess-npz-boundary-cleanup
companions: []
sources: []
---

> **Canonical contract.** This SPEC is the complete, preservation-validated contract for what to build, test, and validate. Source documents are for traceability only.

# Réduire jax_supervised_training au rôle lecture-npz + training pour le domaine échecs

## Why

**Mandat à respecter, issu d'une décision d'architecture actée le 2026-08-01** (session Winston, suite au split `chess_ai` du 2026-07-30) : `chess_ai` possède désormais la génération complète des `.npz` échecs (codec `encode_position`/`move_to_index`/`index_to_move` + la dépendance `python-chess` qui va avec) et le jeu ; `jax_supervised_training` se contente de lire les `.npz` et d'entraîner. `HANDOFF.md` et `docs/contract-chess-ai-training-interface.md` (les deux copies) disent encore l'inverse — ce spec corrige le code pour refléter la vraie frontière, avant que la documentation ne suive.

C'est aussi la fermeture définitive d'un trou trouvé deux fois sans être root-causé : la revue de la Story 9.2 avait flagué `chess` absent de `requirements.txt` (menace AD-21), et le correctif du 2026-07-31 (import paresseux) avait retardé le problème — mais le vrai chemin d'entraînement (`main.py`/`Trainer`) n'a jamais eu besoin du codec, seulement de 7 constantes de schéma. Ce spec ferme la cause racine plutôt que le symptôme.

**Décision affinée en cours de spec** (revue Winston/Aymeric) : garder un fichier `chess_target_encoding.py` réduit aux constantes aurait reproduit, à plus petite échelle, exactement le risque de duplication/dérive silencieuse que ce spec cherche à éliminer (le même argument que CAP-2). Vérifié qu'aucun autre domaine de `dataset_configs.py` (CIFAR10, JAX_DETECTOR, etc.) n'importe jamais `num_classes`/`input_shape` — tous en littéral — donc l'entrée `CHESS` était l'exception à corriger, pas le précédent à prolonger.

## Capabilities

- **CAP-1**
  - **intent:** `chess_target_encoding.py` (racine du repo) est supprimé entièrement. `NUM_MOVES`, `NUM_PLANES`, `NUM_POSITION_PLANES`, `BOARD_SIZE` deviennent des littéraux directement dans les entrées `CHESS`/`CHESS_NO_HISTORY`/`CHESS_NAKAMURA_NO_HISTORY` de `dataset_configs.py` (`num_classes`, `input_shape`), comme le fait déjà toute autre entrée de ce fichier. `POSITION_KEY`/`POLICY_KEY`/`VALUE_KEY` (`"position"`/`"policy"`/`"value"`) sont codés en dur aux points d'usage réels dans `task_strategies.py`, `loss_functions.py`, `data_management.py`, `model_library.py` — décision explicite d'Aymeric : observer l'impact réel avant toute abstraction, pas de risque de dérive concret identifié pour 3 chaînes stables.
  - **success:** `chess_target_encoding.py` n'existe plus ; `grep -rn "chess_target_encoding" *.py` ne retourne plus aucun import ; les 5 fichiers consommateurs (`dataset_configs.py`, `task_strategies.py`, `loss_functions.py`, `data_management.py`, `model_library.py`) et `main.py` s'importent et s'exécutent sans `python-chess` installé, pour tout `task_type` y compris `CHESS`/`CHESS_NO_HISTORY`/`CHESS_NAKAMURA_NO_HISTORY` (échec attendu uniquement à la lecture des `.npz` réels, jamais à l'import) ; les valeurs numériques et les 3 clés de dict restent identiques à avant (`4672`, `29`/`19`, `8`, `"position"`/`"policy"`/`"value"`). Vérifié en permanence par `tests/test_no_chess_dependency.py` (garde-fou ajouté en revue, remplace `tests/test_chess_target_encoding_lazy_import.py`).

- **CAP-2**
  - **intent:** `dataset_builder/chess_pgn_dataset_tools.py` est retiré de `jax_supervised_training` — la génération du dataset échecs n'a plus qu'une seule source de vérité, côté `chess_ai` (déjà copié là le 2026-08-01).
  - **success:** le fichier n'existe plus dans ce repo ; aucun fichier restant de `jax_supervised_training` ne l'importe.

## Constraints

- Non-régression stricte : `tests/test_chess_model.py` doit continuer à passer après le remplacement des imports par des littéraux (mêmes valeurs, source différente).
- `tests/test_chess_target_encoding.py`, `tests/test_chess_pgn_dataset_tools.py` et `tests/test_chess_target_encoding_lazy_import.py` sont supprimés (pas simplifiés) — le code qu'ils testent n'existe plus du tout dans ce repo, rien à garder en garde-fou. `tests/test_no_chess_dependency.py` (nouveau, ajouté en revue) reprend la seule garantie encore pertinente de `test_chess_target_encoding_lazy_import.py` — que les modules non-échecs restent importables sans `python-chess`.
- **Mis à jour pendant l'implémentation** : `tests/test_chess_task_strategy.py` s'est révélé dépendre de `build_chess_dataset` (CAP-2) pour 2 de ses 6 tests — découvert seulement à l'implémentation, non anticipé par ce SPEC. Décision d'Aymeric : supprimer le fichier entièrement plutôt que d'adapter les 4 tests indépendants — `ChessPolicyValueStrategy`/`ChessPolicyValueDataset` seront vraisemblablement remplacés par la future architecture à 2 modèles côté `chess_ai`, investir dans le maintien de ce test maintenant serait prématuré. Conséquence assumée : plus aucune couverture de non-régression sur ces deux classes dans ce repo (documenté dans `deferred-work.md`).
- Ne pas committer tel quel le correctif d'import paresseux du 2026-07-31 (`_bmad-output/implementation-artifacts/spec-lazy-chess-import.md`, statut `done` mais non commité) — ce spec le supersede entièrement.
- Les valeurs des 7 constantes ne changent pas (`NUM_MOVES=4672`, `NUM_PLANES=29`, `NUM_POSITION_PLANES=19`, `BOARD_SIZE=8`, clés `"position"`/`"policy"`/`"value"`) — seule leur forme (littéral vs import) change.
- Consigner dans `deferred-work.md` le choix de coder `POSITION_KEY`/`POLICY_KEY`/`VALUE_KEY` en dur plutôt que de les centraliser, comme item à revisiter si une vraie friction (typo, duplication gênante) apparaît en pratique — pas par principe.

## Non-goals

- Ne touche pas au repo `chess_ai` — chantier séparé, traité après celui-ci dans une autre session.
- Ne modifie pas `docs/contract-chess-ai-training-interface.md` ni `HANDOFF.md` (chess_ai) — mise à jour documentaire différée, actée séparément une fois le PRD `chess_ai` posé.
- Ne relance aucun entraînement ni revalidation de modèle — changement de surface training-side uniquement (imports/constantes), pas de logique d'entraînement touchée.
- Ne centralise pas les 3 clés de dict dans un nouveau module — décision explicite de les coder en dur pour l'instant (voir Constraints, item deferred-work.md).

## Success signal

`main.py` s'exécute pour tout `task_type` existant (`CIFAR10`, `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_DETECTOR`, `KEPLER`, `CHESS`, `CHESS_NO_HISTORY`, `CHESS_NAKAMURA_NO_HISTORY`) dans un environnement sans `python-chess` installé, sauf pour les 3 configs chess qui échouent proprement à la lecture des `.npz` réels (absents dans cet environnement de test), jamais à l'import — la preuve que plus aucun domaine, chess compris à l'import, ne paie de coût de dépendance `python-chess`.
