---
title: 'Import chess paresseux dans chess_target_encoding.py'
type: 'bugfix'
created: '2026-07-31'
status: 'done'
route: 'one-shot'
---

# Import chess paresseux dans chess_target_encoding.py

## Intent

**Problem:** `chess_target_encoding.py` faisait `import chess` (python-chess) au niveau module, ce qui forçait cette dépendance tierce pour TOUT run de `main.py` (CIFAR10, FIGHTERJET_*, KEPLER, pas seulement CHESS) via l'import inconditionnel de `POLICY_KEY`/`VALUE_KEY`/`BOARD_SIZE` dans `model_library.py`. Un run réel `JAX_CLASSIFICATION` sur Colab a cassé avec `ModuleNotFoundError: No module named 'chess'` pour cette raison.

**Approach:** `from __future__ import annotations` (PEP 563) pour différer l'évaluation des annotations de type (`chess.Board`/`chess.Move` dans les signatures), `import chess` déplacé à l'intérieur de chaque fonction qui l'utilise réellement, constantes module-level dérivées de `chess.*` remplacées par un entier fixe documenté ou calculées localement. Corrigé à la source (un seul fichier) plutôt que dans chaque consommateur (`model_library.py`, `task_strategies.py`, `dataset_configs.py`, `loss_functions.py` en bénéficient tous automatiquement, vérifié).

## Suggested Review Order

**Architecture de l'import paresseux**

- Docstring de module - documente le choix (pourquoi optionnel, quels consommateurs en bénéficient) et sert de référence unique pour les commentaires inline.
  [`chess_target_encoding.py:1`](../../chess_target_encoding.py#L1)

- `from __future__ import annotations` - permet les type hints `chess.Board`/`chess.Move` sans import au chargement.
  [`chess_target_encoding.py:92`](../../chess_target_encoding.py#L92)

- `_NUM_PIECE_TYPES = 6` - remplace `len(_PIECE_TYPES)` par un entier fixe documenté, découple `NUM_POSITION_PLANES` de `chess` au chargement.
  [`chess_target_encoding.py:110`](../../chess_target_encoding.py#L110)

**Sites d'import local (le risque de régression comportementale)**

- `encode_position` - `import chess` local + tuple `piece_types` reconstruite localement, assert de cohérence avec `_NUM_PIECE_TYPES`.
  [`chess_target_encoding.py:117`](../../chess_target_encoding.py#L117)

- `_underpromotion_pieces()` - constante `_UNDERPROMOTION_PIECES` convertie en fonction avec `import chess` local, `@functools.lru_cache` pour éviter la réallocation à chaque coup.
  [`chess_target_encoding.py:208`](../../chess_target_encoding.py#L208)

- `move_to_index` - `import chess` local.
  [`chess_target_encoding.py:229`](../../chess_target_encoding.py#L229)

- `index_to_move` - `import chess` local.
  [`chess_target_encoding.py:267`](../../chess_target_encoding.py#L267)

**Vérification (preuve de non-régression)**

- Test permanent : simule `chess` absent (subprocess dédié) et vérifie que `chess_target_encoding` + les 4 consommateurs top-level (`model_library`, `task_strategies`, `dataset_configs`, `loss_functions`) s'importent tous sans erreur.
  [`test_chess_target_encoding_lazy_import.py:1`](../../tests/test_chess_target_encoding_lazy_import.py#L1)
