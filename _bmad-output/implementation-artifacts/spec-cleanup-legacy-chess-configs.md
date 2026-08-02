---
title: 'Nettoyage des configs échecs obsolètes (CHESS, CHESS_NAKAMURA_NO_HISTORY)'
type: 'chore'
created: '2026-08-02'
status: 'done'
route: 'one-shot'
---

## Intent

**Problem:** `dataset_configs.py` portait encore deux entrées échecs devenues sans intérêt : `CHESS` (variante "avec historique", remplacée par `CHESS_NO_HISTORY` comme défaut) et `CHESS_NAKAMURA_NO_HISTORY` (config du tournoi Carlsen vs Nakamura, conclu sans influence mesurable détectée sur la qualité de jeu — voir `deferred-work.md`).

**Approach:** Suppression pure des deux entrées du dict `DATASET_CONFIGS`, ne gardant que `CHESS_NO_HISTORY` comme unique config policy+value du domaine échecs. Correction en cascade des commentaires/docs qui référençaient `CHESS` comme encore existante (`dataset_configs.py`, `data_management.py`, `main.py`, `docs/ChessCnnAttentionPolicyValue.md`, `docs/contract-chess-ai-training-interface.md`).

## Suggested Review Order

**Suppression des configs**

- Les deux entrées mortes disparaissent, `CHESS_NO_HISTORY` devient la seule config échecs du dict — vérifier qu'aucune virgule/accolade orpheline ne reste.
  [`dataset_configs.py:585`](../../dataset_configs.py#L585)

**Commentaires devenus obsolètes (mêmes noms de config cités ailleurs)**

- Commentaire de `CHESS_NO_HISTORY` récrit pour ne plus décrire `CHESS` comme un sibling vivant — devient l'historique de pourquoi cette config est maintenant seule.
  [`dataset_configs.py:585`](../../dataset_configs.py#L585)
- `print_config` référençait `CHESS` par son nom pour justifier l'absence de `class_names`/`image_size` — corrigé vers `CHESS_NO_HISTORY`, seule config à laquelle ça s'applique encore.
  [`dataset_configs.py:679`](../../dataset_configs.py#L679)
- `ChessPolicyValueDataset.__init__` : commentaire sur le défaut `num_planes=29` mis à jour (`CHESS` retirée, défaut confirmé mort en pratique).
  [`data_management.py:627`](../../data_management.py#L627)
- Même correction côté `get_datasets()`, point d'appel du défaut.
  [`data_management.py:773`](../../data_management.py#L773)
- `main.py` : commentaire sur `class_names` généralisé à "les configs échecs" plutôt que de nommer `CHESS` spécifiquement.
  [`main.py:67`](../../main.py#L67)

**Docs**

- Tableau comparatif des deux variantes (avec/sans historique) : ligne `CHESS` marquée retirée plutôt que supprimée silencieusement, pour garder la trace du résultat mesuré (24.43%).
  [`docs/ChessCnnAttentionPolicyValue.md:26`](../../docs/ChessCnnAttentionPolicyValue.md#L26)
- Note de bas de section corrigée : n'affirme plus que "les deux variantes restent disponibles".
  [`docs/ChessCnnAttentionPolicyValue.md:29`](../../docs/ChessCnnAttentionPolicyValue.md#L29)
- Convention de nommage `dataset_configs.py` : ne cite plus les deux clés supprimées.
  [`docs/contract-chess-ai-training-interface.md:94`](../../docs/contract-chess-ai-training-interface.md#L94)

**Suivi (hors périmètre de ce nettoyage)**

- Deux entrées ajoutées à `deferred-work.md` : le futur chantier `CHESS_LEGAL_MOVES` (scindé de cet intent) et les artefacts orphelins (`best_model_chess.pkl` etc., repérés par la revue adversariale, non traités ici).
  [`deferred-work.md:544`](deferred-work.md#L544)
