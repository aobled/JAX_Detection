# Schéma dataset spike "coup-token" — `chess_ai` → `jax_supervised_training`

**Statut : SPIKE / PROVISOIRE — n'est PAS une section du contrat stable
(`contract-chess-ai-training-interface.md` §2).** Document séparé, délibérément : le
spec (`chess_ai/_bmad-output/specs/spec-chess-move-token-poc/SPEC.md`) exclut
explicitement la synchro du contrat stable tant que le spike n'est pas concluant
(Constraints/Non-goals). Ce fichier existe uniquement pour permettre une session
`jax_supervised_training` séparée de consommer le dataset spike (CAP-3) sans avoir à lire
tout le spec. **Peut changer de format sans préavis** si le spike évolue — ne pas fixer
une entrée `dataset_configs.py` "définitive" dessus avant le verdict CAP-4/CAP-5 (voir
Success signal du spec).

Contexte complet, raisonnement, décisions : `SPEC.md` + `.memlog.md` du dossier ci-dessus
(`chess_ai`). Primitives d'encodage : `chess_ai/chess_move_token_encoding.py`
(`encode_move_token_sequence`/`decode_move_token_sequence`). Script de génération :
`chess_ai/chess_move_token_spike_dataset.py`.

## Fichier

Un seul fichier (pas de chunks, pas de préfixe `_chunkN`, contrairement aux 3 datasets du
contrat stable) :

```
/home/aobled/Documents/data/chunks/chess_move_token_spike/chess_move_token_spike.npz
```

Régénéré le 2026-08-10 (2e run, remplace le 1er run à 3190 positions — surdimensionné vs
~2M paramètres du modèle testé, overfitting sévère observé) : `n_games=3000`, `seed=42`,
`teacher_depth=12`, `depth=8`, `opening_plies=4`, `max_halfmoves=400` (mêmes paramètres
par défaut que le dataset baseline `CHESS_SEARCH_TEACHER`, professeur Stockfish),
**généré en parallèle** (`build_move_token_spike_dataset_parallel`, 7 workers sur cette
machine). **416 306 positions**, 1275 positions terminales ignorées (aucun coup légal).
Reproductible **à `n_workers` fixe** : relancer
`chess_move_token_spike_dataset.py --n-games 3000 --seed 42` (sans `--n-workers`, donc
même défaut `(cpu_count-2)//2`) régénère un lot identique — chaque worker a un flux RNG
indépendant et déterministe (`np.random.SeedSequence(seed).spawn(n_workers)`), donc
changer `n_workers` change aussi le découpage des parties et le lot résultant (pas un
bug, une conséquence du parallélisme). Contrairement à la génération non seedée du
dataset baseline réel — voir `SPEC.md` Assumptions.

## Clés `.npz`

| Clé | dtype | shape | Sémantique |
| --- | --- | --- | --- |
| `position` | `float32` | `(416306, 8, 8, 29)` | Identique au dataset baseline `CHESS_SEARCH_TEACHER` (`encode_position`, `include_history=True`, `include_legal_hint=True` — chemin par défaut inchangé, §2.1/§2.6 du contrat stable). Permet de faire tourner le modèle baseline existant (`ChessCnnAttentionPolicyValue`) sur les mêmes positions, pour la comparaison CAP-4. |
| `policy` | `int32` | `(416306,)` | Même sémantique que `CHESS_SEARCH_TEACHER.policy` (§2.6 du contrat stable) : index `move_to_index` (espace `NUM_MOVES=4672`) du coup choisi par le professeur Stockfish à `teacher_depth=12` — **même professeur, même profondeur** que le dataset ayant servi au checkpoint baseline (`best_model_chess_search_teacher.pkl`, 28.00% val PolicyAccuracy). |
| `move_tokens` | `int32` | `(35035101,)` | Concaténation des séquences coup-token de **toutes** les positions, bout à bout. Vocabulaire = **le même espace d'action que `policy`** (`NUM_MOVES=4672`, `move_to_index`/`index_to_move`, `chess_target_encoding.py`) — pas un vocabulaire séparé à apprendre. |
| `move_token_offsets` | `int64` | `(416307,)` | Bornes CSR : la séquence de la position `i` est `move_tokens[move_token_offsets[i]:move_token_offsets[i+1]]`. Longueur de séquence = `offsets[i+1] - offsets[i]`, **variable** (pas de padding, pas de troncature — CAP-1, décision délibérée du 2026-08-10). Sur ce run : min=4, médiane=74, max=403 coups. |

Aucune clé `value` (mêmes raisons que `CHESS_SEARCH_TEACHER` — positions d'auto-jeu, pas
d'issue de partie complétée naturellement associée).

## Ce qui n'est PAS défini par ce schéma (à trancher côté `jax_supervised_training`)

- **Stratégie de padding/batching** pour des séquences de longueur variable (masque
  d'attention causal, `PAD` token ou bucketing par longueur, etc.) — hors scope du côté
  `chess_ai`, qui ne fournit que la donnée brute non tronquée.
- **Tokens spéciaux** (`BOS`/`EOS`/`PAD`) — aucun n'existe dans `move_tokens` tel quel ;
  à ajouter côté modèle si l'architecture en a besoin, pas dans ce fichier.
- Le modèle lui-même (CAP-3 du spec) : transformer causal sur `move_tokens`, même tête
  policy 4672 classes et même loss (cross-entropy) que `ChessCnnAttentionPolicyValue`
  aujourd'hui — voir `SPEC.md` CAP-3 pour l'intent/success exacts.

## Vérification disponible côté `chess_ai`

`chess_move_token_encoding.decode_move_token_sequence(tokens, chess.Board())` redécode
une séquence de tokens en `chess.Move` réels (utile pour un sanity check ponctuel côté
`jax_supervised_training`, ex. vérifier qu'un batch chargé correspond bien à des coups
légaux). Vérifié par round-trip sur 200 positions aléatoires + les deux extrémités du
fichier fusionné (couvrant le 1er et le dernier worker) — 0 échec. Voir `.memlog.md` du
spec pour le détail, y compris la vérification spécifique du recalage des offsets entre
workers (`_consolidate_move_token_workers`).
