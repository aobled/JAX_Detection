# Contrat d'interface : `jax_supervised_training` ↔ `chess_ai`

**Statut : ébauche (2026-07-31), à faire évoluer des deux côtés.** Ce document n'est pas un
PRD ni une epic — c'est la surface d'interface entre deux repos indépendants qui doit
rester identique des deux côtés, contrairement au reste du code échecs qui, lui, est
dupliqué et volontairement laissé libre de diverger (voir `chess_ai/HANDOFF.md`).

**Copie canonique : TBD.** Pour l'instant ce fichier vit uniquement dans
`jax_supervised_training/docs/`. À copier (ou lier) dans `chess_ai` une fois son
instance BMAD installée, pour que les deux PRD/epics puissent le référencer sans
dupliquer son contenu.

## 1. Répartition (rappel de `chess_ai/HANDOFF.md`)

- **`jax_supervised_training`** : génère les `.npz`, définit les modèles, les entraîne
  (`Trainer`/`TaskStrategy`/`main.py`). Ne compose jamais, ne joue jamais.
- **`chess_ai`** : compose des checkpoints déjà entraînés (jeu, tournoi, évaluation).
  Ne réentraîne jamais.

## 2. Ce qui est déjà stable (ne pas changer d'un côté sans changer l'autre)

### 2.1 Encodage de la position

`chess_target_encoding.py::encode_position` — plans 8×8×19 (position seule) ou 8×8×29
(avec historique, `include_history=True`). Fichier dupliqué volontairement entre les deux
repos (cycles de vie indépendants, cf. HANDOFF.md) — mais le **format produit**, lui,
n'est pas négociable indépendamment : toute évolution du schéma de plans doit être portée
des deux côtés à la fois, sous peine de désynchronisation silencieuse (c'est exactement
le risque qu'AD-18 de ce repo décrit pour d'autres formats d'échange).

### 2.2 Espace de sortie policy

`NUM_MOVES = 4672` (64 cases source × 73 types de coup, schéma AlphaZero),
`move_to_index`/`index_to_move` comme seule conversion (jamais réimplémentée
indépendamment). Vaut pour le modèle combiné actuel ; à confirmer que les futurs modèles
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

## 3. Ce qui reste ouvert (à trancher côté PRD `chess_ai`, puis à reporter ici)

- **Modèle 1 "candidats de coups"** (équivalent "detection") : `model_name`/`task_type`
  exacts, forme de sortie (distribution sur un sous-ensemble de coups ? score par coup
  légal ? top-K ?) — non tranché.
- **Modèle 2 "stratégie/évaluateur"** (équivalent "classification") : `model_name`/
  `task_type` exacts, forme de sortie (évaluation d'une position candidate ? classement
  de plusieurs candidats ?) — non tranché.
- **Labels/dataset requis pour chacun** : le dataset actuel (`chess_pgn_dataset_tools.py`)
  produit policy+value pour le modèle combiné existant ; les 2 nouveaux modèles
  pourraient nécessiter un format de label différent (ex. paires de positions à comparer
  pour le modèle 2) — à spécifier avant toute nouvelle entrée `dataset_configs.py`.
- **Nommage `dataset_configs.py`** : whatever convention suit `CHESS`/`CHESS_NO_HISTORY`/
  `CHESS_NAKAMURA_NO_HISTORY` existants, à fixer une fois les 2 modèles nommés.

## 4. Process

Toute epic `chess_ai` qui nécessite une nouvelle capacité d'entraînement se traduit par
une epic **dans `jax_supervised_training`** (nouveau modèle/config/TaskStrategy), sur le
même mode que l'Epic 9 — pas par du travail direct dans ce repo depuis une session
`chess_ai`. Ce contrat est l'unique document que les deux PRD ont besoin de référencer en
commun ; il n'y a pas de PRD ou d'epic partagée entre les deux repos.

À chaque décision prise côté `chess_ai` sur §3, reporter le résultat ici (ou dans la
copie côté `chess_ai` si elle existe déjà) avant de démarrer l'epic côté
`jax_supervised_training`.
