---
title: Distillation depuis la recherche classique (policy-only) — domaine échecs
created: 2026-08-04
updated: 2026-08-04
status: final
---

# PRD: Distillation depuis la recherche classique (policy-only) — domaine échecs
*Working title — confirm.*

## 0. Document Purpose

Ce PRD est destiné à Aymeric (mainteneur unique de `jax_supervised_training`) et sert de source de Functional Requirements pour l'epic à créer via `bmad-create-epics-and-stories`. Il formalise le "comment entraîner ici" d'une capacité dont le "quoi" est déjà tranché côté `chess_ai` (voir `chess_ai/_bmad-output/planning-artifacts/brief-model-without-search-2026-08-03.md`) et dont le contrat de données est déjà figé et versionné dans `docs/contract-chess-ai-training-interface.md` §2.6. Ce PRD ne rouvre ni l'un ni l'autre — il les prend comme entrées externes non renégociables et se limite au périmètre `jax_supervised_training` : nouvelle entrée `dataset_configs.py`, ajustement du chargeur de données, traçabilité du checkpoint exporté. FRs numérotées globalement ; termes du Glossaire utilisés verbatim dans tout le document.

## 1. Vision

`chess_ai` a validé (Story 3.1, 2026-08-04) un nouveau dataset "distillation depuis la recherche classique" : des positions d'auto-jeu étiquetées non plus par des parties PGN humaines (dataset `CHESS_NO_HISTORY` existant) mais par `chess_search.py`, un moteur alpha-bêta + évaluateur matériel déjà validé en conditions réelles. Principe AlphaZero — la recherche sert de professeur à l'entraînement, jamais à l'inférence : le réseau entraîné joue seul, sans plus jamais calculer, répondant à l'objectif produit d'origine de `chess_ai` ("un modèle qui joue seul, sans calcul de force brute à l'inférence").

Ce PRD couvre uniquement le branchement de ce dataset dans le pipeline d'entraînement générique (`Trainer`/`TaskStrategy`) de `jax_supervised_training`, en réutilisant au maximum ce qui existe déjà — décision actée avec Aymeric le 2026-08-04 (Option A ci-dessous) : le modèle `chess_cnn_attention_policy_value` et sa `TaskStrategy` associée sont réutilisés strictement tels quels, malgré l'absence de label value dans ce nouveau dataset (positions d'auto-jeu sans issue de partie complétée). La tête value existante est neutralisée (poids de loss nul, cible factice), pas supprimée ni remplacée — le coût d'implémentation le plus bas des deux options envisagées, au prix d'un checkpoint dont la sortie value est structurellement présente mais non entraînée, ce que ce PRD rend traçable plutôt que silencieux.

C'est un premier test, pas un pari sur le résultat final : Aymeric l'a explicitement cadré ainsi ("je suis incapable de te dire si le résultat sera bon, mais ça me semble être un bon 1er test"). Le succès de cette epic se mesure à "le pipeline fonctionne de bout en bout, sans régression", pas à la force de jeu obtenue.

## 2. Target User

### 2.1 Jobs To Be Done

- En tant que mainteneur unique de `jax_supervised_training`, je veux pouvoir entraîner n'importe quel nouveau dataset échecs produit côté `chess_ai` en ajoutant une configuration, pas du code de plomberie neuf — le pattern `TaskStrategy` existe précisément pour ça (déjà prouvé générique par l'Epic 9).
- En tant que consommateur futur du checkpoint exporté (moi-même plus tard, ou du code côté `chess_ai` comme `chess_model_inference.py`), je veux pouvoir distinguer sans ambiguïté une sortie value entraînée d'une sortie value factice, sans avoir à me souvenir "de mémoire" quel run a utilisé quelle config.

*Pas de User Journeys distinctes ici — outil interne à un seul opérateur, PRD purement technique (voir gabarit de section, cluster "Small-scope").*

## 3. Glossaire

- **Professeur (teacher)** — `chess_search.py`, moteur de recherche alpha-bêta + évaluateur matériel. Choisit le coup-label pour chaque position du nouveau dataset. Jamais appelé à l'inférence.
- **`chess_search_teacher`** — nom du dataset côté `chess_ai` (générateur `chess_search_teacher_dataset_tools.py`, Story 3.1) et préfixe de ses fichiers `.npz`.
- **`CHESS_SEARCH_TEACHER`** — nom retenu pour la nouvelle entrée de `DATASET_CONFIGS` (`dataset_configs.py`, ce repo) qui consomme ce dataset.
- **`POLICY_KEY`/`VALUE_KEY`** — clés `.npz` existantes (`chess_target_encoding.py`, côté `chess_ai`). Ce nouveau dataset n'écrit que `POLICY_KEY`.
- **Value factice (dummy value)** — valeur constante (0.0) substituée à la clé `value` absente des chunks `chess_search_teacher`, pour que le chargeur de données existant (`ChessPolicyValueDataset`) continue de produire un batch de forme inchangée.
- **`value_weight=0.0`** — poids nul du terme value dans `compute_chess_policy_value_loss` (déjà paramétrable, `loss_params`) : le gradient de la tête value n'influence plus l'entraînement, seule la policy apprend.
- **`value_head_trained`** — champ booléen à ajouter à la config sauvegardée avec le checkpoint exporté, `False` pour tout run utilisant `value_weight=0.0`. Permet à un futur consommateur de détecter une sortie value non fiable sans supposition externe (principe déjà acté au contrat §2.3).
- **Checkpoint "export pur"** — format de sauvegarde existant (`params`/`batch_stats`/`config`), déjà consommé par `chess_ai::chess_model_inference.py::load_chess_model`.

## 4. Features

### 4.1 Nouvelle configuration d'entraînement `CHESS_SEARCH_TEACHER`

**Description :** Une nouvelle entrée dans `DATASET_CONFIGS` (`dataset_configs.py`) pointe vers les chunks `chess_search_teacher` produits par `chess_ai` et réutilise, sans les modifier, le `task_type="chess_policy_value"` et le `model_name="chess_cnn_attention_policy_value"` déjà existants (Epic 9). Seule la tête value est neutralisée par les poids de loss, pas par une nouvelle architecture ou une nouvelle stratégie.

**Functional Requirements :**

#### FR-1 : Entrée `dataset_configs.py` dédiée

Le mainteneur peut lancer un entraînement sur le dataset `chess_search_teacher` via une entrée `CHESS_SEARCH_TEACHER` autonome dans `DATASET_CONFIGS`, sans toucher aux entrées existantes.

**Conséquences (testables) :**
- `num_classes=4672`, `num_channels=29`, `input_shape=(8,8,29)` — même schéma de plans que l'existant (§2.1/§2.6 du contrat, `include_history=True`), pas un nouveau format. Attention cependant : aucune config active ne combine déjà `chess_cnn_attention_policy_value`/`chess_policy_value` avec 29 canaux — `CHESS_NO_HISTORY` (seule config `chess_policy_value` active aujourd'hui) est à 19 canaux ; `CHESS_LEGAL_MOVES` est bien à 29 canaux mais avec un modèle et un `task_type` différents (multi-label). Ne pas copier `CHESS_NO_HISTORY` comme gabarit pour `num_channels`/`input_shape` — ces deux valeurs viennent du contrat §2.6, pas d'une config sœur existante.
- `output_prefix` pointe vers un répertoire dédié (`chunks/chess_search_teacher/`), distinct de `chess_no_history/` et `chess_legal_moves/` — aucune collision de fichiers.
- `task_type="chess_policy_value"` et `model_name="chess_cnn_attention_policy_value"` sont réutilisés à l'identique — aucune nouvelle `TaskStrategy`, aucun nouveau modèle enregistré dans `model_library.py`.
- `validate_config()` accepte cette entrée sans modification de sa logique (mêmes clés requises que les configs échecs existantes).

#### FR-2 : Neutralisation de la tête value par pondération

Le terme de loss value n'influence pas l'entraînement de `CHESS_SEARCH_TEACHER`, sans modifier `compute_chess_policy_value_loss` ni `ChessPolicyValueStrategy`.

**Conséquences (testables) :**
- `loss_params = {"policy_weight": 1.0, "value_weight": 0.0}` dans la config `CHESS_SEARCH_TEACHER`.
- Le détail de loss affiché par `generate_reports()` (déjà existant, AD-24) montre `value_loss` non nul en valeur brute (la tête produit toujours une sortie) mais son poids affiché est `0.0` — comportement attendu, pas une anomalie à corriger.

**Out of Scope :** Toute pondération intermédiaire (ni 0 ni 1) — voir Non-Goals et Open Question 2.

### 4.2 Tolérance du chargeur de données à un dataset sans value réelle

**Description :** `ChessPolicyValueDataset.create_tf_dataset` (`data_management.py`) lit aujourd'hui `data["value"]` sans garde — un `KeyError` immédiat sur les chunks `chess_search_teacher`, qui n'écrivent que `position`/`policy` (contrat §2.6). Cette feature rend la lecture tolérante à l'absence de la clé, sans changer le comportement des datasets qui la fournissent déjà.

**Functional Requirements :**

#### FR-3 : Value par défaut quand la clé est absente du `.npz`

Le chargeur de données échecs charge un chunk sans clé `value` en substituant une valeur constante `0.0` à chaque exemple, au lieu de lever une exception.

**Conséquences (testables) :**
- Un chunk `chess_search_teacher` (clés `position`+`policy` uniquement) se charge sans erreur via `ChessPolicyValueDataset`.
- Un chunk `CHESS_NO_HISTORY` existant (clés `position`+`policy`+`value`) continue de charger ses vraies valeurs `value` sans changement de comportement — non-régression vérifiée par exécution réelle, pas seulement par lecture de code.
- La forme et le `dtype` du tenseur `value` produit par `create_tf_dataset` restent inchangés (`scalaire float32`) dans les deux cas — le reste du pipeline (`ChessPolicyValueStrategy.preprocess_batch`, `compute_loss`) ne voit aucune différence de forme.

### 4.3 Traçabilité du statut de la tête value dans le checkpoint exporté

**Description :** Puisque le checkpoint `CHESS_SEARCH_TEACHER` contient structurellement une sortie value jamais entraînée (Option A), sa config sauvegardée porte un signal explicite et lisible par programme — cohérent avec le principe déjà acté au contrat §2.3 ("la config sauvegardée doit porter tout ce qui est nécessaire pour que `chess_ai` charge et utilise le checkpoint sans supposition externe").

**Functional Requirements :**

#### FR-4 : Champ `value_head_trained` dans la config exportée

Un futur consommateur du checkpoint (`chess_ai` ou tout autre code) peut détecter par programme, sans connaître l'historique du run, que la sortie value n'est pas fiable.

**Conséquences (testables) :**
- Tout run dont `loss_params.value_weight == 0` sauvegarde `value_head_trained=False` dans la config embarquée avec le checkpoint (`params`/`batch_stats`/`config`).
- Un run échecs existant (`CHESS_NO_HISTORY`, value entraînée) sauvegarde `value_head_trained=True` — champ ajouté rétroactivement à la logique d'export, pas seulement à la nouvelle config.
- Aucune modification du format de checkpoint lui-même (toujours `params`/`batch_stats`/`config`) — un ajout de clé dans `config`, pas un nouveau format.

**Out of Scope :** La lecture effective de ce champ par `chess_model_inference.py` (`chess_ai`) — hors scope de cette epic, laissée à un futur cycle côté `chess_ai` (voir Open Questions).

### 4.4 Non-régression et synchronisation du contrat d'interface

**Description :** Hérite directement de la contrainte dure de l'Epic 9 (AD-21) — cette epic ajoute une capacité, elle n'en modifie aucune existante.

**Functional Requirements :**

#### FR-5 : Aucune régression sur les domaines existants

`JAX_DETECTOR`, `CHESS_NO_HISTORY`, `CHESS_LEGAL_MOVES` et leurs `TaskStrategy`/loaders associés se comportent à l'identique avant/après cette epic.

**Conséquences (testables) :**
- Aucune ligne de `ClassificationStrategy`/`DetectionStrategy`/`CenterNetDetectionStrategy`/`KeplerStrategy`/`ChessLegalMovesStrategy` n'est modifiée.
- `ChessPolicyValueStrategy` et `ChessPolicyValueDataset` ne changent que pour FR-3/FR-4 (tolérance value absente, champ config) — aucun autre changement de comportement pour les configs qui les utilisaient déjà.
- Validation par exécution réelle (pas seulement par lecture de code) d'au moins un des domaines existants après implémentation.

#### FR-6 : Entraînement de bout en bout validé par exécution réelle

`CHESS_SEARCH_TEACHER` s'entraîne de bout en bout via `Trainer` sans erreur et produit une `PolicyAccuracy` mesurable en validation, une fois les chunks `chess_search_teacher` disponibles côté `chess_ai`.

**Conséquences (testables) :**
- Un run réel (même court, pas nécessairement le run de production final) complète au moins une epoch sans exception.
- La métrique primaire (`PolicyAccuracy`, réutilisée de `ChessPolicyValueStrategy`) est loguée et non constante à travers les steps (signe que le gradient policy circule bien malgré `value_weight=0`).

#### FR-7 : Synchronisation du contrat d'interface

`docs/contract-chess-ai-training-interface.md` documente cette nouvelle capacité côté `jax_supervised_training` (nom de config, réutilisation `chess_policy_value`, statut `value_head_trained`) et la copie est reportée dans `chess_ai/docs/` avant clôture — process déjà établi (§4 du contrat), pas une nouveauté de cette epic.

## 5. Non-Goals (Explicit)

- **Pas de nouveau modèle ni de nouvelle `TaskStrategy`** — Option A actée le 2026-08-04 avec Aymeric ; c'était l'alternative (Option B) envisagée puis écartée pour ce premier test.
- **Pas d'adaptation de `chess_ai` en retour** dans cette epic (ni `chess_model_inference.py`, ni lecture du flag `value_head_trained`) — le format de checkpoint reste consommable tel quel par le code `chess_ai` existant, c'est précisément ce que garantit Option A.
- **Pas d'évaluation de la force de jeu** (taux de coups "corrects" au sens échecs, Elo estimé, tournoi) — le professeur lui-même plafonne la qualité atteignable (matériel + bonus centre uniquement, nommé explicitement dans le brief chess_ai), l'évaluer appartient à un cycle séparé côté `chess_ai`.
- **Pas de génération de dataset** — `chess_ai` est l'unique producteur des `.npz` (contrat §1), déjà fait (Story 3.1 chess_ai, 2026-08-04).
- **Pas de retour sur la pipeline composée légalité+stratégie** (Modèle 1 + Modèle 2) — reste suspendue côté contrat §3, hors scope ici.
- **Pas de signal professeur enrichi** (distribution policy via softmax des scores negamax, ou score racine du professeur comme cible value — deux pistes nommées à parité dans le brief chess_ai) — le premier test reste au plus simple (label policy unique, value factice), conformément à la décision d'Aymeric actée dans ce brief.

## 6. MVP Scope

### 6.1 In Scope

- Entrée `CHESS_SEARCH_TEACHER` dans `dataset_configs.py` (FR-1, FR-2).
- Tolérance de `ChessPolicyValueDataset` à un `.npz` sans clé `value` (FR-3).
- Champ `value_head_trained` dans la config exportée avec tout checkpoint échecs (FR-4).
- Validation de non-régression + validation par exécution réelle (FR-5, FR-6).
- Synchronisation du contrat d'interface (FR-7).

### 6.2 Out of Scope for MVP

- Lecture du flag `value_head_trained` côté `chess_ai` — différé à un futur cycle `chess_ai`, seulement si un consommateur a effectivement besoin de la sortie value un jour. `[NOTE FOR PM]` : à ne pas oublier si `chess_ai` ajoute un jour un affichage de barre d'évaluation basé sur cette sortie.
- Value réelle basée sur le score du professeur — reporté, pas dans le périmètre du premier test (voir Non-Goals).
- Tuning des hyperparamètres GPU/TPU (`micro_batch_size`, `learning_rate`, `decay_steps`, etc.) pour `CHESS_SEARCH_TEACHER` — détail d'implémentation, laissé à la story de dev (probablement copié de `CHESS_NO_HISTORY` comme point de départ, non tuné empiriquement pour ce nouveau dataset).

## 7. Success Metrics

*Enjeu interne/epic technique — critères qualitatifs et binaires suffisent, pas de tableau de bord quantitatif (même posture que le PRD Epic 9).*

**Primary**
- **SM-1** : `CHESS_SEARCH_TEACHER` s'entraîne de bout en bout via `Trainer`/`ChessPolicyValueStrategy` sans erreur, avec une `PolicyAccuracy` en validation qui progresse de façon mesurable (pas un plateau immédiat proche du hasard, ~1/4672). Valide FR-1, FR-2, FR-3, FR-6.
- **SM-2** : Aucune régression mesurée sur `JAX_DETECTOR`/`CHESS_NO_HISTORY`/`CHESS_LEGAL_MOVES`, vérifiée par exécution réelle. Valide FR-5.

**Secondary**
- **SM-3** : Le checkpoint exporté porte `value_head_trained=False` de façon vérifiable (relecture directe du pickle). Valide FR-4.
- **SM-4** : Le contrat d'interface est synchronisé entre les deux repos avant clôture de l'epic. Valide FR-7.

**Counter-metrics (à ne pas optimiser)**
- **SM-C1** : La qualité de jeu du modèle résultant (Elo, taux de victoire, etc.) n'est **pas** un critère chiffré de clôture de cette epic — cohérent avec le plafond de qualité du professeur, nommé explicitement dans le brief `chess_ai`. Contrebalance SM-1 : ne pas prolonger l'entraînement ni complexifier l'architecture pour chasser une performance de jeu qui appartient à un cycle d'évaluation séparé. **Critère de "déception" assumé (décision Aymeric, party mode 2026-08-04)** : jugé qualitativement en jouant contre le modèle et en comparant au comportement de jeu actuel (`chess_search.py`/checkpoints existants), pas par un seuil chiffré — choix explicite de v1, pas un angle mort. Pas de "rollback" à proprement parler si le résultat déçoit : `CHESS_SEARCH_TEACHER` est une entrée `dataset_configs.py` additive et isolée (FR5) — il suffit de cesser de l'utiliser/de la citer, aucune ligne de code à défaire, à condition que FR5 (non-régression) tienne réellement.
- **SM-C2** : Ne pas transformer ce premier test en occasion d'ajouter une vraie cible value ou une nouvelle architecture "pendant qu'on y est" — Option A a été choisie précisément pour rester la moins chère des deux ; toute extension d'architecture appartient à un cycle ultérieur si ce test est jugé prometteur.

## 8. Open Questions

1. **Qui relit `value_head_trained` côté `chess_ai`, et quand ?** Pas cette epic — mais le champ existe pour qu'un futur cycle `chess_ai` n'ait pas à redécouvrir le problème. `[NOTE FOR PM]`.
2. **Faut-il un jour enrichir le signal du professeur ?** Le brief `chess_ai` nomme deux pistes "plus riches" à parité, toutes deux reportées pour ce premier test : (a) une distribution policy via softmax des scores negamax du professeur plutôt qu'un label unique, (b) le score racine du professeur comme cible value réelle (avantage : ne dépend pas d'une partie complétée, contrairement au label ±1/0 actuel). Le premier test reste au plus simple (label policy unique, value factice) — décision d'Aymeric actée dans le brief.
3. **Nom exact et emplacement du champ `value_head_trained`** dans la structure `config` du checkpoint (clé top-level vs sous `loss_params`) — détail à trancher en story de dev, non bloquant pour ce PRD.
4. **Chunk_size et hyperparamètres GPU/TPU précis** pour `CHESS_SEARCH_TEACHER` — laissés à l'implémentation, probablement calqués sur `CHESS_NO_HISTORY` en première approche puis ajustés une fois le volume réel de chunks connu (même pattern que `CHESS_LEGAL_MOVES`, dont les `decay_steps` ont été recalculés après un premier run réel).

## 9. Assumptions Index

- [CONFIRMÉ §0] Enjeu (stakes) traité comme interne/solo, même gabarit de rigueur que le PRD Epic 9 (2026-07-27) — confirmé par Aymeric en revue.
- [CONFIRMÉ §4.1] `output_prefix` = `{DATA_ROOT}/chunks/chess_search_teacher/chess_search_teacher` — confirmé par le smoke-test `__main__` de `chess_search_teacher_dataset_tools.py` (chess_ai, ligne 171), symétrique avec les entrées échecs existantes. Réserve mineure : c'est le chemin du smoke-test manuel, pas nécessairement celui que `chess_ai` utilisera pour la génération de production (paramètre Python, pas exposé en CLI) — à revérifier au moment de pointer `output_prefix` vers les vrais chunks.
- [ASSUMPTION §4.3] Nom de champ `value_head_trained` proposé par Winston pendant la session, pas un nom déjà utilisé ailleurs dans le codebase — libre à ajuster en story de dev (Open Question 3).
