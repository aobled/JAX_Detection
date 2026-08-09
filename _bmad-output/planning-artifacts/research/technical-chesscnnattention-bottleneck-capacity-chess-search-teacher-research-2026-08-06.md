---
stepsCompleted: [1, 2, 3, 4]
inputDocuments: []
workflowType: 'research'
lastStep: 1
research_type: 'technical'
research_topic: 'Capacité architecturale de ChessCnnAttention (bottleneck tokens, token_dim, profondeur) pour CHESS_SEARCH_TEACHER'
research_goals: 'Comprendre les mécanismes en jeu (pourquoi le modèle plafonne à ~29% Val PolicyAccuracy sans overfitting) pour permettre à Aymeric de décider lui-même des leviers à essayer, sans trancher à sa place.'
user_name: 'Aymeric'
date: '2026-08-06'
web_research_enabled: true
source_verification: true
---

# Research Report: technical

**Date:** 2026-08-06
**Author:** Aymeric
**Research Type:** technical

---

## Research Overview

Diagnostic technique portant sur `ChessCnnAttentionLegalMoves` / `ChessCnnAttentionPolicyValue` (model_library.py), architecture CNN 8×8 + bottleneck cross-attention Perceiver-style, entraînée sur le dataset `CHESS_SEARCH_TEACHER` (dataset_configs.py). Point de départ empirique : run du 2026-08-05 (`chess_search_teacher.png`) — Train PolicyAccuracy 31.18%, Val PolicyAccuracy 29.26%, gap 1.92%, overfitting Simpson 0.71% (quasi nul). Le faible gap train/val suggère une sous-capacité du modèle (ou une difficulté intrinsèque de la tâche) plutôt qu'un surapprentissage, ce qui oriente la recherche vers les leviers de capacité architecturale (`num_bottleneck_tokens` actuellement 8, `token_dim` actuellement 64, `num_heads` actuellement 4) plutôt que vers de la régularisation.

---

## Technical Research Scope Confirmation

**Research Topic:** Capacité architecturale de ChessCnnAttention (bottleneck tokens, token_dim, profondeur) pour CHESS_SEARCH_TEACHER
**Research Goals:** Comprendre les mécanismes en jeu (pourquoi le modèle plafonne à ~29% Val PolicyAccuracy sans overfitting) pour permettre à Aymeric de décider lui-même des leviers à essayer, sans trancher à sa place.

**Technical Research Scope:**

- Architecture Analysis - patterns Perceiver-style / bottleneck cross-attention, interaction num_bottleneck_tokens / token_dim / num_heads
- Implementation Approaches - pratiques établies pour dimensionner ce type de bottleneck attentionnel sur tâches denses
- Technology Stack - Flax/JAX MultiHeadDotProductAttention, littérature sur les ratios tokens/dimension/têtes
- Integration Patterns - interaction avec le reste du pipeline (backbone CNN, tête policy 4672 classes, LR schedule)
- Performance Considerations - coût compute/mémoire de l'augmentation du nombre de tokens vs gain attendu, lecture des signaux diagnostiques (gap train/val) pour distinguer sous-capacité d'autres causes

**Research Methodology:**

- Current web data with rigorous source verification
- Multi-source validation for critical technical claims
- Confidence level framework for uncertain information
- Comprehensive technical coverage with architecture-specific insights

**Scope Confirmed:** 2026-08-06

---

## Technology Stack Analysis

_Note méthodologique : le template générique de cette étape (langages, bases de données, cloud) est adapté au périmètre réel — une architecture Perceiver-style bottleneck en Flax/JAX pour une tâche de policy échecs. Les sous-sections sans pertinence (bases de données, infra cloud) sont explicitement marquées N/A plutôt que remplies artificiellement._

### Framework : Flax `MultiHeadDotProductAttention`

La contrainte de divisibilité `token_dim % num_heads == 0` (déjà assertée dans le code, `model_library.py:1003-1005`) est bien la seule contrainte dure du framework — `qkv_features` peut être fixé indépendamment de `token_dim` si besoin (ex. `nn.MultiHeadDotProductAttention(num_heads=8, qkv_features=16)`), mais le code actuel laisse Flax l'inférer automatiquement des dimensions d'entrée.
_Source: [flax.linen.MultiHeadDotProductAttention docs](https://flax.readthedocs.io/en/v0.6.10/api_reference/_autosummary/flax.linen.MultiHeadDotProductAttention.html), [flax/linen/attention.py](https://github.com/google/flax/blob/main/flax/linen/attention.py)_

### Pattern architectural : bottleneck Perceiver-style (cross-attention + learned queries)

C'est le point central. Le papier Perceiver IO original et ses dérivés (Flamingo Perceiver Resampler) éclairent directement le choix `num_bottleneck_tokens=8` :

- **Le nombre de latents est un vrai levier de capacité, pas un simple détail d'implémentation.** Perceiver IO utilise typiquement 256 à 512 latents — pas 8. Les GFLOPs croissent linéairement avec le nombre de latents (N), et la taille du tableau de latents permet de construire des Transformers plus profonds, mais **"la sévérité du bottleneck peut restreindre la capacité du réseau à capturer toute l'information."**
_Source: [Perceiver IO: a scalable, fully-attentional model](https://huggingface.co/blog/perceiver), [arXiv 2107.14795](https://ar5iv.labs.arxiv.org/html/2107.14795)_

- **Le bottleneck de type "resampler" (queries apprises figées en nombre, cross-attention sur les tokens spatiaux) est documenté comme perdant du détail spatial fin.** Une étude récente sur les VLM note explicitement : *"spatial information is largely absent from the resampler output"* et que les tâches nécessitant une compréhension spatiale fine sont mal servies par ce pattern — **augmenter l'encodeur en amont sans augmenter le résampler produit un gain limité en aval** ("up-scaling the upstream encoder produces limited downstream improvement, indicating sub-optimal use of rich input features").
_Source: [Vision Remember (arXiv 2506.03928)](https://arxiv.org/html/2506.03928), [Perceiver Resampler Architecture — Emergent Mind](https://www.emergentmind.com/topics/perceiver-resampler-architecture)_

  **Pertinence directe pour ce cas** : prédire un coup d'échecs (case source + case destination parmi 4672) est structurellement une tâche à détail spatial fin — c'est précisément le type de tâche où la littérature documente une perte d'information au niveau du resampler. Compresser 64 tokens-case vers seulement 8 requêtes apprises est un ratio de compression bien plus agressif que les usages documentés du pattern (qui gardent typiquement des centaines de latents, ou dans le cas des moteurs d'échecs spécialisés — voir ci-dessous — la totalité des 64 tokens sans compression).

### Point de comparaison domaine : réseaux transformers spécialisés échecs

Les moteurs d'échecs neuronaux à base de transformer (Leela Chess Zero BT4, DeepMind "Grandmaster-level chess without search") traitent chacune des 64 cases comme un token de séquence — **et gardent les 64 tokens tout du long, sans jamais les compresser vers un petit ensemble de requêtes apprises.** Leela BT4 : 15 couches transformer, hidden size **1024**, 32 têtes d'attention, séquence de 64 tokens.
_Source: [Transformer Progress — Leela Chess Zero blog](https://lczero.org/blog/2024/02/transformer-progress/)_

Comparaison directe avec l'architecture actuelle :

| | `ChessCnnAttentionPolicyValue` (actuel) | Leela BT4 |
|---|---|---|
| Tokens portant l'info spatiale | 8 (compressés depuis 64 par cross-attention) | 64 (jamais compressés) |
| Dimension de représentation (`token_dim` / hidden size) | 64 | 1024 |
| Têtes d'attention | 4 | 32 |
| Paramètres totaux | 382 017 (confirmé par `chess_search_teacher.png`) | plusieurs dizaines de millions |

L'écart n'est pas d'un ordre de grandeur mais de deux : le modèle actuel est ~16× plus petit en dimension de représentation, et compresse ses 64 tokens spatiaux à 8 là où le point de comparaison du domaine ne les compresse jamais. Ce n'est pas une critique du choix initial (382K paramètres est délibérément un budget "petit modèle embarqué", cf. AD-23 cité dans le code) — c'est un signal que **la capacité de représentation, pas uniquement `num_bottleneck_tokens`, est le facteur dimensionnant.**

### Ceiling intrinsèque de la tâche (à ne pas confondre avec sous-capacité)

Recherche additionnelle : la précision "top-1" en prédiction de coup n'a **pas de plafond publié fiable pour AlphaZero/Leela** dans les sources trouvées (le chiffre exact n'apparaît pas dans les papers indexés par cette recherche), mais deux repères qualitatifs ressortent :
- Sur des positions aléatoires sans contexte, le ceiling "hasard" est ≈6.43% (nombre moyen de coups légaux par position) — donc 29% est largement au-dessus du hasard, signe que le modèle apprend un signal réel.
- Le PRD du projet (`prd-jax_supervised_training-2026-08-04/prd.md`, ligne 168, décision Aymeric actée en party mode le 2026-08-04) note explicitement que **la qualité de jeu du modèle "belongs to a separate evaluation cycle" et est plafonnée par la qualité du professeur** (`chess_search.py`, alpha-bêta + évaluateur matériel simple) — un professeur alpha-bêta n'est pas Stockfish, donc le label "meilleur coup" appris peut lui-même être bruité indépendamment de la capacité du modèle élève.
_Source: recherche web (aucune source fiable trouvée pour un chiffre précis AlphaZero top-1), + `prd.md` local ligne 168_

### Bases de données / Infrastructure cloud

N/A pour ce périmètre — pas de composant base de données ou déploiement cloud pertinent à cette question d'architecture de modèle.

---

## Integration Patterns Analysis

_Note méthodologique : le template générique (API REST/gRPC, message queues, microservices) ne s'applique pas à une architecture de modèle. Réinterprété selon le périmètre confirmé à l'étape 1 : comment le bottleneck attentionnel s'intègre au reste du pipeline (backbone CNN, calendrier d'apprentissage, coût compute)._

### Profondeur du bottleneck : le code actuel n'empile qu'un seul bloc

L'architecture Perceiver IO de référence empile **k blocs**, chacun composé d'une cross-attention **suivie de l auto-attentions** sur le tableau de latents (complexité `O(kMN + klN²)`). Le code actuel (`model_library.py:1048-1052`) n'a **qu'un seul bloc** : une cross-attention (`inputs_q=queries, inputs_kv=board_tokens`) suivie d'**une seule** auto-attention. `k=1, l=1` alors que l'architecture de référence empile typiquement plusieurs blocs.
_Source: [Perceiver: General Perception with Iterative Attention (arXiv 2103.03206)](https://ar5iv.labs.arxiv.org/html/2103.03206)_

**Implication pratique** : la profondeur de traitement dans l'espace latent est un levier orthogonal à `num_bottleneck_tokens` — augmenter le nombre de tokens sans augmenter le nombre de passes d'auto-attention laisse le réseau avec une seule occasion de faire circuler l'information entre les 8 (ou plus) tokens latents avant le pooling final.

### Coût compute : token_dim est partagé entre backbone CNN et bottleneck — num_bottleneck_tokens ne l'est pas

Point d'intégration important : `token_dim` (actuellement 64) dimensionne **à la fois** les canaux du backbone CNN (`nn.Conv(self.token_dim, ...)`, `SeparableConv(self.token_dim, ...)`) **et** la dimension d'attention du bottleneck. Augmenter `token_dim` augmente donc le coût de tout le backbone CNN, pas seulement de l'attention. À l'inverse, `num_bottleneck_tokens` n'affecte que le bottleneck (la taille du tableau de latents `queries`), pas le backbone.

- Le nombre de paramètres d'un bloc Transformer croît approximativement en **hd²** (hidden dim au carré) — donc augmenter `token_dim` a un effet multiplicatif sur le coût.
_Source: [How OpenAI/DeepMind calculates transformer training cost](https://masteringllm.medium.com/how-openai-or-deepmind-calculates-cost-of-training-a-transformer-based-models-b0b629f0942b)_
- Le coût d'augmenter `num_bottleneck_tokens` (N, la taille du tableau de latents) croît **linéairement**, pas quadratiquement, pour la partie cross-attention (`O(kMN)` avec M=64 tokens-case fixes) — c'est le levier le moins cher des deux à faire varier.
_Source: [Perceiver IO (arXiv 2107.14795)](https://ar5iv.labs.arxiv.org/html/2107.14795)_

**Conséquence pratique** : pour un budget d'expérimentation donné, tester `num_bottleneck_tokens` (8→16→32) coûte structurellement moins cher que tester `token_dim` (64→128), alors que la littérature (§ précédente) suggère que c'est justement `token_dim` qui accuse le plus grand écart avec les architectures échecs de référence (64 vs 1024 pour Leela BT4). Le modèle total ne fait que 382K paramètres sur des plateaux 8×8 — la marge de manœuvre compute est large avant de devenir un problème pratique sur ce projet (GPU/TPU déjà en place pour l'entraînement).

### Interaction avec le reste du pipeline d'entraînement

- Le `learning_rate`/`warmup_steps`/`decay_steps` de `CHESS_SEARCH_TEACHER` sont **copiés tels quels de `CHESS_NO_HISTORY`** (`dataset_configs.py:745-748`, commentaire explicite "point de départ non tuné"), donc un changement de capacité du modèle (plus de paramètres) n'a **pas** été accompagné d'un re-tuning du schedule — un LR calé pour un modèle plus petit peut être sous-optimal pour un modèle agrandi.
- `value_weight=0.0` neutralise la tête value sans la supprimer — un changement de capacité du bottleneck partagé (`pooled`) affecte les deux têtes (policy et value) même si seule la policy est actuellement entraînée sur ce dataset ; pas un obstacle, mais un point à garder en tête si la tête value est réactivée plus tard sur un autre dataset avec le même `token_dim`.

---

## Architectural Patterns and Design

_Note méthodologique : le template générique (patterns microservices, sécurité API, déploiement) réinterprété — la question de "design" pertinente ici n'est pas structurelle-logicielle mais **méthodologique** : comment distinguer, empiriquement, une sous-capacité architecturale d'une difficulté intrinsèque de la tâche ou du signal professeur, avant de décider d'agrandir quoi que ce soit._

### Le piège diagnostique : un petit gap train/val ne suffit PAS à conclure "sous-capacité"

Point important qui nuance la lecture initiale du graphique `chess_search_teacher.png` : un gap train/val faible (1.92% ici) est **cohérent avec deux explications différentes**, pas une seule :

1. **Sous-capacité réelle** — le modèle est trop petit pour représenter la fonction cible, train et val plafonnent ensemble en dessous de ce qu'un modèle plus grand atteindrait.
2. **Tâche/signal intrinsèquement difficile** — le modèle a une capacité suffisante, mais la cible elle-même (label "meilleur coup" unique d'un professeur alpha-bêta simple, cf. section précédente) contient un plafond de performance atteignable, indépendant de la taille du modèle.

La distinction ne se fait **pas** en regardant le gap seul, mais en regardant le **niveau absolu** de la loss/accuracy : les deux cas produisent un plateau train≈val, mais pour des raisons différentes.
_Source: [Training Deep Networks from Zero to Hero (arXiv 2109.02752)](https://arxiv.org/pdf/2109.02752), [Overfitting vs Underfitting — Hex](https://hex.tech/blog/overfit-vs-underfit/)_

### Protocole empirique pour trancher (sans avoir à le faire à ta place)

La méthode standard pour séparer les deux hypothèses est un test direct, pas une inspection de courbe : **augmenter délibérément la capacité (n'importe quel levier — `num_bottleneck_tokens`, `token_dim`, ou nombre de blocs) sur un run court, et observer le **Train Accuracy**, pas le Val Accuracy** :

- Si le **Train Accuracy** monte significativement avec plus de capacité → c'était bien une sous-capacité (le modèle actuel ne pouvait pas mémoriser/représenter suffisamment même l'ensemble d'entraînement). Le Val Accuracy suivra ou non selon le volume de données.
- Si le **Train Accuracy** reste plafonné même avec beaucoup plus de capacité → le plafond vient de la tâche/du signal professeur (label alpha-bêta simple, cf. PRD l.168), pas de l'architecture — agrandir le modèle n'aidera pas.

Ce test est peu coûteux ici : vu la taille actuelle du modèle (382K paramètres, plateaux 8×8), un run de quelques epochs avec `num_bottleneck_tokens` doublé (8→16, le levier le moins cher d'après l'analyse compute ci-dessus) suffit à observer si le Train Accuracy décolle.

---

## Validation empirique (2026-08-06)

Test exécuté par Aymeric : `num_bottleneck_tokens` 8→16 sur `CHESS_SEARCH_TEACHER` (`dataset_configs.py:738`), 15 epochs, mêmes hyperparamètres sinon (LR/warmup/decay copiés de `CHESS_NO_HISTORY`, inchangés). Comparaison directe avec le run de référence (`chess_search_teacher.png`, 2026-08-05) :

| | 8 tokens (run du 2026-08-05) | 16 tokens (run du 2026-08-06) | Δ |
|---|---|---|---|
| Train PolicyAccuracy (finale) | 31.18% | 31.05% | **-0.13 pt** |
| Val PolicyAccuracy (finale) | 29.26% | 29.37% | +0.11 pt |
| Gap train/val | 1.92% | 1.68% | quasi inchangé |

**Lecture selon le protocole défini plus haut : le Train Accuracy n'a pas bougé (légèrement en baisse) malgré le doublement de `num_bottleneck_tokens`.** D'après le protocole, c'est le signal qui pointe vers l'hypothèse 2 (plafond intrinsèque à la tâche/au signal professeur) plutôt que l'hypothèse 1 (sous-capacité liée au nombre de tokens) — **pour ce levier précis**. Cohérent avec l'intuition d'Aymeric ("aucun vrai changement à part en micro%").

**Ce que ce résultat ne dit PAS** : il ne clôt pas la piste capacité en général, seulement `num_bottleneck_tokens`. Deux leviers de capacité restent non testés et identifiés comme structurellement plus significatifs dans cette recherche (§ Technology Stack Analysis, comparaison Leela BT4) :
- `token_dim` (64, partagé avec le backbone CNN — écart de 16× avec les moteurs échecs-transformer de référence)
- profondeur du bottleneck (`k=1, l=1` actuellement, vs plusieurs blocs empilés dans le Perceiver IO de référence)

Si l'hypothèse "plafond intrinsèque à la tâche" se confirme aussi sur ces deux leviers (Train Accuracy qui ne bouge toujours pas), ça rejoindrait la position déjà actée dans le PRD (l.168, décision Aymeric du 2026-08-04) : la qualité de jeu est plafonnée par le professeur alpha-bêta et n'est pas un critère chiffré de clôture de cette epic — auquel cas complexifier encore l'architecture ne serait pas le bon niveau de levier pour ce dataset précis.

---

## Validation empirique — Test 2 : token_dim (2026-08-07)

Test exécuté par Aymeric : `token_dim` 64→128 (`num_bottleneck_tokens` reste à 16), 15 epochs, run complet jusqu'au bout. Comparaison avec les deux runs précédents (`token_dim=64` dans les deux cas) :

| | 8 tokens / dim 64 (05/08) | 16 tokens / dim 64 (06/08) | 16 tokens / **dim 128** (07/08) |
|---|---|---|---|
| Train PolicyAccuracy (finale) | 31.18% | 31.05% | **42.51%** |
| Val PolicyAccuracy (finale) | 29.26% | 29.37% | **35.28%** |
| Gap train/val (finale) | 1.92pt | 1.68pt | **7.23pt** |

**Lecture selon le même protocole : le Train Accuracy a nettement décollé (+11pt vs les runs précédents), ET le Val Accuracy a suivi (+~6pt).** Contrairement au test `num_bottleneck_tokens`, ce résultat confirme l'hypothèse 1 (sous-capacité) pour `token_dim` — cohérent avec la comparaison Leela BT4 de la section Technology Stack (écart de dimension identifié comme le plus significatif des deux leviers).

**Trajectoire du gap sur les 15 epochs** (Val a clairement ralenti/plafonné en toute fin, pendant que Train continuait de grimper) :

| Epoch | Train | Val | Gap |
|---|---|---|---|
| 5 | 32.77% | 30.85% | 1.92pt |
| 8 | 38.45% | 34.02% | 4.43pt |
| 11 | 41.11% | 35.02% | 6.09pt |
| 13 | 42.20% | 35.32% | 6.88pt |
| 14 | 42.40% | 35.26% | 7.14pt (Val recule légèrement) |
| 15 | 42.51% | 35.28% | 7.23pt (Val quasi plat) |

Le schéma classique de surapprentissage apparaît nettement dans les 2-3 derniers epochs (Train continue, Val plafonne/oscille) — **le pronostic initial d'Aymeric ("on va avoir plus d'overfitting") se confirme dans les derniers epochs**, après une première moitié de run où Val progressait encore activement (pas de signal de surapprentissage prématuré). `patience=8` n'a pas déclenché d'arrêt anticipé (Val continuait de progresser marginalement jusqu'à la fin du schedule cosine).

**Conclusion pour ce levier** : `token_dim` est confirmé comme le vrai levier de capacité (contrairement à `num_bottleneck_tokens`), avec un gain net de +6pt de Val Accuracy — mais il ouvre la porte à un besoin de régularisation supplémentaire (dropout) ou de volume de données accru pour continuer à en tirer parti sans que le gap ne se creuse davantage.

---

## Validation empirique — Test 3 : token_dim 128→256, nouveau dataset depth=12 (2026-08-07/08)

**Changement de baseline important** : entre le Test 2 et celui-ci, le dataset `chess_search_teacher` a été régénéré côté `chess_ai` avec un professeur `depth=12` (était `depth=8`) — 10 000 parties, 1 402 252 positions (~2× le volume `depth=8`). Les PolicyAccuracy de cette section ne sont **pas comparables** à celles des Tests 1-2 ci-dessus (tâche plus difficile, coups moins "évidents"), mais restent comparables **entre elles** (même dataset `depth=12` des deux côtés).

Comparaison à `dropout=0.25`, `epochs=25` (`decay_steps` recalculé à la main pour ce volume, cf. `dataset_configs.py` — le recalcul automatique ne fonctionne pas pour le domaine échecs, voir `deferred-work.md` 2026-08-07) :

| | token_dim=128 (dernier point connu, epoch réelle 13/25) | token_dim=256 (final, epoch réelle ~26/25 — 1 epoch dupliquée suite à une interruption accidentelle, voir `deferred-work.md`) |
|---|---|---|
| Train | 24.93% | 39.27% |
| Val | 24.80% | 27.63% |
| Gap | +0.13pt | **+11.64pt** |

**Note méthodologique** : le run `token_dim=128`/`depth=12` n'a jamais été suivi jusqu'à son epoch 25 dans cette conversation (dernier point partagé : epoch réelle 13) — la comparaison ci-dessus n'est donc pas totalement bouclée des deux côtés. Ceci dit, aux points de comparaison intermédiaires disponibles (epochs réelles 1 à 13), `token_dim=256` menait systématiquement en Val avec une marge croissante (ex. +1.86pt à l'epoch réelle 11), donc l'avantage Val de `dim=256` sur `dim=128` à ce stade est bien établi — seule l'ampleur exacte de l'écart final reste incertaine.

**Lecture du plateau de fin de run** : Train ET Val se sont aplatis ensemble dans les 2-3 derniers epochs (Train 39.22%→39.27%→39.27%, Val 27.64%→27.63%→27.63%), synchronisé avec la fin du schedule cosinus (LR→~0 vers l'epoch 25 par construction). Ce plateau simultané des deux courbes ressemble davantage à un **arrêt mécanique dû à la fin du schedule LR** (déjà rencontré sur le run `depth=12`/`dim=128`, résolu en passant `epochs` de 15 à 25) qu'à un plafond intrinsèque de capacité ou de données — un plafond de capacité se manifesterait plutôt par un Val qui plafonne/recule *pendant que* Train continue de grimper, pas les deux ensemble. **Distinct du gap lui-même (+11.64pt)**, qui est un signal indépendant et solide : `token_dim=256` avec le même `dropout=0.25` et le même volume de données surapprend nettement plus que `token_dim=128` ne le faisait à un stade comparable — la capacité a maintenant dépassé ce que la régularisation/le volume actuels peuvent absorber, contrairement au saut précédent (64→128) où le dropout suffisait à garder un train/val raisonnable.

**Implication pratique** : avant de retester la capacité une nouvelle fois (`token_dim=512`, coûteux en VRAM — voir la marge déjà tendue à 78% sur une carte 6 Go), le prochain test à moindre coût est de repousser `epochs` au-delà de 25 (même logique que précédemment) pour vérifier si le plateau de fin de run est bien un artefact du schedule plutôt qu'un vrai plafond. Si le gap continue de se creuser sur un schedule plus long, ce serait le signal net pour augmenter `dropout_rate` au-delà de 0.25 sur cette config précise, ou pour reconsidérer le volume de données (piste mise de côté par Aymeric pour l'instant).



