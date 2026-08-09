---
title: 'Stratégie CHESS_SEARCH_TEACHER — combler le gap train/val en capacité augmentée'
type: 'living-doc'
created: '2026-08-08'
status: 'active'
owner: 'Aymeric + Winston'
---

# Stratégie CHESS_SEARCH_TEACHER — combler le gap train/val

Document vivant (pas le `deferred-work.md` global, cross-dataset) — dédié au chantier de capacité/régularisation sur `CHESS_SEARCH_TEACHER`, dataset `depth=12` (10 000 parties, 141 chunks, 1 402 252 positions). Mis à jour au fil des tests, pas figé.

## État des lieux (2026-08-08)

Comparaison de référence, même dataset `depth=12`, même `dropout_rate=0.25`, mêmes 25 epochs — seul `token_dim` diffère :

| | token_dim=128 | token_dim=192 | token_dim=256 |
|---|---|---|---|
| Params | 875 073 | ~1 482 305 | 2 204 225 |
| Taille PKL | 3.5 Mo | 5.9 Mo | 8.8 Mo |
| Train finale | 27.49% | 32.81% | 39.27% |
| Val finale | 26.16% | 27.19% | 27.63% |
| Gap (dernier epoch) | 1.33pt | 5.62pt | 11.64pt |
| Overfitting (Simpson, toute la courbe) | -0.41% | — | 32.49% |

## 🏆 Meilleur résultat à ce jour (2026-08-08) : token_dim=192, dropout=0.35, label_smoothing=0.2

| | dim=192 (base, dropout=0.25) | dim=192 + dropout=0.35 | **dim=192 + dropout=0.35 + label_smoothing=0.2** | dim=256 (dropout=0.25, sans smoothing) |
|---|---|---|---|---|
| Params | ~1 482 305 | ~1 482 305 | ~1 482 305 | 2 204 225 |
| Train finale | 32.81% | 31.12% | 31.55% | 39.27% |
| Val finale | 27.19% | 27.38% | **28.00%** | 27.63% |
| Gap | 5.62pt | 3.74pt | **3.55pt** | 11.64pt |

**Premier levier de cette campagne qui améliore le Val ET réduit le gap en même temps** (+0.62pt de Val, -0.19pt de gap par rapport à dropout=0.35 seul) — contrairement à weight_decay (effet nul) et dropout (gap réduit, Val quasi figé). **Bat `token_dim=256` sur les trois axes simultanément** : Val plus haut (28.00% vs 27.63%), gap 3.3× plus sain (3.55pt vs 11.64pt), 33% moins de paramètres (~1.48M vs 2.2M). Confirme l'intuition initiale d'Aymeric sur le label smoothing.

**Signal supplémentaire, pas encore exploité** : contrairement au run `dim=256` (Train ET Val s'aplatissaient ensemble en fin de schedule, cohérent avec une fin de LR plutôt qu'un plafond réel), ce run **n'a toujours pas plafonné à l'epoch 25** — Val progresse encore franchement jusqu'au bout (27.66%→28.00% sur les 5 derniers epochs, pas de palier net). Un test `epochs` allongé (même geste peu coûteux que sur le run `dim=128`/`depth=12`, `epochs: 25 -> 35` par ex., `decay_steps` à recalculer en conséquence) pourrait encore progresser sur cette config précise — candidat naturel pour le prochain test, avant d'envisager le volume de données (Option 4).

**`token_dim=192` (Option 5, testée le 2026-08-08) confirme être le meilleur point coût/bénéfice des trois** : +1.03pt de Val par rapport à 128 pour +4.29pt de gap (~4.2pt de gap "payés" par point de Val gagné) — contre +0.44pt de Val supplémentaire de 192 à 256 pour +6.02pt de gap (~13.7pt payés par point de Val, un rapport ~3× moins bon). Le rendement se dégrade nettement au-delà de 192 sur ce volume de données.

**Décision actée (2026-08-08)** : `token_dim=192` retenu comme nouvelle base de travail. Prochain levier à tester : Weight Decay (Option 1), avant Label Smoothing (Option 3) — cf. Next steps.

**Rappel important** : `chess_search_teacher.png` (courbe de ce run) ne montre que les epochs 8-25 — historique tronqué par le bug de reprise non corrigé (`deferred-work.md`, 2026-08-08, "off-by-one sur la reprise d'entraînement"). Les vrais chiffres finaux ci-dessus restent exacts (lus dans l'annotation, pas dans la courbe tronquée).

## Options identifiées, par ordre de priorité proposé

### Option 5 — token_dim intermédiaire (192) — ✅ **terminé, retenu comme nouvelle base (2026-08-08)**

Chercher un point entre 128 (gap sain, Val plus bas) et 256 (Val plus haut, gap important) plutôt que de trancher entre les deux extrêmes déjà mesurés. `token_dim` doit rester divisible par `num_heads` (4) — 192 convient (192/4=48).

- **Effort** : trivial, une ligne de config (`dataset_configs.py`), même discipline qu'aujourd'hui (un seul levier à la fois).
- **Risque** : nul — juste un point de mesure supplémentaire sur un axe déjà exploré deux fois.
- **Résultat** : Val 27.19% (+1.03pt vs 128, -0.44pt vs 256), gap 5.62pt — meilleur rapport coût/bénéfice des trois points mesurés (voir tableau plus haut). **Retenu comme config de départ pour la suite du chantier.**

### Option 1 — Weight decay

Jamais retuné cette session (`weight_decay=5e-5`, hérité tel quel de `CHESS_NO_HISTORY` depuis la création de `CHESS_SEARCH_TEACHER`). Mécanisme orthogonal au dropout (pénalité L2 via AdamW). 5e-5 est plutôt faible dans l'absolu (valeurs courantes 1e-4 à 1e-2) — marge réelle inexploitée.

- **Effort** : trivial, une ligne de config.
- **Risque** : faible — testé seul (dropout inchangé), comparaison propre garantie par construction.
- **Résultat (2026-08-08) : effet nul, dans le bruit.** `weight_decay: 5e-5 -> 5e-4` (×10) sur `token_dim=192` : Train 32.81%→32.92% (+0.11pt), Val 27.19%→27.25% (+0.06pt), Gap 5.62pt→5.67pt (+0.05pt) — aucun effet mesurable dans un sens ou dans l'autre. Note annexe trouvée en préparant ce test : `optax.adamw` (`trainer.py:201`) s'applique sans masque, donc décale aussi les paramètres `BatchNorm`/biais — pas la pratique la plus fine, pas changé pour ce test (hors scope).
  - **Hypothèse pour expliquer le résultat nul** : la pénalité L2 du weight decay est un effet doux (biaise les poids vers de petites valeurs) comparé au dropout (masquage stochastique, réduit la co-adaptation) — probablement pas le bon type de régularisation pour combler un écart piloté par un vrai mismatch capacité/volume de données, plutôt qu'un simple excès de magnitude des poids.
  - **Décision proposée (Winston, à valider avec Aymeric)** : ne pas insister sur ce levier (un saut ×100 depuis la valeur d'origine, 5e-3, pourrait révéler un effet à une échelle beaucoup plus forte, mais le dropout a déjà fait ses preuves sur ce type d'écart — cf. run 2026-08-07, dim=128/depth=8, 0.1→0.25) — passer à l'Option 2 (dropout) en priorité plutôt que ré-itérer sur weight_decay à l'aveugle.

### Option 2 — Dropout, encore augmenté

`dropout_rate` actuellement à 0.25 — modéré dans l'absolu (0.3-0.5 est courant pour un modèle de cette taille), donc de la marge existe malgré l'intuition initiale d'Aymeric ("déjà très haut").

- **Effort** : trivial, une ligne de config.
- **Risque** : réel à surveiller — la dernière hausse (0.1→0.25 sur `dim=128`/dataset `depth=8`) avait réduit le gap mais quasi pas amélioré le Val. Le lever à nouveau pourrait faire baisser Train ET Val ensemble sans bénéfice net. **Regarder le Val absolu, pas seulement le gap, à chaque test.**
- **Résultat (2026-08-08) : confirme exactement l'hypothèse posée avant le run.** `dropout_rate: 0.25 -> 0.35` sur `token_dim=192` : Train 32.81%→31.12% (-1.69pt), Val 27.19%→27.38% (+0.19pt seulement), Gap 5.62pt→3.74pt (-1.88pt). Deuxième confirmation du même schéma que le run 0.1→0.25 (2026-08-07) : le dropout referme le gap en rognant sur Train, sans faire vraiment progresser le Val. **Paramètres conservés** (meilleur point connu à ce jour), mais le dropout est maintenant écarté comme levier de progression du Val — deux mesures concordantes, pas un coup de bruit isolé.

### Option 3 — Label smoothing sur la tête policy

Techniquement solide — voir section dédiée ci-dessous pour la discussion complète (pourquoi ce n'est pas une "augmentation").

- **Effort réel, mesuré** : petit, pas moyen comme estimé initialement. `smooth_labels()` (`utils.py:15-28`, déjà validée sur `FIGHTERJET_CLASSIFICATION`) réutilisée telle quelle — prend des labels entiers, exactement le format de `targets["policy"]`. Le pont config→loss existait déjà (`ChessPolicyValueStrategy.compute_loss` relaie `**self.loss_params`, `task_strategies.py:499-500`) — ajouter `label_smoothing` au dict `loss_params` a suffi, aucun nouveau mécanisme de plomberie.
- **Implémenté et testé le 2026-08-08** (`loss_functions.py::compute_chess_policy_loss`/`compute_chess_policy_value_loss`) : `label_smoothing=0.0` (défaut) reste bit-à-bit identique à l'ancienne formule sparse (`optax.softmax_cross_entropy_with_integer_labels`) — vérifié par test direct. `label_smoothing>0` bascule vers la variante dense (`optax.softmax_cross_entropy`) avec cible assouplie. Gradient fini vérifié. `value_loss` jamais affectée (MSE, pas une classification). Suite de tests existante (`tests/test_chess_search_teacher_loader.py`) : 5/5 toujours au vert.
- **Risque** : faible — changement additif, comportement par défaut inchangé pour toutes les autres configs (`CHESS_NO_HISTORY` inclus, qui n'a pas cette clé dans son `loss_params`).
- **Statut** : **lancé le 2026-08-08** sur `token_dim=192`/`dropout=0.35` (meilleure config connue) — `label_smoothing=0.2` (0.1 initialement proposé — valeur standard de la littérature — revu à 0.2 avant le run : le dropout venait de montrer un effet Val de seulement ~0.2pt pour un changement substantiel, donc 0.1 risquait de rester ambigu, noyé dans ce même bruit).

### Option 4 — Volume de données supplémentaire, même typologie de professeur

Accord entre Aymeric et Winston : ne pas mélanger les types de labels (recherche `depth=12` vs coup joué par un humain GM) — si extension de volume, la faire via le même professeur (`chess_search.py`, depth=12), pas via des parties humaines annotées différemment.

- **Point ouvert soulevé par Aymeric** : comment éviter les doublons si on relance une génération de parties depuis la même source (auto-jeu + professeur) — voir section dédiée ci-dessous.
- **Effort** : le plus lourd des 5 — génération de données côté `chess_ai`, hors de ce repo.
- **Statut** : en discussion, pas priorisé avant les options 1-2-3-5 (moins cher, plus rapide à mesurer).

## Discussion — Option 3 : pourquoi le label smoothing n'est pas une "augmentation"

Point soulevé par Aymeric : une règle personnelle s'était établie contre toute "augmentation" sur les échecs — jugée absurde pour ce domaine. Cette règle est déjà actée dans le code (`data_management.py:615-617`, docstring `ChessPolicyValueDataset`) : flip/zoom/translation "n'ont pas de sens géométrique direct sur un plateau encodé en planes" — vrai et toujours valable. Un plateau d'échecs n'a pas la symétrie d'une image naturelle (pions avancent dans une seule direction, roque asymétrique par côté) : une rotation ou un flip vertical produirait une position qui n'a simplement plus de sens.

**Le label smoothing n'est pas dans cette famille.** Il ne touche jamais l'entrée (le plateau) — seulement la cible utilisée dans la loss : au lieu d'un objectif "ce coup précis est correct à 100%, tous les autres à 0%", une petite masse de probabilité est répartie sur les autres classes (`y_smooth = y*(1-ε) + ε/K`). C'est une déclaration sur la confiance qu'on accorde au label, pas une transformation géométrique ni un exemple synthétique — ça s'applique aussi bien à un problème de classification d'images qu'à une tâche de langage, indépendamment de toute structure spatiale.

**Pourquoi c'est même particulièrement pertinent ici** : le label actuel (coup choisi par la recherche alpha-bêta depth=12) est traité comme LA seule bonne réponse, alors que beaucoup de positions ont plusieurs coups quasi équivalents — un target dur à 100% est probablement sur-confiant par rapport à la réalité du jeu. Le label smoothing encode explicitement cette incertitude. Il y a même un lien direct avec une piste déjà identifiée dans le PRD d'origine (`prd.md` l.174, Open Question 2) : "distribution policy via softmax des scores negamax du professeur" — un signal de professeur plus riche, mis de côté pour le premier test. Le label smoothing est une **approximation low-cost de cette même idée**, sans toucher à la génération côté `chess_ai`.

## Discussion — Option 4 : le risque réel n'est pas le doublon en soi

Question d'Aymeric : comment éviter les doublons si on relance une génération de parties depuis la même source.

**Le vrai risque n'est pas le gaspillage de volume (des positions dupliquées n'ajoutent juste pas d'information nouvelle, sans casser quoi que ce soit) — c'est la fuite train/val.** Si une position (quasi-)dupliquée se retrouve répartie par hasard entre un chunk assigné au train et un chunk assigné au val (`ChessPolicyValueDataset`, split par fraction de chunks, graine fixe), le modèle pourrait "reconnaître" en val une position vue en train — un Val artificiellement gonflé, pas une vraie mesure de généralisation.

Deux questions à vérifier côté `chess_ai` (hors périmètre de ce repo) avant de relancer une génération :
- La génération actuelle varie-t-elle sa graine aléatoire (ouvertures, auto-jeu) d'un run à l'autre, ou repartirait-elle des mêmes parties si relancée telle quelle ?
- Un dédoublonnage par contenu (hash position+coup) avant finalisation des chunks serait la garde-fou le plus robuste, indépendamment de la réponse à la question précédente.

## Next steps

1. ~~Lancer le test `token_dim=192` (Option 5)~~ ✅ fait, retenu comme base (2026-08-08).
2. ~~Weight Decay (Option 1)~~ ✅ fait — effet nul, écarté (2026-08-08).
3. ~~Dropout (Option 2)~~ ✅ fait — gap réduit (5.62→3.74pt) mais Val quasi inchangé (+0.19pt), deuxième confirmation du même schéma que le run 2026-08-07. Paramètres conservés (`dropout=0.35`), mais écarté comme levier de progression du Val.
4. ~~Label smoothing (Option 3)~~ ✅ **fait — meilleur résultat de la campagne (2026-08-08)**. `label_smoothing=0.2` : Val 27.38%→28.00% (+0.62pt) ET gap 3.74→3.55pt (-0.19pt) simultanément — bat `token_dim=256` sur tous les axes. Voir section 🏆 en haut du document. Paramètres retenus comme nouvelle meilleure config.
5. **En cours (2026-08-08) : `epochs: 25 -> 35` sur cette même config** (`dim=192`/`dropout=0.35`/`label_smoothing=0.2`) — Val n'avait pas plafonné à l'epoch 25 (encore +0.34pt sur les 5 derniers epochs, décélération mais pas de palier net). `decay_steps` recalculé pour ce volume : gpu 123225→172515, tpu 246475→345065 (`dataset_configs.py`, marqué `⚠️ TEMPORAIRE`).
   - ⚠️ **Test invalidé (2026-08-08)** : lancé par erreur en reprise (checkpoint de l'ancien schedule non vidé), pas en run propre. Au-delà du bug d'off-by-one déjà connu, changer `decay_steps` tout en reprenant un checkpoint entraîné sous l'ancien schedule crée une vraie discontinuité de LR (voir `deferred-work.md`, nouvelle entrée 2026-08-08) — le modèle n'a jamais rebattu 0.2800 avant l'early stopping (epoch 32). **Résultat non concluant, pas une preuve que plus d'epochs n'aiderait pas** — juste un test contaminé. **Non retenté** : Aymeric préfère garder la config propre connue plutôt que refaire un run propre pour une question dont il attendait de toute façon peu. `epochs`/`decay_steps` revenus à 25/123225/246475 (config propre restaurée).
6. Volume de données (Option 4) : reste en réserve si un besoin de progression supplémentaire se présente — questions de dédoublonnage à trancher côté `chess_ai` avant de relancer une génération.

## Config retenue (2026-08-08) — clôture de la campagne de régularisation

`token_dim=192`, `dropout_rate=0.35`, `weight_decay=5e-5` (inchangé), `label_smoothing=0.2` — **Val=28.00%, Train=31.55%, gap=3.55pt**, meilleur résultat toutes configs confondues (bat `token_dim=256` sur les trois axes : Val, gap, taille). Config active dans `dataset_configs.py`. Campagne considérée close sauf besoin futur de repousser plus loin (volume de données, Option 4, en réserve).
