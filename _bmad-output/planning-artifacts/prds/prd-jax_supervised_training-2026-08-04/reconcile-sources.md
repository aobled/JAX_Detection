# Réconciliation d'entrées — PRD `CHESS_SEARCH_TEACHER` vs 4 documents source

**Date** : 2026-08-04
**PRD vérifié** : `prd-jax_supervised_training-2026-08-04/prd.md`
**Méthode** : lecture intégrale des 4 documents source + du PRD, comparaison FR par FR / section par section. Toute citation de ligne renvoie au fichier tel que lu à cette date.

---

## 1. `docs/contract-chess-ai-training-interface.md` (§2.6 et §2.3)

**Verdict : aligné, aucune incohérence forte.**

- §2.6 (l.105-117) : `POLICY_KEY="policy"`, aucun `VALUE_KEY`, encodage inchangé (`include_legal_hint=True`, plans §2.1 existants), préfixe dédié `chess_search_teacher`. Tous ces points sont repris fidèlement par le PRD (Glossaire l.37-38, FR-1 l.56-59, FR-3 §4.2, Non-Goals l.135).
- §2.3 (l.59-66) : principe "la config sauvegardée doit porter tout ce qui est nécessaire pour que `chess_ai` charge sans supposition externe", invoqué explicitement par le PRD §4.3 (l.88) pour justifier `value_head_trained` — citation fidèle.

**Gap mineur (précision, pas contradiction)** : §2.6 ne fixe pas explicitement 29 (vs 19) comme nombre de canaux pour ce dataset — le texte renvoie seulement à "NUM_POSITION_PLANES/NUM_PLANES existants (§2.1)", qui liste les *deux* variantes possibles (19 sans historique, 29 avec). Le PRD (FR-1, l.56) affirme `num_channels=29` en citant "§2.1/§2.6 du contrat" comme si le contrat le figeait — en réalité, la valeur exacte (29, donc `include_history=True`) n'est confirmée que par le code du générateur (document 3, voir §3 ci-dessous), pas par le texte du contrat lui-même. À corriger en citant aussi la source code, ou en demandant que le contrat soit rendu plus explicite sur ce point lors de sa prochaine synchro.

---

## 2. `chess_ai/.../brief-model-without-search-2026-08-03.md`

**Verdict : globalement aligné, un gap réel identifié.**

### Gap réel : signal policy "plus riche" non repris

Section "Piste professeur, approfondie", point 1 (l.160-166) nomme **deux** pistes plus riches pour une étape ultérieure, symétriques l'une de l'autre :
- (a) **distribution sur les candidats** (softmax des scores negamax) au lieu d'un label policy unique ;
- (b) **score racine du professeur** comme nouvelle cible value (au lieu du label ±1/0 actuel basé sur le résultat de partie).

Le PRD ne reprend que (b) : Non-Goals l.137 ("Pas de cible value alternative... score racine négamax du professeur, mentionné comme piste plus riche dans le brief chess_ai") et Open Question 2 (l.174). **L'option (a) — distribution policy plus riche — n'apparaît nulle part dans le PRD** (ni Non-Goals, ni Open Questions, ni Assumptions Index), alors qu'elle est nommée au même niveau que (b) dans le document source comme piste "plus riche" envisageable plus tard. Absence probablement anodine (le PRD acte "label unique" comme le choix simple pour ce 1er test, ce qui exclut (a) et (b) de façon symétrique), mais l'asymétrie de traitement (b) documentée / (a) silencieuse est un vrai trou de traçabilité si un futur lecteur cherche "pourquoi pas une distribution de policy plus riche" dans ce PRD.

### Points vérifiés alignés (pas de gap)

- "Plafond du professeur" (l.171-177, matériel + bonus centre) ↔ PRD Non-Goals l.134, SM-C1 l.168. Fidèle.
- "Statut : SUSPENDU, pas rejeté" pour la pipeline composée (l.222-224) ↔ PRD Non-Goals l.136 ("reste suspendue côté contrat §3"). Fidèle, vocabulaire "suspendu" préservé (pas transformé en "abandonné").
- "Coût comparé... la moins chère à valider" (l.178-181, réutilise `ChessCnnAttentionPolicyValue` tel quel) ↔ Vision PRD l.19 (Option A). Fidèle.
- Origine des positions (self-play Stockfish via `generate_selfplay_positions`, l.167-170) : absente du PRD — correct, hors périmètre (génération de dataset explicitement en Non-Goals PRD l.135).

---

## 3. `chess_ai/dataset_builder/chess_search_teacher_dataset_tools.py`

**Verdict : aligné sur le contenu, mais une assumption du PRD est en réalité déjà résolue par le code.**

### Observation : Assumption §4.1 déjà confirmée, pas seulement "à confirmer"

Le PRD liste en `[ASSUMPTION §4.1]` (l.181) : *"`output_prefix` exact supposé être `{DATA_ROOT}/chunks/chess_search_teacher/chess_search_teacher`... à confirmer contre le chemin réel utilisé côté `chess_ai`"*.

Le code (l.169-172, bloc `__main__`) hardcode exactement ce chemin :
```python
output_prefix="/home/aobled/Documents/data/chunks/chess_search_teacher/chess_search_teacher",
```
Ce n'est pas une contradiction — l'assumption du PRD est *correcte* — mais elle est présentée comme encore ouverte alors que la source qui devait la trancher (ce fichier) le fait déjà, de façon vérifiable. À marquer "confirmé" plutôt que laissé en Assumptions Index lors du polish.

### Points vérifiés alignés (pas de gap)

- Seule `POLICY_KEY` écrite comme clé de label, aucun `VALUE_KEY` (l.30-41, docstring `_save_teacher_chunk`) ↔ PRD Glossaire l.37-38, FR-3.
- `include_history: bool = True` par défaut (l.53) → confirme concrètement le `num_channels=29` du PRD FR-1, mais **cette confirmation vient du code, pas du contrat** (voir gap mineur §1 ci-dessus — même point, vu depuis l'autre source).
- `teacher_depth` par défaut 4 (l.71-78) ↔ cohérent avec "profondeur 4" du contrat §2.6 ; absent du PRD, correctement laissé en Open Question 4 (hyperparamètres/détails d'implémentation hors scope PRD).
- Positions terminales sans coup légal ignorées silencieusement (pas d'erreur) (l.89-100) : détail de génération, hors du périmètre `jax_supervised_training` — absence dans le PRD est correcte (Non-Goals l.135, génération de dataset exclue).

---

## 4. `chess_ai/chess_model_inference.py` (`select_model_move` l.51-110, `load_chess_model` l.23-48)

**Verdict : aligné, aucune incohérence.**

- `load_chess_model` (l.23-48) construit le modèle depuis `config["model_name"]`/`config["num_classes"]`/`config.get("num_channels", NUM_PLANES)` — aucune dépendance à un nom de fichier ou une convention externe, exactement le principe cité par le PRD §4.3 (contrat §2.3). Puisque `CHESS_SEARCH_TEACHER` réutilise `model_name`/`num_classes`/`num_channels` du même schéma que l'existant, `load_chess_model` fonctionne sans adaptation — confirme le Non-Goal PRD l.133 ("le format de checkpoint reste consommable tel quel").
- `select_model_move` (l.86-88) : `value = float(np.asarray(out[VALUE_KEY][0]))` est extrait mais, selon la docstring (l.76-77), **"pas encore utilisée pour la sélection, juste remontée pour affichage/debug éventuel"**. Ceci confirme directement que Option A (tête value factice) ne casse rien côté sélection de coup — seul un futur affichage/debug verrait une valeur non entraînée (donc potentiellement trompeuse). Le PRD a déjà anticipé exactement ce point via la note `[NOTE FOR PM]` en §6.2 (l.151, "à ne pas oublier si `chess_ai` ajoute un jour un affichage de barre d'évaluation basé sur cette sortie"). Aucun gap.
- Rien dans ce fichier ne contredit l'affirmation du PRD Non-Goals (l.133) qu'aucune adaptation `chess_model_inference.py` n'est nécessaire dans cette epic.

---

## 5. Observation complémentaire (hors des 4 sources demandées — trouvée en croisant avec `dataset_configs.py`, pertinente pour FR-1)

Non demandée explicitement mais découverte en vérifiant la phrase FR-1 "num_channels=29... identiques aux configs échecs à historique existantes (§2.1/§2.6 du contrat), pas de nouveau schéma de plans" (l.56) :

`dataset_configs.py` montre qu'**aucune config actuellement active ne combine `model_name="chess_cnn_attention_policy_value"` (task_type `chess_policy_value`) avec `num_channels=29`** :
- `CHESS_NO_HISTORY` (seule config `chess_policy_value` active) : `num_channels=19` (l.596, "19, pas 29 - pas d'historique").
- `CHESS_LEGAL_MOVES` : `num_channels=29` (l.640) mais **`model_name="chess_cnn_attention_legal_moves"`**, `task_type="chess_legal_moves"` — un modèle et une stratégie complètement différents (multi-label sigmoid BCE, pas policy+value).
- L'ancienne config `CHESS` (policy+value, 29 canaux, avec historique) a été **retirée le 2026-08-02** (commentaire l.584-586 + commit `2e44404 Retirer les configs échecs CHESS et CHESS_NAKAMURA_NO_HISTORY`).

Donc `CHESS_SEARCH_TEACHER` serait en réalité la **première config actuellement active** à combiner `ChessCnnAttentionPolicyValue` avec une entrée 29 canaux/historique — pas une simple réutilisation d'une combinaison déjà éprouvée en production récente. La phrase "pas de nouveau schéma de plans" reste vraie au sens strict (le *format* de plans 29 canaux existe déjà, via `CHESS_LEGAL_MOVES`), mais pourrait laisser croire à tort qu'un dev peut copier `CHESS_NO_HISTORY` tel quel pour le `num_channels`/`input_shape` — ce serait une erreur (19 au lieu de 29). Cette confusion potentielle est amplifiée par le fait que le PRD ne nomme jamais explicitement `CHESS_LEGAL_MOVES` comme la seule source actuelle de "29 canaux avec historique" à consulter pour cette dimension (bien qu'il en cite un autre aspect ailleurs, la collision de préfixe). Recommandation pour la story de dev : préciser explicitement que le point de référence pour `num_channels=29`/`input_shape=(8,8,29)` est `CHESS_LEGAL_MOVES` (même si son modèle/task_type diffèrent), pas `CHESS_NO_HISTORY`.

---

## Synthèse

| # | Source | Type | Sévérité | Résumé |
|---|--------|------|----------|--------|
| 1 | Contrat §2.6/§2.3 | Gap mineur | Faible | Le contrat ne fixe pas explicitement 29 canaux pour ce dataset ; seule la valeur par défaut du code (doc 3) le fait. |
| 2 | Brief chess_ai | **Gap réel** | Moyenne | Piste "distribution policy plus riche" (softmax negamax) nommée dans le brief au même niveau que la piste value alternative, mais absente du PRD (Non-Goals/Open Questions). |
| 3 | dataset_builder | Observation | Faible | Assumption §4.1 du PRD déjà confirmée par le code, présentée à tort comme encore ouverte. |
| 4 | chess_model_inference.py | Aucun gap | — | Confirme intégralement les affirmations du PRD (Non-Goals, tête value factice sans impact fonctionnel). |
| 5 | dataset_configs.py (hors scope) | Observation notable | Moyenne | FR-1 pourrait induire en erreur : aucune config active ne combine déjà `chess_cnn_attention_policy_value` + 29 canaux ; le seul exemple 29 canaux (`CHESS_LEGAL_MOVES`) utilise un modèle différent. |

Aucune incohérence forte (affirmation du PRD directement contredite par un des 4 documents) n'a été trouvée. Les points 2 et 5 sont les plus substantiels et méritent un ajustement avant polish/finalisation.
