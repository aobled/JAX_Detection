# Rétro rapide — chantier compute_dtype (2026-08-17)

*Format solo, pas de dialogue d'équipe fictif (convention déjà établie pour ce projet).*
*Pas un Epic tracké (aucune entrée `epics.md`/`sprint-status.yaml`) — chaîne ad hoc `bmad-spec` → `bmad-architecture` → 4× `bmad-quick-dev`, 7 commits (`034002a`..`0d6ab1f`).*

## Livré

Mécanisme `compute_dtype` généralisé à 11 des 12 modèles de `MODELS` (les 6 non-échecs à 100%). Un amendement d'architecture réel (AD-4, BatchNorm/LayerNorm) fait et validé en cours de route. Un code mort trouvé et supprimé (`centernet_lite`). Validation matérielle réelle TPU pour CIFAR10/FIGHTERJET_CLASSIFICATION (avant et après l'amendement) ; pas encore pour les modèles de détection.

## Ce qui a bien marché

- **La revue à 2 angles (Blind Hunter + Edge Case Hunter) a trouvé quelque chose de réel à chaque cycle, même sur du travail "mécanique".** Pas juste de la validation cosmétique : régression `KeyError`, validation manquante, et surtout la découverte BatchNorm/LayerNorm qui a changé une décision d'architecture déjà livrée. Confirme que la revue systématique vaut le coût même quand le patron semble déjà établi et répété.
- **Vérifier "ce modèle est-il vraiment utilisé" avant d'écrire le spec a évité du travail perdu.** La découverte que `centernet_lite` était mort est venue d'une lecture de `dataset_configs.py` faite AVANT de lancer l'implémentation, pas après.
- **Découper le rollout en petits cycles quick-dev séparés (plutôt qu'un seul gros diff)** a permis à la découverte BatchNorm/LayerNorm (survenue au milieu du rollout) d'être intégrée directement dans l'implémentation initiale des cycles suivants, au lieu de nécessiter un retrofit de masse après coup.

## Ce qui a été dur / surprenant

- **L'amendement AD-4 n'a pas été anticipé au design initial.** La première version (Conv/Dense uniquement) était raisonnable sur le papier (raisonnement "seules les couches matmul lourdes bénéficient d'un calcul réduit", vérifié correct en isolation) mais a raté l'effet de reset en cascade — trouvé seulement en inspectant `capture_intermediates` sur le VRAI réseau, pas sur des tests de couche isolée. Un test de couche isolée aurait donné un faux sentiment de sécurité.
- **Deux fois, un subagent d'implémentation s'est arrêté en attendant son propre process en arrière-plan sans rapporter le résultat** — récupéré via reprise, mais un aller-retour de plus à chaque fois.

## Leçon à garder

Pour du travail de précision numérique/mixed-precision spécifiquement : tester sur la topologie réelle du réseau (pas seulement des couches isolées) est ce qui révèle les effets en cascade — une leçon qui vaudra pour la généralisation aux modèles échecs si elle reprend un jour.

## Ouvert (déjà tracé dans `deferred-work.md`, pas dupliqué ici)

- Validation matérielle réelle pour `AircraftDetectorUNet`/`CenterNet` — à faire au prochain vrai run `FIGHTERJET_DETECTION`/`JAX_DETECTOR`.
- Rollout aux 5 modèles échecs — session future, explicitement différé par Aymeric.
- Gaps de test mineurs (`float16` jamais exercé, dérive `batch_stats` sur la durée).
