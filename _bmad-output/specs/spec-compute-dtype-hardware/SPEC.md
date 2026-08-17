---
id: SPEC-compute-dtype-hardware
companions: ['../../planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md']
sources: []
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability only — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Sélection automatique de compute_dtype par matériel

## Why

`jax_supervised_training` vise à rester compatible avec tout type de modèle, mais le mécanisme de précision mixte interne (`compute_dtype`, dtype de calcul des couches matmul lourdes, poids maîtres restant `float32`) n'existe aujourd'hui que sur `ChessMoveTokenTransformer`, né du spike échecs Epic 11 (2026-08-11) et jamais généralisé. Pire, la valeur est une string codée en dur par config (`dataset_configs.py:966`) plutôt que dérivée du matériel réellement détecté au runtime — alors que ce même fichier (`main.py:31-41`) détecte déjà le backend pour un autre cast (celui des données d'entrée). En clôturant la branche d'exploration échecs, ce point avait été explicitement reporté "à une future epic" pour généralisation (sprint-status.yaml, action epic 11, backlog) : c'est cette dette qui est traitée ici.

## Capabilities

- **CAP-1**
  - **intent:** Le point d'entrée d'entraînement dérive `compute_dtype` automatiquement du backend matériel détecté au runtime, au lieu de le lire comme string codée en dur dans la config du dataset.
  - **success:** Lancer n'importe quelle config sur TPU applique `bfloat16` au calcul interne des couches concernées sans qu'aucune clé `compute_dtype` ne soit présente dans la config ; le même lancement sur GPU ou CPU applique `float32`.

- **CAP-2**
  - **intent:** Le mécanisme est générique et disponible à tout type de modèle, pas dupliqué domaine par domaine ni réservé aux échecs.
  - **success:** CIFAR10 (`sophisticated_cnn_32_plus`) ET FIGHTERJET_CLASSIFICATION (`sophisticated_cnn_128_lite`) déclarent/consomment `compute_dtype` sans logique de détection matérielle propre à leur classe ; validé par exécution réelle en local (CIFAR10, GPU/CPU) puis via Colab (FIGHTERJET_CLASSIFICATION, TPU), sans régression sur les autres domaines (échecs, détection). Chaque modèle adopté a un test dédié prouvant que le dtype est réellement observé (paramètres/sortie diffèrent sous `bfloat16` vs `float32`), pas seulement que l'appel ne lève pas d'erreur.

## Constraints

- Non-régression obligatoire sur `CHESS_MOVE_TOKEN`, seul consommateur actuel de `compute_dtype` (comportement `bfloat16`/TPU inchangé, checkpoint-compatible) — y compris sa logique de validation string→dtype existante (`model_library.py:1218-1221`) et ses tests associés, préservés tels quels, non centralisés/retirés.
- Les poids maîtres (checkpoint) restent `float32` dans tous les cas — seule la dtype de CALCUL des couches varie ; pattern déjà établi par `ChessMoveTokenTransformer`, à préserver.
- Le forwarding est **par introspection** : `main.py` n'injecte `compute_dtype` qu'aux modèles dont la factory déclare explicitement un paramètre **nommé** `compute_dtype` — jamais un registre maintenu à la main, jamais un flag de config par dataset. *(Révise la décision initiale "forwarding inconditionnel" — voir Decisions.)*
- `compute_dtype` s'applique uniquement aux couches matmul lourdes (`nn.Conv`/`nn.Dense`) — jamais à `nn.BatchNorm`/`nn.LayerNorm`/`nn.Embed`, en cohérence avec le précédent `ChessMoveTokenTransformer`.
- Sur GPU, le choix par défaut (`float32`) est une décision de prudence projet (aucun gain mesuré à ce jour, 2 crashs GPU locaux déjà documentés dans ce projet), pas une limite matérielle — activer bfloat16/float16 sur GPU reste explicitement hors scope.
- La seule façon sanctionnée pour un modèle d'être exempté de `compute_dtype` dérivé du matériel est de ne pas déclarer le paramètre nommé — un flag de config par dataset reste interdit dans tous les cas, y compris présenté comme une exception ponctuelle.

## Non-goals

- Activer ou tester `bfloat16`/`float16` sur GPU (décision distincte, à mesurer séparément plus tard).
- Modifier le mécanisme séparé de cast de l'ENTRÉE (`dtype` global, `main.py:31-41`, `float16` sur TPU et GPU) — sujet distinct.
- Étendre `compute_dtype` aux 9 autres modèles de `MODELS` (`aircraft_detector_unet`/`centernet`/`centernet_lite`, `chess_cnn_attention_policy_value`/`legal_moves`, `chess_token_candidate_model`, `chess_token_one_move_model`, `kepler_1d_cnn`, `sophisticated_cnn_128_plus`) — rollout complet différé, décision à reprendre modèle par modèle après revue de l'impact CIFAR10/FIGHTERJET.
- Créer un nouveau domaine ou dataset.

## Success signal

Un test réel en local sur CIFAR10 (GPU/CPU) démontre que `compute_dtype` se résout à `float32` sans régression, et un test réel via Colab sur FIGHTERJET_CLASSIFICATION (TPU) démontre que `compute_dtype` se résout à `bfloat16` et s'applique correctement au calcul du modèle — même niveau de rigueur que la validation AD-29 (Epic 11) : inspection réelle du dtype/des valeurs après passage dans le pipeline (test dédié par modèle, pas une simple absence d'erreur), pas seulement lecture de code.

## Decisions

*(2026-08-17, résolues par Aymeric)*

- La string codée en dur `"compute_dtype": "bfloat16"` de `CHESS_MOVE_TOKEN` (`dataset_configs.py:966`) est retirée une fois la dérivation automatique en place ; la détection runtime devient la source unique.
- Scope confirmé à exactement 2 modèles : CIFAR10 (`sophisticated_cnn_32_plus`) et FIGHTERJET_CLASSIFICATION (`sophisticated_cnn_128_lite`) — les deux nécessaires pour que le plan de test (local + Colab TPU) valide réellement quelque chose, ce sont deux classes top-level séparées malgré des sous-modules partagés.

*(2026-08-17, renversée pendant la spine `bmad-architecture` — voir companion)*

- **Forwarding par introspection, PAS inconditionnel.** La décision initiale ("forwarding unconditionnel, pas de registre/introspection") a été renversée après l'audit brownfield de la spine `architecture-compute-dtype-hardware-2026-08-17` : un forwarding littéralement inconditionnel aurait cassé 6 des 12 modèles de `MODELS` (`TypeError`, aucun `**kwargs` ni champ `compute_dtype`) et fait absorber silencieusement la valeur par 3 autres (`aircraft_detector_unet`/`centernet`/`centernet_lite`, bug pré-existant déjà documenté : `**kwargs` accepté en façade mais jamais transmis au constructeur) — contredisant l'objectif de non-régression de CAP-2 lui-même. Le détail du mécanisme et sa justification complète vivent dans le companion `ARCHITECTURE-SPINE.md` (AD-3).
