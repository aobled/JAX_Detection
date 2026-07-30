# Index de documentation — jax_supervised_training

_Généré par `bmad-document-project`, 2026-07-12. Point d'entrée principal pour tout travail assisté par IA sur ce projet._

## Vue d'ensemble du projet

- **Type** : monolithe, 1 part
- **Langage principal** : Python (JAX/Flax)
- **Architecture** : Strategy + Factory + Dependency Injection

## Documentation générée (scan brownfield)

- [Project Overview](./project-overview.md)
- [Architecture](./architecture.md)
- [Arborescence annotée (Source Tree)](./source-tree-analysis.md)
- [Guide de développement](./development-guide.md)
- [Audit code mort & duplications](./dead-code-and-duplication-audit.md) — base de travail du futur PRD de refactor

## Documentation existante (rédigée par l'auteur du projet)

- [strat_global.md](./strat_global.md) — Architecture globale du pipeline (diagramme mermaid)
- [strat_chunks.md](./strat_chunks.md) — Stratégie de chunking des datasets
- [strat_data_augmentation.md](./strat_data_augmentation.md) — Augmentation de données
- [strat_loss.md](./strat_loss.md) — Fonctions de loss
- [strat_LR.md](./strat_LR.md) — Learning rate scheduling
- [strat_pkl.md](./strat_pkl.md) — Sérialisation/export des modèles
- [strat_pruning_quantization.md](./strat_pruning_quantization.md) — Pruning/quantization
- [strat_reporting.md](./strat_reporting.md) — Reporting/métriques
- [strat_unet_detection.md](./strat_unet_detection.md) — Détection via UNet
- [strat_video_detection.md](./strat_video_detection.md) — Traitement vidéo
- [AircraftDetectorUNet.md](./AircraftDetectorUNet.md) — Doc du modèle UNet
- [ChessCnnAttentionPolicyValue.md](./ChessCnnAttentionPolicyValue.md) — Doc du modèle échecs (CNN + bottleneck d'attention Perceiver-style, Epic 9)

## Pour la suite (PRD brownfield)

Quand on passera à la rédaction du PRD de refactor, pointer le workflow PRD vers cet `index.md` — il résume déjà : la classification du projet, la stack, le pattern architectural, et surtout l'inventaire de dette technique (`dead-code-and-duplication-audit.md`) qui doit alimenter directement les epics du refactor.

## Getting Started

```bash
python main.py FIGHTERJET_CLASSIFICATION   # ou FIGHTERJET_DETECTION
```
Voir [development-guide.md](./development-guide.md) pour le détail (environnement conda `jax_env`, variable `JAX_SUPERVISED_TRAINING_DATA_ROOT` pour Colab).
