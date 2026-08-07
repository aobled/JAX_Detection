---
title: 'Fix couleur ticks mineurs LR + superposition axes Accuracy/LR (training chart)'
type: 'bugfix'
created: '2026-08-07'
status: 'done'
route: 'one-shot'
review_loop_iteration: 0
context: []
---

# Fix couleur ticks mineurs LR + superposition axes Accuracy/LR (training chart)

## Intent

**Problem:** Sur le graphique d'entraînement (`TrainingVisualizer.plot_training_curves`), les labels de ticks mineurs de l'axe Learning Rate (ex. "6×10⁻⁴") s'affichaient en noir au lieu de vert (`tick_params` ne cible que `which='major'` par défaut), et l'axe LR occupait 60pt de largeur supplémentaire (`outward`) séparé de l'axe Accuracy, alors que les deux pourraient partager la même zone.

**Approach:** Ajout de `which='both'` sur `ax3.tick_params` pour colorer aussi les ticks mineurs. Réduction de l'offset `outward` de l'axe LR de 60pt à 18pt (superposition quasi complète avec l'axe Accuracy tout en gardant un petit écart pour éviter la collision texte-sur-texte entre labels des deux axes, confirmée par revue adversariale). Repositionnement vertical des deux labels d'axe (`Accuracy %` en haut, `Learning Rate` en bas) via `set_label_coords`. Légende déplacée de `lower right` à `upper left` car la marge droite est désormais occupée par les deux axes sur toute la hauteur du graphique (LR log peut descendre jusqu'en bas de plage en fin d'entraînement).

## Suggested Review Order

- Fix principal : ticks mineurs LR colorés en vert (cause du bug : `tick_params` cible `which='major'` par défaut)
  [`reporting.py:827`](../../reporting.py#L827)

- Superposition des axes Accuracy/LR : offset réduit à 18pt (pas 0 — collision de labels confirmée empiriquement en revue adversariale, voir commentaire) + repositionnement des deux labels d'axe en haut/bas
  [`reporting.py:801-813`](../../reporting.py#L801-L813)

- Légende déplacée en `upper left` : `lower right` entrait en collision avec les labels LR bas de plage une fois les deux axes superposés
  [`reporting.py:837-841`](../../reporting.py#L837-L841)
