---
title: 'Fonction de reporting reporting_all_images_by_boxes_count'
type: 'feature'
created: '2026-08-26'
status: 'done'
route: 'one-shot'
review_loop_iteration: 0
context: []
---

# Fonction de reporting reporting_all_images_by_boxes_count

## Intent

**Problem:** `tools/reporting_dataset_pandas.py` n'offrait aucun moyen d'afficher/exporter la liste complète des images ayant exactement un nombre donné de boîtes (`box_count`), alors que ce besoin ponctuel revient régulièrement pour l'exploration manuelle du dataset.

**Approach:** Ajout d'une fonction autonome `reporting_all_images_by_boxes_count(df, boxes_count_nb)`, sur le modèle des fonctions `reporting_*` existantes (groupby par `base_image_name`, filtre, print, export CSV), triée par `base_image_name` pour une lecture manuelle cohérente.

## Suggested Review Order

- Nouvelle fonction de reporting, filtre par nombre exact de boîtes et exporte en CSV trié par image.
  [`reporting_dataset_pandas.py:214`](../../tools/reporting_dataset_pandas.py#L214)
