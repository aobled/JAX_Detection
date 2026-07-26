# Refactorisation — `tools/boxes_process_manual_tkinter.py`

**Origine** : `deferred-work.md` (item du 2026-07-15, repris explicitement en session dédiée le 2026-07-24, comme annoncé lors de l'audit Groupe 4/4 du 2026-07-19 — "une session dédiée séparée lui sera consacrée"). Analyse et stratégie par Winston (architecte) ; l'implémentation étape par étape reviendra à Amelia (dev), une story à la fois.

**Contrainte non-négociable (AD-20, `ARCHITECTURE-SPINE.md`)** : ce fichier est un consommateur protégé du pipeline `FIGHTERJET_DETECTION`/`AircraftDetectorUNet` legacy — il charge `best_model_detection.pkl` (UNet) et `best_model.pkl` (classification) via `load_detection_model`/`build_predict_fn`/`decode_segmentation_and_detect_batch` (`inference_utils.py`). **Le refactor ne doit strictement rien changer au comportement observable ni au pipeline utilisé** (pas de bascule vers `JAX_DETECTOR`/`build_single_pass_predict_fn` — ce serait une décision fonctionnelle séparée, hors sujet ici). Chaque étape doit être testable indépendamment sur de vraies images (Aymeric en a plusieurs centaines à trier — validation continue en conditions réelles, pas de test synthétique nécessaire).

## État des lieux (1284 lignes, lecture complète)

Deux classes :
- **`ImageManager`** (lignes 7-54, ~48 lignes) — déjà correctement isolée : liste des images, dossier de sauvegarde, lecture des boîtes JSON. Rien à faire ici.
- **`PhotoViewer`** (lignes 56-1274, ~1220 lignes) — classe unique, 8 responsabilités mélangées, aucune séparation :

| Cluster | Méthodes | Lignes approx. |
|---|---|---|
| **A. Setup/cycle de vie GUI** | `__init__`, `bind_events`, `quit_app` | 56-121, 203-216, 1273-1274 |
| **B. Dispatch clavier/souris** | `handle_key_press`, `handle_numeric_keypad`, `handle_regular_key`, `on_click`, `on_drag`, `on_mouse_move`, `on_right_click/drag/release`, `zoom_in/out` | 217-258, 342-349, 370-418, 539-590 |
| **C. Rendu Canvas** | `load_image`, `zoom`, `draw_bounding_boxes`, `update_handles`, `draw_crop_zone`, `get_color_for_bbox_count`, `update_window_color*`, `update_title_with_progress` | 311-368, 419-484, 486-537, 816-848, 1028-1046 |
| **D. CRUD boîtes (logique métier)** | `add_new_box`, `delete_bbox`, `edit_category_name`, `update_bbox_coords`, `fill_box_full_image`, `fill_box_full_width` | 259-298, 419-421, 591-629, 631-719 |
| **E. Persistance JSON** | `validate_and_fix_bbox_coordinates`, `ensure_json_consistency`, `save_json_with_consistency_check` | 123-201 |
| **F. Navigation/export image** | `show_next_image`, `delete_and_next`, `crop_bottom_zone`, `image_save_folder`, `image_save_tmp` | 299-309, 320-340, 721-914 |
| **G. Auto-traitement (thread)** | `start/stop_auto_processing`, `auto_process_all`, `_reload_current_image`, `_load_next_image_for_auto_processing` | 914-1026 |
| **H. Inférence JAX (ML)** | `toggle_predictions`, `load_jax_models`, `generate_and_show_predictions`, `accept_predictions` | 1048-1271 |

Pas un défaut fonctionnel — **le fichier marche**, confirmé par un usage réel prolongé. C'est un problème de responsabilité unique (SRP) et de duplication, pas de correction.

## Constats vérifiés (pas supposés)

- **Code mort confirmé** : `_reload_current_image` (ligne 996) est défini, jamais appelé nulle part (grep confirmé) — `_load_next_image_for_auto_processing` (ligne 1012, corps identique) l'a remplacé sans que l'original soit retiré.
- **Duplication confirmée (conversion de coordonnées zoom→original)** : le même calcul (`x1_original = x1_zoomed / zoom_factor`, etc.) apparaît **5 fois** à l'identique (lignes 663-666, 709-712, 751-754, 806-809, 1178-1181) — jamais factorisé en une fonction.
- **Duplication confirmée (`image_save_folder` vs `image_save_tmp`)** : ~90% de code partagé (construction de `bbox_coords_map`, conversion de coordonnées, sauvegarde JSON via `save_json_with_consistency_check`) — ne diffèrent que par le déplacement du fichier image et l'avancée à l'image suivante.
- **Cluster H (inférence JAX) est le plus indépendant** : ne touche `self` que pour lire `self.original_image`/`self.bbox_coords` et écrire `self.last_predictions`/`self.image`/`self.tk_image` — aucune dépendance profonde au reste de l'état GUI. C'est le candidat naturel d'extraction la plus sûre.
- **Cluster E (persistance JSON) est presque pur** : `validate_and_fix_bbox_coordinates`/`ensure_json_consistency` ne dépendent que de leurs paramètres, pas de `self` — extraction quasi mécanique.

## Stratégie de refactorisation — incrémentale, testée à chaque étape

Principe directeur : **extraire du plus indépendant/moins risqué vers le plus couplé/plus risqué**, jamais l'inverse. Chaque étape est une story Amelia distincte, livrable et testable seule, sans dépendre des étapes suivantes.

### Étape 1 — Extraire l'inférence JAX (cluster H)
Nouvelle classe (nouveau fichier, ex. `tools/boxes_manual_prediction_assistant.py`) portant `load_jax_models` + `generate_and_show_predictions` + l'état associé (`det_predict_fn`, `clf_predict_fn`, `det_config`, `clf_config`, `dataset_mean`, `dataset_std`, `last_predictions`). `PhotoViewer.toggle_predictions`/`accept_predictions` deviennent de simples appelants. **Pourquoi en premier** : plus gros gain de lisibilité (~170 lignes extraites d'un bloc), dépendances externes (jax/cv2) déjà distinctes du reste, risque de régression le plus facile à isoler (si un import casse, ça plante immédiatement et visiblement à l'activation de la touche `p`, pas ailleurs).
**Test** : touches `p` (prédiction) et `a` (accepter) sur quelques images réelles, comparer visuellement au comportement actuel.

### Étape 2 — Extraire la persistance JSON (cluster E)
Fonctions quasi pures déplacées vers un module dédié (ex. `tools/boxes_manual_json_store.py`) ou une petite classe `BoxAnnotationStore`. Risque minimal — peu de couplage à `self` à démêler.
**Test** : sauvegarde (`s`), édition de catégorie (`r`), nouvelle boîte (`n`) — vérifier que les JSON produits sont identiques (diff textuel) à ceux d'avant refactor sur les mêmes images.

### Étape 3 — Factoriser la conversion zoom→original
Une seule fonction pure `_zoomed_to_original(x1, y1, x2, y2, zoom_factor)`, remplace les 5 duplications. Mécanique, faible risque, facile à doter d'un test unitaire simple (pas besoin d'images réelles pour celui-ci).
**Test** : `f`/`l` (fill box), sauvegarde après zoom — vérifier que les coordonnées enregistrées sont identiques à avant.

### Étape 4 — Fusionner `image_save_folder`/`image_save_tmp`
Extraire un helper commun paramétré (déplacer le fichier ou non, avancer à l'image suivante ou non). Élimine la duplication tout en gardant les deux touches (`s`/`t`) avec leur comportement actuel intact.
**Test** : les deux touches séparément, plus le mode auto-traitement (`Ctrl+A`, qui appelle `image_save_folder` en interne) sur un petit lot.

### Étape 5 (optionnelle, plus tard, plus risquée) — Séparer rendu Canvas (C) et état des boîtes (D)
La séparation View/Model la plus profonde — touche le cœur de l'interaction (drag des poignées, highlight au survol). À ne faire que si les étapes 1-4 donnent satisfaction et que l'envie de continuer est là. Pas de plan détaillé pour l'instant — prématuré tant que les étapes précédentes n'ont pas validé la méthode.

### Nettoyage sans risque, indépendant du reste
Supprimer `_reload_current_image` (mort) — peut se faire à n'importe quel moment, y compris avant l'étape 1.

## Suivi

Chaque étape complétée doit être ajoutée ici avec la date et le résultat du test réel. Pas de story BMad formelle prévue pour l'instant (rythme dicté par la disponibilité d'Aymeric pour tester) — à réévaluer si le rythme s'accélère.

- 2026-07-24 : analyse complète + stratégie définie (Winston). Aucune étape encore commencée.
- 2026-07-24 : **Étape 1 implémentée (Amelia)**. Nouveau fichier `tools/boxes_manual_prediction_assistant.py` (171 lignes) — classe `PredictionAssistant`, porte `load_models()` (ex-`load_jax_models`) et `predict(original_image, true_boxes_x1y1x2y2)` (ex-`generate_and_show_predictions`, partie calcul). Design : la méthode `predict()` retourne `(result_image, avg_iou, predictions)` plutôt que d'écrire directement dans l'état tkinter — `PhotoViewer.generate_and_show_predictions` (réduite à ~25 lignes) fait le resize zoom et la mise à jour canvas/titre, seule responsabilité qui lui reste. `PhotoViewer.__init__` remplace les 7 attributs JAX individuels par `self.prediction_assistant = PredictionAssistant()`. `tools/boxes_process_manual_tkinter.py` passe de 1284 à 1156 lignes.
  - **Vérifié (pas juste écrit)** : `ast.parse` sur les deux fichiers (syntaxe OK) ; import réel de `boxes_manual_prediction_assistant` en isolation (`PredictionAssistant()` s'instancie, `load_models`/`predict` présents) ; import réel de `boxes_process_manual_tkinter` comme module (pas lancé en `__main__`, donc pas de GUI ouverte) — la chaîne d'imports se résout entièrement, `PhotoViewer.__init__` a la même signature qu'avant, `load_jax_models` a bien disparu de `PhotoViewer`, `toggle_predictions`/`generate_and_show_predictions`/`accept_predictions` toujours présentes ; grep de contrôle confirme qu'aucune référence aux 7 anciens attributs (`jax_models_loaded`, `det_predict_fn`, etc.) ne traîne plus dans le fichier.
  - **✅ Testé manuellement par Aymeric (2026-07-24)** : "tests effectués, pas de régression". Étape 1 définitivement close.
- 2026-07-24 : **Étape 2 implémentée (Amelia)**. Nouveau fichier `tools/boxes_manual_json_store.py` (87 lignes) — 3 fonctions pures (module, pas de classe : aucun état à porter) : `validate_and_fix_bbox_coordinates`, `ensure_json_consistency`, `save_json_with_consistency_check`, comportement identique à l'implémentation inline. `PhotoViewer.save_json_with_consistency_check` (8 sites d'appel dans le fichier, signature externe inchangée `self.save_json_with_consistency_check(file_path, data)`) devient un wrapper de 8 lignes qui récupère `current_image_name`/`image_width`/`image_height` depuis `self` et délègue à `json_store.save_json_with_consistency_check(...)`. Les 2 autres méthodes (`validate_and_fix_bbox_coordinates`, `ensure_json_consistency`) n'avaient aucun appelant externe — supprimées de `PhotoViewer`, pas juste déplacées avec un wrapper. `tools/boxes_process_manual_tkinter.py` passe de 1156 à 1089 lignes.
  - **Vérifié** : `ast.parse`, import complet du module (signatures/méthodes attendues présentes, les 2 méthodes supprimées confirmées absentes), grep de contrôle (aucune référence résiduelle). **Plus loin que l'étape 1** : test fonctionnel réel du nouveau module (pas juste import) — coordonnées hors limites correctement corrigées, champs JSON manquants correctement ajoutés, fichier écrit puis relu identique à ce qui était attendu. Test manuel GUI par Aymeric (touches `s`/`t`/`r`/`n` — tout ce qui passe par la sauvegarde JSON) encore à faire avant clôture définitive.
  - **✅ Testé manuellement par Aymeric (2026-07-24)** : "tests effectués, pas de régression". Étape 2 définitivement close.
- 2026-07-24 : **Étape 3 implémentée (Amelia)**. Adaptation mineure de la stratégie de Winston : plutôt qu'une fonction libre `_zoomed_to_original` dans un nouveau fichier, la méthode `zoomed_to_original(x1, y1, x2, y2)` a été ajoutée directement à `ImageManager` (qui possède déjà `zoom_factor`) — pas de nouveau fichier pour 9 lignes de logique, cohérent avec le principe "boring technology"/Rule of Three de Winston. Les 5 duplications (2 dans `fill_box_full_image`/`fill_box_full_width` produisant width/height, 3 dans `image_save_folder`/`image_save_tmp`/`generate_and_show_predictions` produisant x2/y2 directement) remplacées par des appels à `self.image_manager.zoomed_to_original(...)`. Note honnête : le fichier passe de 1089 à 1098 lignes (légère augmentation, pas réduction) — l'objectif de cette étape était l'élimination de la duplication (source unique de vérité pour la conversion), pas la réduction du nombre de lignes.
  - **Bug auto-détecté et corrigé avant vérification finale** : un `return bounding_boxes` résiduel (inatteignable, copié par erreur en ajoutant la méthode juste après `get_bounding_boxes`) s'est glissé dans `zoomed_to_original` — repéré en relisant le fichier après l'édition, corrigé immédiatement, avant tout test.
  - **Vérifié** : `ast.parse`, import complet, grep de contrôle (plus aucune occurrence du pattern dupliqué), **test numérique d'équivalence** — comparaison explicite ancienne formule (`(x2_zoomed - x1_zoomed) / zoom_factor`) vs nouvelle (`x2_original - x1_original` après conversion séparée) sur des valeurs non triviales : écart < 1e-12 (bruit de calcul flottant, pas une divergence réelle). Test manuel GUI par Aymeric (zoom + `f`/`l`/`s`/`t`/`p`) encore à faire.
  - **✅ Testé manuellement par Aymeric (2026-07-24)** : "tests effectués, pas de régression". Étape 3 définitivement close.
- 2026-07-24 : **Étape 4 implémentée (Amelia)**. Nouvelle méthode privée `PhotoViewer._export_bbox_annotations(target_dir, remove_source, handle_deleted)` — cœur partagé entre `image_save_folder` (déplace l'image + les JSON vers `save/`, retire les boîtes supprimées, navigue vers l'image suivante) et `image_save_tmp` (sauvegarde sur place, ne touche pas aux boîtes supprimées, ne navigue pas). Différence de comportement entre les deux méthodes explicitée par les 2 paramètres booléens plutôt que devinée en comparant le code dupliqué. Au passage, une variable locale `origin_image_path` inutilisée dans `image_save_tmp` (calculée mais jamais lue) a été retirée — vérifié sans effet de bord (`get_image_path()` est un simple `os.path.join`). `tools/boxes_process_manual_tkinter.py` passe de 1098 à 1084 lignes — vraie réduction cette fois, la duplication ~90% éliminée pour de bon.
  - **Vérifié, le plus poussé des 4 étapes** : au-delà de l'import, un **test fonctionnel avec vrais fichiers sur disque** (répertoire temporaire, vraie image PIL, vrais JSON) exerçant `_export_bbox_annotations` dans ses deux configurations : boîte mappée (coordonnées mises à jour) vs non mappée (inchangée) vs supprimée (retirée en mode dossier, préservée en mode tmp) — 6 assertions sur l'état réel du système de fichiers après exécution, pas une supposition. Un premier essai de ce test a lui-même révélé une erreur *dans le test* (pas dans le code : bbox de test dépassant les limites de l'image, correctement clampée par `validate_and_fix_bbox_coordinates` de l'Étape 2 — confirmation croisée que les étapes 2, 3 et 4 fonctionnent ensemble correctement).
  - **✅ Testé manuellement par Aymeric (2026-07-24)**, "complet sur plusieurs images" : "je valide". Étape 4 définitivement close.

## Bilan final (4/4 étapes détaillées terminées — chantier clos pour l'instant)

`tools/boxes_process_manual_tkinter.py` : 1284 → 1084 lignes. 3 nouveaux fichiers/éléments dédiés créés (`boxes_manual_prediction_assistant.py`, `boxes_manual_json_store.py`, `ImageManager.zoomed_to_original`). Zéro régression sur les 4 étapes, chacune testée réellement par Aymeric en plus des vérifications automatiques.

**Décision d'Aymeric (2026-07-24)** : s'arrêter là pour ce fichier. Le gain est déjà réel (séparation des responsabilités, duplication éliminée) et suffisant pour continuer à trier les images sereinement. L'Étape 5 (séparation Canvas/état des boîtes, la plus risquée, jamais détaillée par choix) reste en backlog — nécessiterait une nouvelle session de planification avec Winston avant toute implémentation, pas un simple "continuer" avec Amelia.

Document conservé comme référence si le sujet est repris un jour — pas à réanalyser depuis zéro.
