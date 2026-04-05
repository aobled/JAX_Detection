# Stratégies d'Évaluation de l'Erreur (Loss Functions)

Au cœur du Framework JAX Vision se trouve un mécanisme polymorphique redoutablement efficace. Plutôt que de truffer la boucle d'entraînement de conditions de type `if classification` puis `if detection`, le `Trainer` (`trainer.py`) est devenu totalement aveugle à la tâche.

Il se contente de demander dynamiquement le calcul de l'erreur à la classe *Stratégie* ("Strategy") qui lui a été injectée au démarrage depuis `main.py`.

---

## 1. Comment et où est appelée la Loss ?

Toutes les pertes sont calculées à l'intérieur de la *Closure* (la fonction imbriquée) encerclée par le décorateur `@jax.jit` (`train_step` et `eval_step` dans `trainer.py`). 

```python
# Cœur du trainer.py (Totalement agnostique de la tâche)
def loss_fn(params):
    outputs, new_batch_stats = self.state.apply_fn(vars, images, ...)
    # La magie polymorphique :
    loss = self.strategy.compute_loss(outputs, targets, use_onehot_labels=use_onehot)
    return loss, (outputs, new_batch_stats)
```
Ce fonctionnement asynchrone permet au **calcul mathématique d'être compilé de manière native sur le TPU (ou GPU)** sans aucune baisse de performance liée aux couches logiques de Python.

---

## 2. La Stratégie de Classification

**Acteur responsable :** `ClassificationStrategy` (dans `task_strategies.py`).

### La Mathématique : L'Entropie Croisée (Cross Entropy)
Le but de ce modèle est de définir une probabilité à 100% sur un seul avion.
La stratégie utilise les opérations ultra-optimisées de la librairie **Optax** (`optax.softmax_cross_entropy` ou `optax.softmax_cross_entropy_with_integer_labels`). 

### L'intelligence Adaptative
Pendant cette extraction, la méthode identifie de manière autonome comment comparer les sorties :
- **Mode Dur (Integer Labels)** : Par défaut, si l'image est un "Rafale" (Classe 24), la cible est le chiffre `24`.
- **Mode Lisse (One-Hot Probabilities)** : Si vous avez activé le *Mixup* (`alpha=0.2`) ou le *Label Smoothing* (`0.1`) depuis la configuration, la cible n'est plus un seul chiffre entier mais un spectre (ex: `[0.85% Rafale, 0.15% Nuage]`). La stratégie switch automatiquement sur la fonction Cross-Entropy standard d'Optax pour comparer ces lois de probabilités pures.

---

## 3. La Stratégie de Détection

**Acteur responsable :** `DetectionStrategy` (dans `task_strategies.py`) faisant appel à `loss_functions.py`.

### La Mathématique : La Grille YOLO (You Only Look Once)
C'est le sommet mathématique de ce framework. Le modèle ne sort pas un simple chiffre, mais une grille en 3 dimensions (une map spatiale locale, type `7x7x5` pour 49 cases et 5 variables : `x, y, w, h, confiance`).

La méthode `compute_grid_loss(outputs, targets)` décompose l'erreur du réseau en **trois gradients distincts** pour le punir intelligemment sur 3 axes :

1. **La Vérité Géométrique (`loss_coord`)**  
   Basée sur l'Erreur Quadratique Moyenne (MSE) pour l'étalonnage. Elle punit un Centre `x,y` de la boîte si ce dernier est décalé par rapport à l'avion cible. Pour la hauteur et la largeur de la boîte `w,h`, l'équation emploie intentionnellement des **racines carrées**. *Pourquoi ?* Car une erreur de 20 pixels sur un immense porteur A400M n'a pas d'importance, mais une erreur de 20 pixels sur un minuscule chasseur Rafale au loin détruit la boîte englobante.
2. **La Vérité Absolue (`loss_obj`)**  
   Pour la cellule de la grille (parmi les 49 cases) qui a la chance de "posséder" le centre de gravité de l'avion ciblé, le réseau est puni si son score de certitude ("je vois un objet !") est inférieur à 100%. 
3. **Le Ciel Vide (`loss_noobj`)**  
   La plus compliquée statistiquement à gérer. Le ciel est vaste. Pour les 48 cases qui ne contiennent rien, le système est intensément corrigé (via le coefficient pondérateur empirique standardisé `lambda_noobj = 0.5`) si son attention grimpe > 0%.

---

La flexibilité de la librairie et l'introduction du design pattern *Stratégie* fait de ce fichier un point de départ fantastique. S'il fallait rajouter un troisième point d'entrée pour la Segmentation d'instance U-NET, il suffirait de greffer une nouvelle classe de Stratégie (`SegmentationStrategy`) utilisant la perte *Dice Loss* ou *BCE*, et le Trainer la compilerait sur le champ !
