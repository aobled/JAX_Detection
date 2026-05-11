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

## 3. La Stratégie de Détection (Segmentation Sémantique U-Net)

**Acteur responsable :** `DetectionStrategy` (dans `task_strategies.py`) faisant appel à `loss_functions.py` (`compute_segmentation_loss`).

### La Mathématique : La Hybrid Loss (BCE + Dice)
L'architecture a évolué d'une grille YOLO vers un réseau de Segmentation Sémantique (U-Net). Le modèle génère une "Heatmap" (carte de chaleur) de la même taille que l'image d'entrée, où chaque pixel contient la probabilité d'appartenir à un avion.

Le défi majeur de cette approche est le **déséquilibre de classe extrême** : 99% de l'image est du ciel (classe 0), et 1% est un avion (classe 1). Si l'on utilisait une simple Erreur Quadratique Moyenne (MSE), le modèle pourrait obtenir un score artificiellement excellent en prédisant un écran entièrement noir. 

Pour contrer cela, la fonction de perte combine deux stratégies complémentaires :

1. **Weighted Binary Cross Entropy (BCE)**  
   Évalue pixel par pixel la certitude du réseau. Le fonctionnement naturel du BCE s'appuie sur la courbe de la fonction **Logarithme**. 
   - Plus le réseau se trompe en étant sûr de lui, plus l'erreur tend vers l'infini (ex: prédire `0.99` sur un pixel vide génère une erreur de `-log(0.01) = 4.6`). Le BCE pénalise l'arrogance !
   - **Pénalité des Faux Positifs (`false_positive_penalty`)** : Pour lutter contre les "fausses alarmes" (le réseau détecte un avion dans un nuage vide), nous avons introduit un multiplicateur mathématique. Si le réseau invente un avion, l'erreur logarithmique est multipliée par ce paramètre (par défaut `2.0`). Cela force le réseau à être conservateur et à préférer le doute plutôt que la fausse détection.

2. **Dice Loss (Sørensen–Dice coefficient)**  
   C'est la pièce maîtresse pour détecter les objets minuscules. Le coefficient de Dice calcule le taux de recouvrement global (similaire à l'IoU) entre la Heatmap prédite et le masque réel, *indépendamment de la taille de l'objet*. Mathématiquement : `2 * Intersection / Union`.
   Grâce au Dice Loss, rater un avion minuscule de 3 pixels pénalise le réseau avec la même intensité que rater un bombardier massif de 300 pixels. C'est ce qui force l'architecture à traquer les plus petites cibles.

La combinaison de ces deux pertes (`bce_weight=0.5` + `dice_weight=0.5`) garantit des Heatmaps d'une précision chirurgicale, résistantes au bruit de fond, et capables de capter des objets lointains.
