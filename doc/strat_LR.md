# Stratégie de Learning Rate (LR) et Planification (Scheduling) 📉

Dans notre architecture de Trainer JAX/Flax, le taux d'apprentissage (Learning Rate) n'est **pas constant** (fixe). Nous utilisons un mécanisme dynamique industriellement prouvé appelé **Warmup Cosine Decay Schedule**.

Ce document explique pourquoi et comment il fonctionne, et pourquoi la classification et la détection ont des comportements si différents.

---

## 1. Le Fonctionnement : "Warmup Cosine Decay"

Le planificateur (`optax.warmup_cosine_decay_schedule`) pilote la valeur du LR en trois phases. Une courbe visuelle prendrait la forme d'une cloche tronquée :

### Phase A : Le Préchauffage (Warmup)
* **Mécanisme :** Le LR commence à **0.0** et augmente linéairement jusqu'à atteindre sa valeur maximale (la "peak value", définie dans `dataset_configs.py` sous `tpu: { learning_rate: X }`).
* **Durée :** Pilotée par la variable `warmup_steps`.
* **Pourquoi ?** Si vous lâchez un modèle inexpérimenté dans un paysage d'optimisation avec un grand LR, l'explosion des gradients détruira les poids initiaux. Le préchauffage le "réveille" doucement pour lui donner une bonne direction avant d'accélérer.

### Phase B : La Chute Cosinusoïdale (Cosine Decay)
* **Mécanisme :** À partir du moment où le LR a atteint son sommet, il commence à baisser doucement (plateau), accélère sa chute en milieu de course, puis s'adoucit à l'approche de zéro.
* **Durée :** Pilotée par la variable `decay_steps`.
* **Pourquoi ?** À mesure que le modèle se rapproche de la cible idéale de l'Accuracy/Loss, ses "pas" doivent se raccourcir pour ne pas sauter par-dessus l'optimum. S'il gardait son LR de pointe, la loss se mettrait à "osciller" sans jamais converger au milieu du cratère.

### Phase C : Fin de course (End Value)
* **Mécanisme :** Une fois le `decay` terminé, le LR stagne à sa valeur finale quasi nulle (`1e-6`). Le modèle effectue ce qu'on appelle du "fine-tuning" millimétrique.

---

## 2. L'Analyse du Constat (Classification vs. Détection)

Vous avez noté une chute asymétrique du LR selon la tâche. Voici pourquoi vous avez raison ! Le paramétrage interne est radicalement différent d'une modalité à l'autre :

### Cas A : La Classification (Chute Aiguë)
Dans `dataset_configs.py`, sous `FIGHTERJET_CLASSIFICATION`, il n'y a **aucune** mention explicite de `warmup_steps` ou `decay_steps`. Le modèle a hérité des valeurs par défaut injectées dans le ventre de `trainer.py` :
* `warmup_steps = 1200`
* `decay_steps = 6000`

Un Chunk de classification comprend généralement autour de 27,000 images, soit ~52 pas (steps) d'entrainement matérielle après gradients accumulés. **Le modèle épuise donc ses 6000 steps de chute en près de 12 à 15 Epochs**.
**Résultat :** Vous voyez le LR monter vite, puis chuter rapidement et stagner à zéro. C'est idéal car la classification de silhouette s'apprend généralement vite.

### Cas B : La Détection (Chute Extrêmement Lente / "Plateau")
Dans `dataset_configs.py`, sous `FIGHTERJET_DETECTION`, des hyperparamètres spécifiques ont été déclarées pour contrer l'optimisation YOLO-like :
* `warmup_steps = 2000`
* `decay_steps = 90000` 🔥

La Détection a reçu une consigne de prendre près de **100,000 itérations** pour terminer son cycle ! Au rythme actuel du nombre d'images de détection (surtout lors de l'apprentissage des ancres géométriques multi-échelles), le decay est "étiré" sur l'équivalent de 200 à 300 Epochs.
**Résultat :** Face aux 30 ou 40 epochs que vous lancez, votre courbe de LR semble ne "jamais" descendre sur un graphique linéaire ! Le modèle maintient sa dynamique de recherche au sommet le plus longtemps possible, forçant l'ajustement sans fin de la MSE Loss des Bounding Boxes.

---

## 3. Recommandations Poursuites

1. **Uniformité ou Spécialisation ?**
   C'est normal d'avoir ces comportements asymétrique. La détection (localisation) est exponentiellement plus difficile que la classification. Un LR qui chute trop tôt en détection gèle les boîtes avant de les lisser géométriquement. Je recommande de **garder ces différences**.
   
2. **Déclaration Formelle en Classification**
   Plutôt que de laisser `trainer.py` appliquer ses règles par défaut caché (1200/6000), il est de bonne pratique de reporter ces valeurs par défaut manuellement dans les configs `FIGHTERJET_CLASSIFICATION`, afin d'avoir vos paramètres à l'œil et de pouvoir les tuner si le modèle stagne.

3. **Adaptation au Temps T**
   Si votre entraînement est programmé pour moins d'Epochs ("Je veux entraîner sur 30 Epochs"), votre courbe `decay_steps` doit logiquement s'accorder avec le math de : `(dataset_size / batch_size) * 30 epochs`.
   Si le `decay` est de `90,000` pour un entrainement coupé à `20,000` steps, le modèle n'aura jamais l'occasion faire son "Fine-Tuning millimétrique" de validation avec un LR réduit.
