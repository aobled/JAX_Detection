# Stratégies de Reporting et Visualisation (Reporting)

L'évaluation post-entraînement est une brique critique pour garantir qu'un réseau de neurones a réellement compris son domaine, et pas simplement mémorisé les données. Dans notre framework `JAX Vision`, cette logique est concentrée dans `reporting.py` et s'appuie sur trois grands piliers.

---

## 1. La Matrice de Confusion (`confusion_matrix_from_pkl`)

Utilisée en tâche de Classification, elle confronte l'intégralité du dataset de Validation aux prédictions du réseau.

### Sérialisation et Certification
Plutôt que d'utiliser le modèle tel qu'il existe encore dans la Mémoire RAM à la fin de l'entraînement, la Stratégie force la méthode à recharger mécaniquement le fichier `best_model.pkl` fraîchement exporté sur le disque dur. 
**Pourquoi ?** C'est une assurance Qualité (QA). Si la matrice de confusion sort un score parfait depuis le `.pkl`, cela garantit mathématiquement que le fichier n'est pas corrompu et qu'il est prêt à être envoyé en production (sur un Raspberry Pi, un serveur distant, ou un Jupyter Notebook) de manière totalement autonome.

### Ingénierie
La méthode ouvre le `.pkl`, farfouille à l'intérieur pour trouver sa carte d'identité (le dictionnaire `config`) et se recrée elle-même à partir de rien :
- Elle reconfigure les dimensions du modèle.
- Elle recrée le nombre de classes dynamiquement (fini le "2" codé en dur).
- Elle applique l'inférence sur tout le Dataset pour traquer les faux positifs.

---

## 2. Inférence en Masse (`show_predictions_from_dir`)

C'est l'outil de diagnostic visuel par excellence. Il prend un répertoire physique de votre disque dur contenant des milliers de photos et les jette en pâture au modèle.

### Fonctionnalités Clés
- **Pré-traitement Isométrique :** La fonction reproduit exactement les transformées (Resize, Grayscale, Normalisation Statistique `Mean/Std`) que le modèle a connues pendant son entraînement via `data_management.py`.
- **Mode Enquêteur (`Target Class`) :** L'une des fonctionnalités les plus puissantes. Si on lui donne l'instruction `target_class="F-22"`, l'algorithme va balayer les milliers d'images, ignorer toutes celles où le modèle a eu juste, et **isoler uniquement les erreurs**. 
- Il génère ensuite des planches (grids) visuelles de ces erreurs avec le taux de confiance en pourcentage, permettant de découvrir des biais du dataset (ex: le modèle confond toujours le F-22 avec le F-35 sous tel angle).

---

## 3. L'Analyseur de Trajectoire (`TrainingVisualizer`)

Cette classe capture l'historique complet (Loss, Validation, Learning Rate) et dessine la dynamique d'apprentissage. Actuellement, elle utilise un triple axe pour afficher les interdépendances temporelles (si le LR chute, est-ce que la Loss suit ?).

### La Dualité : Classification vs Détection

Ce graphe met en évidence la flexibilité forcée de notre architecture polymorphique :

1. **En Classification**
   - L'Axe Y2 représente la véritable **Accuracy** (de 0 à 100%).
   - L'écart rouge/bleu entre Train et Val est hachuré, il dessine visuellement la **"Zone d'Overfitting"**.

2. **En Détection (Le Hack du Checkpoint)**
   - Il n'y a pas d'Accuracy évidente en détection à calculer en *nanosecondes* sur TPU (calculer de l'IoU de multi-boîtes englobantes ferait s'effondrer la vitesse). 
   - Par conséquent, la `DetectionStrategy` utilise une ruse mathématique : elle renvoie **`l'inverse de la loss (-loss)`** en se faisant passer pour la métrique Accuracy.
   - Le `Trainer` et le `CheckpointManager` se font tromper joyeusement : ils cherchent toujours à "Maximaliser l'Accuracy", en sauvegardant le modèle dès que la valeur grimpe (ex: on passe de `-2.3` à `-1.8`, le modèle est sauvé !).
   - Sur le *TrainingVisualizer*, l'axe "Accuracy %" affichera donc pour la détection **une courbe négative qui monte vers 0**, servant de *Score Relatif* plutôt que de véritable pourcentage.
