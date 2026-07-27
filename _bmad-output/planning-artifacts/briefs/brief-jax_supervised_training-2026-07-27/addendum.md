---
title: "Addendum — Brief moteur d'échecs"
created: 2026-07-27
updated: 2026-07-27
---

# Addendum : approfondissements et pistes rejetées

Ce document conserve le détail qui n'a pas sa place dans le brief lean, mais qui sera utile à la session d'architecture (Winston) et au PRD.

## État de l'art détaillé (recherche du 2026-07-27)

**DeepMind, "Grandmaster-Level Chess Without Search"** (arXiv:2402.04494, février 2024, Ruoss et al.) — architecture : transformer décodeur pur, **sans CNN**, 270M paramètres (16 couches, 8 têtes, embeddings 1024-dim ; variantes 9M également testées). Signal d'entraînement : **pas le résultat de parties humaines** — supervision sur des annotations **Stockfish 16** (valeurs d'action, valeurs d'état, et imitation comportementale des coups choisis par Stockfish), via le dataset `ChessBench` publié (10M parties, ~15 milliards de points). Résultat : 2895 Elo blitz Lichess face à des humains, 93.5% de précision sur puzzles, sans recherche à l'inférence. À retenir : prouve qu'un backbone CNN n'est pas nécessaire pour atteindre un niveau GM, mais le design de label est une imitation de moteur, pas d'humain — moins comparable à notre cas que Maia. Sources : [google-deepmind/searchless_chess](https://github.com/google-deepmind/searchless_chess), [ar5iv 2402.04494](https://ar5iv.labs.arxiv.org/html/2402.04494).

**Maia Chess (CSSLab, Toronto)** — le précédent le plus proche de notre design. Maia original : CNN résiduel pur, sans attention, entraîné uniquement sur des parties humaines Lichess (par tranche d'Elo, ~1100-1900), objectif explicite de prédire le coup *humain* plutôt que le coup objectivement meilleur, sans recherche. Évolution du projet : **Maia-2** unifie les niveaux d'Elo dans un seul modèle ; **Maia-3** bascule vers un transformer ("Chessformer") avec tokens par case + **Geometric Attention Bias** (encodage positionnel conscient de la géométrie des échecs). Le projet le plus comparable au nôtre a donc lui-même migré vers l'attention avec le temps. Sources : [maiachess.com](https://www.maiachess.com/), [CSSLab/maia-chess](https://github.com/CSSLab/maia-chess), [CSSLab/maia3](https://github.com/CSSLab/maia3), [Maia-2 arXiv](https://arxiv.org/pdf/2409.20553).

**AlphaVile** (Czech et al., TU Darmstadt, arXiv:2304.14918) — hybride CNN+ViT dans une boucle AlphaZero. Constat clé : les blocs ViT purs étaient trop lents pour apporter un bénéfice, et les plus gros gains Elo (+180 vs AlphaZero) venaient de changements de représentation d'entrée et de la loss de la value head, pas de l'attention elle-même en tant que mécanisme de couverture du receptive field. Aucune source trouvée ne justifie l'attention par la couverture de receptive field sur 8×8 — cette justification initiale (intuition de départ) est un homme de paille ; la justification qui tient est le raisonnement relationnel/géométrique entre pièces précises. Source : [Representation Matters: ViT vs Chess (ar5iv)](https://ar5iv.labs.arxiv.org/html/2304.14918), [TU Darmstadt](https://www.informatik.tu-darmstadt.de/fb20/ueber_uns_details_308928.en.jsp).

**Pièges connus de l'imitation de parties humaines** : bruit des coups (blunders, incohérence) ; mélange blitz (instinctif) / classique (calculatoire) reflétant des processus cognitifs différents et ne devant pas être pondéré naïvement ensemble ; prévisibilité non monotone selon le niveau (chez Maia, les joueurs de niveau intermédiaire sont les plus prévisibles — faibles et forts le sont moins) ; déséquilibre de classes par ouverture/ECO. Un dernier piège concerne la value head : la supervision par résultat brut de partie est un credit assignment bruité sans recherche. DeepMind contourne ce problème en utilisant des évaluations Stockfish plutôt que le résultat de partie — une option à reconsidérer si la value head s'avère trop bruitée en pratique, à condition d'accepter de réintroduire un moteur externe uniquement pour la génération de labels, ce qu'Aymeric a explicitement écarté pour ce cycle.

## Design de label rejeté : "coups du gagnant = True"

Hypothèse de départ d'Aymeric : filtrer les coups d'entraînement de la policy head pour ne garder que ceux joués par le camp qui remporte finalement la partie (behavioral cloning côté gagnant uniquement), en écartant les nulles.

**Pourquoi rejeté** : ce filtre ne fait pas ce qu'il promet. Une partie gagnée peut contenir une boulette du futur gagnant en début de partie — elle serait quand même incluse comme "vraie" par la règle, puisque c'est lui qui gagne au final. Symétriquement, la grande majorité des coups du perdant, en dehors de la séquence qui a réellement fait perdre la partie, sont des coups de grand maître tout à fait corrects. Le filtre écarte donc pour rien environ la moitié des données saines (les bons coups du perdant), tout en conservant les boulettes du gagnant qu'il était censé exclure.

**Design retenu à la place** (voir brief, section Dataset) : construction par position plutôt que par partie — la policy head imite tous les coups joués des deux côtés sans filtrage par résultat ; seule la value head porte le résultat de partie (+1/0/-1 côté joueur au trait), nulles incluses avec value=0. Directement inspiré de la pratique de Maia (aucun filtrage par résultat sur la policy).

## Note sur `./chess/chess_game.py`

Fichier existant (240 lignes), plateau Tkinter pour jouer manuellement à deux (aucune IA ne joue de coup). Utilise `python-chess` + `cairosvg`/Pillow pour le rendu, et optionnellement Stockfish (via binaire local `./chess/stockfish` ou `PATH`) pour une barre d'avantage — avec repli automatique sur une évaluation matérielle simple (somme des valeurs de pièces) si Stockfish est introuvable. Aymeric n'ajoute volontairement pas de binaire Stockfish à ce dépôt, ce qui active ce repli par défaut. Base de départ pressentie pour un futur banc de test d'intégration du modèle entraîné (hors scope de cette epic) : il faudrait ajouter une logique de sélection de coup par le modèle, qui n'existe pas aujourd'hui dans ce fichier.
