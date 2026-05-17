import os
import copy
import pickle
import random
import hashlib
import numpy as np


class GeneticAlgorithm:
    """
    Algorithme génétique spécialisé pour l'évolution de query_tokens
    d'un bottleneck Transformer dans un modèle JAX/Flax.

    Structure attendue des tokens :
        (1, num_tokens, embedding_dim)

    Exemple :
        (1, 16, 256)

    Philosophie :
    - Mutations douces
    - Conservation de la géométrie latente
    - Crossovers cohérents
    - Respect des normes des tokens
    """

    def __init__(
        self,
        token_shape=(1, 16, 256),
        mutation_sigma=0.01,
        mutation_rate=0.15,
        crossover_rate=0.5,
        keep_norm=True,
        random_seed=42,
    ):

        self.token_shape = token_shape

        self.num_tokens = token_shape[1]
        self.embedding_dim = token_shape[2]

        self.mutation_sigma = mutation_sigma
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.keep_norm = keep_norm

        np.random.seed(random_seed)
        random.seed(random_seed)

    # ============================================================
    # UTILITAIRES
    # ============================================================

    def get_unique_filename(self, tokens):
        """
        Génère un hash SHA256 unique basé sur les valeurs des tokens.
        """
        tokens_bytes = tokens.tobytes()
        hash_object = hashlib.sha256(tokens_bytes)
        return f"tokens_{hash_object.hexdigest()}"

    def save_tokens(self, tokens, output_dir="./population"):
        """
        Sauvegarde un individu .npy
        """

        os.makedirs(output_dir, exist_ok=True)

        filename = self.get_unique_filename(tokens) + ".npy"
        filepath = os.path.join(output_dir, filename)

        np.save(filepath, tokens)

        return filepath

    def load_tokens(self, filepath):
        """
        Charge un individu .npy
        """
        return np.load(filepath)

    # ============================================================
    # INITIALISATION
    # ============================================================

    def create_random_population(
        self,
        origin_tokens,
        population_size=10,
        output_dir="./population"
    ):
        """
        Génère une population initiale proche du modèle d'origine.
        """

        population = []

        for _ in range(population_size):

            individual = copy.deepcopy(origin_tokens)

            individual = self.mutate(
                individual,
                sigma=self.mutation_sigma
            )

            filepath = self.save_tokens(individual, output_dir)

            population.append(filepath)

        return population

    # ============================================================
    # MUTATION
    # ============================================================

    def mutate(
        self,
        tokens,
        sigma=None,
        mutation_rate=None,
    ):
        """
        Mutation douce des tokens.

        IMPORTANT :
        - mutation par token entier
        - pas coordonnée isolée
        - conservation possible des normes
        """

        if sigma is None:
            sigma = self.mutation_sigma

        if mutation_rate is None:
            mutation_rate = self.mutation_rate

        tokens_mut = copy.deepcopy(tokens)

        for token_idx in range(self.num_tokens):

            if np.random.rand() < mutation_rate:

                original_norm = np.linalg.norm(
                    tokens_mut[0, token_idx]
                )

                noise = np.random.normal(
                    0,
                    sigma,
                    size=(self.embedding_dim,)
                ).astype(np.float32)

                # Mutation douce additive
                tokens_mut[0, token_idx] += noise

                # Conservation de la norme originale
                if self.keep_norm:

                    new_norm = np.linalg.norm(
                        tokens_mut[0, token_idx]
                    )

                    if new_norm > 1e-8:

                        tokens_mut[0, token_idx] *= (
                            original_norm / new_norm
                        )

        return tokens_mut

    # ============================================================
    # CROSSOVER
    # ============================================================

    def crossover(
        self,
        parent_a,
        parent_b,
        method="token"
    ):
        """
        Crossover cohérent entre deux parents.

        Méthodes :
        - token : échange de tokens complets
        - blend : interpolation douce
        """

        child = copy.deepcopy(parent_a)

        # --------------------------------------------------------
        # TOKEN CROSSOVER
        # --------------------------------------------------------

        if method == "token":

            for token_idx in range(self.num_tokens):

                if np.random.rand() < self.crossover_rate:

                    child[0, token_idx] = parent_b[0, token_idx]

        # --------------------------------------------------------
        # BLEND CROSSOVER
        # --------------------------------------------------------

        elif method == "blend":

            alpha = np.random.uniform(0.25, 0.75)

            child = (
                alpha * parent_a
                + (1.0 - alpha) * parent_b
            ).astype(np.float32)

        else:
            raise ValueError(f"Méthode inconnue : {method}")

        return child

    # ============================================================
    # SELECTION
    # ============================================================

    def tournament_selection(
        self,
        scored_population,
        tournament_size=3
    ):
        """
        Sélection par tournoi.

        scored_population :
            [
                (filepath, score),
                ...
            ]

        Plus le score est faible, meilleur est l'individu.
        """

        selected = random.sample(
            scored_population,
            tournament_size
        )

        selected = sorted(selected, key=lambda x: x[1])

        return selected[0][0]

    # ============================================================
    # EVOLUTION
    # ============================================================

    def evolve_population(
        self,
        scored_population,
        offspring_size=10,
        crossover_method="token",
        output_dir="./population_next"
    ):
        """
        Génère une nouvelle population.
        """

        os.makedirs(output_dir, exist_ok=True)

        new_population = []

        while len(new_population) < offspring_size:

            # --------------------------------------------
            # Sélection des parents
            # --------------------------------------------

            parent_a_path = self.tournament_selection(
                scored_population
            )

            parent_b_path = self.tournament_selection(
                scored_population
            )

            parent_a = self.load_tokens(parent_a_path)
            parent_b = self.load_tokens(parent_b_path)

            # --------------------------------------------
            # Crossover
            # --------------------------------------------

            child = self.crossover(
                parent_a,
                parent_b,
                method=crossover_method
            )

            # --------------------------------------------
            # Mutation
            # --------------------------------------------

            child = self.mutate(child)

            # --------------------------------------------
            # Sauvegarde
            # --------------------------------------------

            child_path = self.save_tokens(
                child,
                output_dir
            )

            new_population.append(child_path)

        return new_population

    # ============================================================
    # CHECKPOINT
    # ============================================================

    def inject_tokens_into_checkpoint(
        self,
        checkpoint_path,
        token_file,
        output_checkpoint
    ):
        """
        Injecte des query_tokens dans un checkpoint .pkl
        """

        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        tokens = np.load(token_file)

        checkpoint['params']['token_bottleneck']['query_tokens'] = tokens

        with open(output_checkpoint, "wb") as f:
            pickle.dump(checkpoint, f)

    # ============================================================
    # ANALYSE
    # ============================================================

    def token_norms(self, tokens):
        """
        Retourne la norme L2 de chaque token.
        """

        return np.linalg.norm(tokens[0], axis=-1)

    def compare_tokens(self, tokens_a, tokens_b):
        """
        Distance moyenne entre deux ensembles de tokens.
        """

        diff = tokens_a - tokens_b

        return np.mean(np.abs(diff))

    def diversity_score(self, population_files):
        """
        Score simple de diversité de population.
        """

        vectors = []

        for f in population_files:

            t = np.load(f)

            vectors.append(t.flatten())

        vectors = np.array(vectors)

        return np.std(vectors)


# ================================================================
# EXEMPLE D'UTILISATION
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Charger tokens originaux
    # ------------------------------------------------------------

    with open("../best_model_detection.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    origin_tokens = checkpoint['params']['token_bottleneck']['query_tokens']

    print(origin_tokens.shape)

    # ------------------------------------------------------------
    # Initialiser GA
    # ------------------------------------------------------------

    ga = GeneticAlgorithm(
        token_shape=origin_tokens.shape,
        mutation_sigma=0.01,
        mutation_rate=0.25,
        crossover_rate=0.5,
        keep_norm=True
    )

    # ------------------------------------------------------------
    # Population initiale
    # ------------------------------------------------------------

    population = ga.create_random_population(
        origin_tokens,
        population_size=100,
        output_dir="./population_gen0"
    )

    print("\nPopulation créée :")
    for p in population:
        print(p)

    # ------------------------------------------------------------
    # Exemple crossover
    # ------------------------------------------------------------

    parent_a = np.load(population[0])
    parent_b = np.load(population[1])

    child = ga.crossover(parent_a, parent_b, method="token")
    child = ga.mutate(child, sigma=0.005)
    child_path = ga.save_tokens(child, output_dir="./children")

    print("\nChild généré :")
    print(child_path)

    # ------------------------------------------------------------
    # Injection dans checkpoint
    # ------------------------------------------------------------

    ga.inject_tokens_into_checkpoint(
        checkpoint_path="../best_model_detection.pkl",
        token_file=child_path,
        output_checkpoint="../best_model_detection_child.pkl"
    )

    print("\nCheckpoint enfant généré.")

