import os
import csv
import json
import copy
import pickle
import random
import hashlib
import subprocess
from datetime import datetime

import numpy as np
import tqdm


class GeneticAlgorithm:
    """
    Algorithme génétique spécialisé pour l'évolution
    des query_tokens d'un Transformer bottleneck.

    Philosophie :
    -------------------------
    - mutation douce
    - crossover cohérent
    - conservation de géométrie latente
    - évaluation indépendante par token file
    - cache des évaluations
    - historique complet des métriques
    """

    def __init__(
        self,
        checkpoint_path="../best_model_detection.pkl",
        evaluation_script="./loss_detection_only_directory.py",
        population_dir="./population",
        evaluation_dir="./evaluations",

        mutation_sigma=0.01,
        mutation_rate=0.25,
        crossover_rate=0.5,

        keep_norm=True,
        elite_ratio=0.2,

        random_seed=42
    ):

        self.checkpoint_path = checkpoint_path
        self.evaluation_script = evaluation_script

        self.population_dir = population_dir
        self.evaluation_dir = evaluation_dir

        self.mutation_sigma = mutation_sigma
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.keep_norm = keep_norm
        self.elite_ratio = elite_ratio

        np.random.seed(random_seed)
        random.seed(random_seed)

        os.makedirs(self.population_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)

        # Cache global des évaluations
        self.global_cache = self._load_global_cache()

        # =========================================================
        # Chargement modèle origine
        # =========================================================
        with open(self.checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.origin_tokens = checkpoint['params']['token_bottleneck']['query_tokens']

        self.token_shape = self.origin_tokens.shape
        self.num_tokens = self.token_shape[1]
        self.embedding_dim = self.token_shape[2]

        print("\n🧬 GeneticAlgorithm initialisé")
        print(f"   Tokens shape : {self.token_shape}")

    # =========================================================
    # PARSING METRICS (JSON uniquement)
    # =========================================================
    def parse_metrics(self, text):
        """
        Parse les métriques depuis la sortie JSON de loss_detection_only_directory.py.
        Retourne un dictionnaire avec les métriques ou None en cas d'erreur.
        """
        try:
            # Le script retourne UNIQUEMENT un JSON (dernière ligne)
            lines = text.strip().split('\n')
            json_line = None
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{'):
                    json_line = line
                    break
            
            if json_line:
                metrics = json.loads(json_line)
                # Convertir les valeurs numériques si nécessaire (au cas où)
                for key, value in metrics.items():
                    if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        metrics[key] = float(value) if '.' in value else int(value)
                return metrics
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"❌ Erreur parsing JSON: {e}")
            return None
        
        return None

    # =========================================================
    # HASH / FICHIERS
    # =========================================================

    def get_hash(self, tokens):
        tokens_bytes = tokens.tobytes()
        hash_object = hashlib.sha256(tokens_bytes)
        return hash_object.hexdigest()

    def get_token_filename(self, tokens):
        return f"tokens_{self.get_hash(tokens)}.npy"

    def save_tokens(self, tokens, generation):
        gen_dir = os.path.join(
            self.population_dir,
            f"gen_{generation:04d}"
        )
        os.makedirs(gen_dir, exist_ok=True)
        filename = self.get_token_filename(tokens)
        filepath = os.path.join(gen_dir, filename)
        np.save(filepath, tokens)
        return filepath

    # =========================================================
    # MUTATION
    # =========================================================

    def mutate(self, tokens):
        tokens_mut = copy.deepcopy(tokens)
        for token_idx in range(self.num_tokens):
            if np.random.rand() < self.mutation_rate:
                original_norm = np.linalg.norm(tokens_mut[0, token_idx])
                noise = np.random.normal(
                    0,
                    self.mutation_sigma,
                    size=(self.embedding_dim,)
                ).astype(np.float32)
                tokens_mut[0, token_idx] += noise
                # conservation norme
                if self.keep_norm:
                    new_norm = np.linalg.norm(tokens_mut[0, token_idx])
                    if new_norm > 1e-8:
                        tokens_mut[0, token_idx] *= (original_norm / new_norm)
        return tokens_mut

    # =========================================================
    # CROSSOVER
    # =========================================================

    def crossover(self, parent_a, parent_b):
        child = copy.deepcopy(parent_a)
        for token_idx in range(self.num_tokens):
            if np.random.rand() < self.crossover_rate:
                child[0, token_idx] = parent_b[0, token_idx]
        return child

    # =========================================================
    # POPULATION INITIALE
    # =========================================================

    def create_initial_population(self, population_size=20):
        population = []
        print("\n🌱 Création population initiale...")
        # individu origine
        origin_path = self.save_tokens(self.origin_tokens, generation=0)
        population.append(origin_path)
        # mutants
        for _ in range(population_size - 1):
            individual = self.mutate(self.origin_tokens)
            filepath = self.save_tokens(individual, generation=0)
            population.append(filepath)
        return population

    # =========================================================
    # CHECKPOINT
    # =========================================================

    def inject_tokens_into_checkpoint(self, token_file, output_checkpoint):
        with open(self.checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        tokens = np.load(token_file)
        checkpoint['params']['token_bottleneck']['query_tokens'] = tokens
        with open(output_checkpoint, "wb") as f:
            pickle.dump(checkpoint, f)

    # =========================================================
    # CACHE GLOBAL
    # =========================================================

    def _load_global_cache(self):
        cache = {}
        for file in os.listdir(self.evaluation_dir):
            if file.startswith("generation_") and file.endswith(".csv"):
                csv_path = os.path.join(self.evaluation_dir, file)
                with open(csv_path, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for k, v in row.items():
                            try:
                                row[k] = float(v) if '.' in v else int(v)
                            except ValueError:
                                pass
                        cache[row["hash"]] = row
        print(f"📦 Cache global chargé : {len(cache)} individus uniques.")
        return cache

    # =========================================================
    # EVALUATION
    # =========================================================

    def already_evaluated(self, token_hash, evaluation_csv):
        if not os.path.exists(evaluation_csv):
            return False
        with open(evaluation_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["hash"] == token_hash:
                    return True
        return False

    def evaluate_individual(self, token_file, generation):
        token_hash = os.path.basename(token_file).replace(".npy", "")
        evaluation_csv = os.path.join(
            self.evaluation_dir,
            f"generation_{generation:04d}.csv"
        )

        # =====================================================
        # CACHE
        # =====================================================
        if self.already_evaluated(token_hash, evaluation_csv):
            print(f"⏩ Déjà dans le CSV courant : {token_hash}")
            return None

        if token_hash in getattr(self, 'global_cache', {}):
            print(f"♻️  Récupération depuis le cache global : {token_hash}")
            metrics = copy.deepcopy(self.global_cache[token_hash])
            metrics["generation"] = generation
            self.append_evaluation_csv(evaluation_csv, metrics)
            return metrics

        # =====================================================
        # Création checkpoint temporaire
        # =====================================================
        tmp_checkpoint = os.path.join(self.evaluation_dir, f"{token_hash}.pkl")
        self.inject_tokens_into_checkpoint(token_file, tmp_checkpoint)

        # =====================================================
        # Appel script externe
        # =====================================================
        print(f"\n🧪 Evaluation : {token_hash}")
        cmd = [
            "python",
            self.evaluation_script,
            "--checkpoint",
            tmp_checkpoint
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        error = result.stderr

        # =====================================================
        # PARSING METRIQUES JSON
        # =====================================================
        metrics = self.parse_metrics(output)
        if metrics is None:
            print("\n❌ ERREUR PARSING JSON")
            print("STDOUT:", output)
            print("STDERR:", error)
            # Nettoyage même en cas d'erreur
            if os.path.exists(tmp_checkpoint):
                os.remove(tmp_checkpoint)
            return None
        
        metrics["hash"] = token_hash
        metrics["token_file"] = token_file
        metrics["generation"] = generation
        metrics["timestamp"] = str(datetime.now())

        # =====================================================
        # FITNESS (utiliser avg_loss comme métrique principale)
        # =====================================================
        if "avg_loss" in metrics:
            metrics["fitness"] = metrics["avg_loss"]
        elif "total_loss" in metrics:
            metrics["fitness"] = metrics["total_loss"]
        else:
            print("\n❌ Aucune métrique de loss trouvée dans le JSON")
            print("Clés disponibles:", list(metrics.keys()))
            if os.path.exists(tmp_checkpoint):
                os.remove(tmp_checkpoint)
            return None

        # =====================================================
        # CSV
        # =====================================================
        self.append_evaluation_csv(evaluation_csv, metrics)

        # nettoyage
        if os.path.exists(tmp_checkpoint):
            os.remove(tmp_checkpoint)
            
        # Mise à jour du cache global
        if hasattr(self, 'global_cache'):
            self.global_cache[token_hash] = metrics
            
        return metrics

    # =========================================================
    # CSV
    # =========================================================

    def append_evaluation_csv(self, csv_path, metrics):
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    # =========================================================
    # CHARGER EVALUATIONS
    # =========================================================

    def load_generation_scores(self, generation):
        csv_path = os.path.join(
            self.evaluation_dir,
            f"generation_{generation:04d}.csv"
        )
        if not os.path.exists(csv_path):
            return []
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["fitness"] = float(row["fitness"])
                rows.append(row)
        rows = sorted(rows, key=lambda x: x["fitness"])
        return rows

    # =========================================================
    # SELECTION
    # =========================================================

    def tournament_selection(self, scored_population, tournament_size=3):
        selected = random.sample(scored_population, tournament_size)
        selected = sorted(selected, key=lambda x: x["fitness"])
        return selected[0]

    # =========================================================
    # EVOLUTION
    # =========================================================

    def evolve_generation(self, generation, next_population_size=20):
        scored_population = self.load_generation_scores(generation)
        print(f"\n📊 Individus évalués : {len(scored_population)}")

        # =====================================================
        # ELITISME
        # =====================================================
        elite_count = max(1, int(next_population_size * self.elite_ratio))
        elites = scored_population[:elite_count]
        print(f"🏆 Elites conservés : {elite_count}")

        new_population = []
        next_gen = generation + 1

        # =====================================================
        # COPIE ELITES
        # =====================================================
        for elite in elites:
            tokens = np.load(elite["token_file"])
            filepath = self.save_tokens(tokens, next_gen)
            new_population.append(filepath)

        # =====================================================
        # CREATION ENFANTS
        # =====================================================
        while len(new_population) < next_population_size:
            parent_a = self.tournament_selection(scored_population)
            parent_b = self.tournament_selection(scored_population)
            tokens_a = np.load(parent_a["token_file"])
            tokens_b = np.load(parent_b["token_file"])
            child = self.crossover(tokens_a, tokens_b)
            child = self.mutate(child)
            filepath = self.save_tokens(child, next_gen)
            new_population.append(filepath)
        return new_population

    # =========================================================
    # POPULATION EVALUATION
    # =========================================================

    def evaluate_population(self, population, generation):
        results = []
        for token_file in tqdm.tqdm(population):
            metrics = self.evaluate_individual(token_file, generation)
            if metrics is not None:
                results.append(metrics)
        return results


# =============================================================
# EXEMPLE COMPLET
# =============================================================

if __name__ == "__main__":
    ga = GeneticAlgorithm(
        checkpoint_path="../best_model_detection.pkl",
        evaluation_script="./loss_detection_only_directory.py",
        mutation_sigma=0.05,
        mutation_rate=0.3,
        crossover_rate=0.5,
        keep_norm=False,
        elite_ratio=0.2
    )

    num_generations = 10  # Nombre de générations à simuler
    population_size = 25  # Taille de la population

    print("\n==============================================")
    print("🚀 Lancement de la Génération 0")
    print("==============================================")
    population = ga.create_initial_population(population_size=population_size)
    ga.evaluate_population(population, generation=0)

    for gen in tqdm.tqdm(range(1, num_generations)):
        print(f"\n==============================================")
        print(f"🚀 Lancement de la Génération {gen}")
        print(f"==============================================")
        
        # On évolue à partir de la génération gen-1
        population = ga.evolve_generation(generation=gen-1, next_population_size=population_size)
        # On évalue la nouvelle génération
        ga.evaluate_population(population, generation=gen)

    print("\n✅ Evolution terminée.")
