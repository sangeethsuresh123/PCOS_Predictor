import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class GeneticFeatureSelector:
    """Genetic Algorithm for feature selection optimized for MLP ensemble models"""

    def __init__(self, population_size=50, generations=100, mutation_rate=0.05,
                 crossover_rate=0.8, selection_method='tournament', tournament_size=3,
                 alpha=0.9, beta=0.1, evaluation_method='mlp_ensemble',
                 use_feature_diversity=True, diversity_weight=0.05):
        """
        Initialize GA parameters optimized for MLP ensemble feature selection

        Args:
            evaluation_method: 'mlp_proxy', 'mlp_ensemble', or 'random_forest'
            use_feature_diversity: Whether to include feature diversity in fitness
            diversity_weight: Weight for diversity component in fitness
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.alpha = alpha  # Weight for accuracy
        self.beta = beta    # Weight for feature count penalty
        self.diversity_weight = diversity_weight  # Weight for diversity bonus
        self.use_feature_diversity = use_feature_diversity
        self.evaluation_method = evaluation_method

        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.scaler = StandardScaler()  # For MLP preprocessing

        # Initialize base classifier based on evaluation method
        self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize the appropriate classifier for fitness evaluation"""
        if self.evaluation_method == 'mlp_proxy':
            # Single MLP as proxy for ensemble
            self.base_classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                learning_rate_init=0.001,
                solver='adam'
            )
            print("Using MLP proxy for feature evaluation")

        elif self.evaluation_method == 'mlp_ensemble':
            # Multiple MLPs to simulate ensemble
            self.mlp_ensemble = [
                MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=200,
                    random_state=42 + i,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
                    learning_rate_init=0.001 * (1 + i * 0.1),
                    solver='adam'
                ) for i in range(3)
            ]
            print("Using MLP ensemble for feature evaluation")

        else:  # random_forest
            self.base_classifier = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )
            print("Using Random Forest for feature evaluation")

    def create_population(self, n_features):
        """Create initial population with smart initialization"""
        population = []

        # Create diverse initial population
        for i in range(self.population_size):
            if i == 0:
                # First individual: select all features (baseline)
                individual = np.ones(n_features, dtype=int)
            elif i == 1:
                # Second individual: select 30% of features randomly
                individual = np.random.choice(
                    [0, 1], size=n_features, p=[0.7, 0.3])
            elif i == 2:
                # Third individual: select 50% of features randomly
                individual = np.random.choice(
                    [0, 1], size=n_features, p=[0.5, 0.5])
            else:
                # Random selection with bias toward fewer features
                individual = np.random.choice(
                    [0, 1], size=n_features, p=[0.75, 0.25])

            # Ensure at least one feature is selected
            if np.sum(individual) == 0:
                individual[np.random.randint(0, n_features)] = 1

            population.append(individual)

        return np.array(population)

    def _calculate_feature_diversity(self, individual, correlation_matrix):
        """Calculate diversity bonus based on feature correlation"""
        selected_features = np.where(individual == 1)[0]

        if len(selected_features) <= 1:
            return 0.0

        # Get correlation submatrix for selected features
        selected_corr = correlation_matrix[np.ix_(
            selected_features, selected_features)]

        # Calculate average absolute correlation (excluding diagonal)
        mask = ~np.eye(selected_corr.shape[0], dtype=bool)
        avg_correlation = np.mean(np.abs(selected_corr[mask]))

        # Diversity bonus: lower correlation = higher diversity
        diversity_bonus = 1.0 - avg_correlation

        return diversity_bonus

    def fitness_function(self, individual, X, y):
        """
        Enhanced fitness function optimized for MLP ensemble
        F(Ci) = α·Accuracy(Ci) - β·NSF(Ci) + γ·Diversity(Ci)
        """
        # Convert pandas DataFrame to numpy array if needed
        if hasattr(X, 'values'):
            X_np = X.values
        else:
            X_np = X
        if hasattr(y, 'values'):
            y_np = y.values
        else:
            y_np = y

        # Get selected features
        selected_features = np.where(individual == 1)[0]

        # Penalty for selecting no features
        if len(selected_features) == 0:
            return -1.0

        # Extract selected features
        X_selected = X_np[:, selected_features]

        try:
            # Scale features for MLP (important for neural networks)
            if self.evaluation_method in ['mlp_proxy', 'mlp_ensemble']:
                X_scaled = self.scaler.fit_transform(X_selected)
            else:
                X_scaled = X_selected

            # Calculate accuracy based on evaluation method
            if self.evaluation_method == 'mlp_proxy':
                accuracy = self._evaluate_single_mlp(X_scaled, y_np)

            elif self.evaluation_method == 'mlp_ensemble':
                accuracy = self._evaluate_mlp_ensemble(X_scaled, y_np)

            else:  # random_forest
                cv_scores = cross_val_score(
                    self.base_classifier, X_scaled, y_np,
                    cv=3, scoring='accuracy', n_jobs=-1
                )
                accuracy = np.mean(cv_scores)

            # NSF (Number of Selected Features) component
            nsf = len(selected_features)
            total_features = len(individual)
            nsf_normalized = nsf / total_features

            # Base fitness calculation
            fitness = self.alpha * accuracy - self.beta * nsf_normalized

            # Add diversity bonus if enabled
            if self.use_feature_diversity and hasattr(self, 'correlation_matrix'):
                diversity_bonus = self._calculate_feature_diversity(
                    individual, self.correlation_matrix)
                fitness += self.diversity_weight * diversity_bonus

            return fitness

        except Exception as e:
            # Return low fitness for problematic individuals
            return -1.0

    def _evaluate_single_mlp(self, X_scaled, y):
        """Evaluate using single MLP with cross-validation"""
        try:
            cv_scores = cross_val_score(
                self.base_classifier, X_scaled, y,
                cv=3, scoring='accuracy', n_jobs=1  # Reduced parallelism for MLPs
            )
            return np.mean(cv_scores)
        except:
            return 0.0

    def _evaluate_mlp_ensemble(self, X_scaled, y):
        """Evaluate using lightweight MLP ensemble"""
        try:
            ensemble_scores = []

            for mlp in self.mlp_ensemble:
                # Use simple train-test split instead of CV for speed
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )

                mlp.fit(X_train, y_train)
                score = mlp.score(X_val, y_val)
                ensemble_scores.append(score)

            # Return mean ensemble performance
            return np.mean(ensemble_scores)

        except:
            return 0.0

    def selection(self, population, fitness_scores):
        """Enhanced selection mechanism"""
        selected = []

        if self.selection_method == 'tournament':
            for _ in range(self.population_size):
                # Tournament selection
                tournament_indices = np.random.choice(
                    len(population),
                    size=min(self.tournament_size, len(population)),
                    replace=False
                )
                tournament_fitness = [fitness_scores[i]
                                      for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())

        elif self.selection_method == 'roulette':
            # Roulette wheel selection with fitness scaling
            min_fitness = min(fitness_scores)
            adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]

            # Apply fitness scaling to increase selection pressure
            scaled_fitness = [f**2 for f in adjusted_fitness]
            total_fitness = sum(scaled_fitness)

            if total_fitness > 0:
                probabilities = [f / total_fitness for f in scaled_fitness]
            else:
                probabilities = [1/len(population)] * len(population)

            for _ in range(self.population_size):
                selected_idx = np.random.choice(
                    len(population), p=probabilities)
                selected.append(population[selected_idx].copy())

        return np.array(selected)

    def crossover(self, parent1, parent2):
        """Enhanced crossover operation"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Uniform crossover with feature preservation
        offspring1 = np.zeros_like(parent1)
        offspring2 = np.zeros_like(parent2)

        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
            else:
                offspring1[i] = parent2[i]
                offspring2[i] = parent1[i]

        # Ensure at least one feature is selected
        if np.sum(offspring1) == 0:
            offspring1[np.random.randint(0, len(offspring1))] = 1
        if np.sum(offspring2) == 0:
            offspring2[np.random.randint(0, len(offspring2))] = 1

        return offspring1, offspring2

    def mutate(self, individual):
        """Enhanced mutation operation"""
        mutated = individual.copy()

        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit

        # Ensure at least one feature is selected
        if np.sum(mutated) == 0:
            mutated[np.random.randint(0, len(mutated))] = 1

        return mutated

    def select_features(self, X, y):
        """Main GA execution with enhanced features"""
        # Convert pandas DataFrame to numpy array if needed
        if hasattr(X, 'values'):
            X_np = X.values
        else:
            X_np = X
        if hasattr(y, 'values'):
            y_np = y.values
        else:
            y_np = y

        n_features = X_np.shape[1]

        # Pre-compute correlation matrix for diversity calculation
        if self.use_feature_diversity:
            try:
                self.correlation_matrix = np.corrcoef(X_np.T)
                # Handle NaN values in correlation matrix
                self.correlation_matrix = np.nan_to_num(
                    self.correlation_matrix)
                print("Feature correlation matrix computed for diversity calculation")
            except:
                self.use_feature_diversity = False
                print("Could not compute correlation matrix, disabling diversity bonus")

        # Create initial population
        population = self.create_population(n_features)

        print(f"Starting Enhanced Genetic Algorithm Feature Selection...")
        print(f"Population Size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Initial Features: {n_features}")
        print(f"Evaluation Method: {self.evaluation_method}")
        print(f"Alpha (Accuracy Weight): {self.alpha}")
        print(f"Beta (Feature Count Penalty): {self.beta}")
        if self.use_feature_diversity:
            print(f"Diversity Weight: {self.diversity_weight}")

        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = self.fitness_function(individual, X_np, y_np)
                fitness_scores.append(fitness)

            # Track best individual
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = population[current_best_idx].copy()

            self.fitness_history.append(self.best_fitness)

            # Print progress
            if generation % 20 == 0 or generation == self.generations - 1:
                selected_count = np.sum(self.best_individual)
                avg_fitness = np.mean(fitness_scores)
                print(f"Generation {generation}: Best Fitness = {self.best_fitness:.4f}, "
                      f"Avg Fitness = {avg_fitness:.4f}, "
                      f"Features Selected = {selected_count}/{n_features}")

            # Create next generation
            if generation < self.generations - 1:
                # Elitism: Keep best individual
                selected_parents = self.selection(population, fitness_scores)
                selected_parents[0] = self.best_individual.copy()

                # Crossover and Mutation
                new_population = [self.best_individual.copy()]  # Elitism

                for i in range(1, self.population_size, 2):
                    parent1 = selected_parents[i % len(selected_parents)]
                    parent2 = selected_parents[(i + 1) % len(selected_parents)]

                    # Crossover
                    child1, child2 = self.crossover(parent1, parent2)

                    # Mutation
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    new_population.extend([child1, child2])

                # Keep population size consistent
                population = np.array(new_population[:self.population_size])

        # Get final selected features
        feature_indices = np.where(self.best_individual == 1)[0]
        print("Feature indices: ", feature_indices)
        X_selected = X_np[:, feature_indices]

        print(f"\nFeature Selection Complete!")
        print(f"Selected {len(feature_indices)} features out of {n_features}")
        print(f"Final Fitness Score: {self.best_fitness:.4f}")
        print(
            f"Feature reduction: {(1 - len(feature_indices)/n_features)*100:.1f}%")

        return X_selected, feature_indices

    def get_selected_feature_names(self, feature_names, feature_indices):
        """Return names of selected features for SHAP interpretation"""
        if isinstance(feature_names, (list, np.ndarray)):
            selected_feature_names = [feature_names[i]
                                      for i in feature_indices]
        else:
            # If feature_names is None or not iterable, create generic names
            selected_feature_names = [f"feature_{i}" for i in feature_indices]

        return selected_feature_names

    def get_fitness_history(self):
        """Return fitness evolution history"""
        return self.fitness_history

    def get_feature_importance(self, feature_names=None):
        """Get feature importance based on selection"""
        if self.best_individual is None:
            return None

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(
                len(self.best_individual))]

        importance_dict = {}
        for i, selected in enumerate(self.best_individual):
            importance_dict[feature_names[i]] = int(selected)

        return importance_dict

    def plot_fitness_evolution(self):
        """Plot fitness evolution over generations"""
        if not self.fitness_history:
            print("No fitness history available")
            return

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.fitness_history, 'b-', linewidth=2)
            plt.title('Fitness Evolution Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
