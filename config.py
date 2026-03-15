DATA_CONFIG = {
    # 'csv_path': 'D:\Research\kottarathil\kottarathil_csv_new4.csv',
    # 'csv_path': '/content/drive/MyDrive/Research/implementation/data/kottarathil_csv_hardcoded.csv',
    'csv_path': 'data/pcos_train_val.csv',
    'target_column': 'PCOS (Y/N)',
    # 'test_size': 0.3,
    'test_size': 0.3,
    'random_state': 42
}

# Genetic Algorithm configuration (from Table 2)
GENETIC_ALGORITHM_CONFIG = {
    'population_size': 50,
    'generations': 100,
    'mutation_rate': 0.05,
    'crossover_rate': 0.8,
    'selection_method': 'tournament',
    'tournament_size': 3,
    'crossover_type': 'uniform',
    'mutation_type': 'bit_flip',
    'fitness_function': 'model_accuracy'
}

SMOTE_CONFIG = {
    'random_state': 42,
    'k_neighbors': 5
}

# MLP configurations (from Table 5 continued)
"""MLP_CONFIGURATIONS = {
    'config_1': {
        'hidden_layer_sizes': (64, 64),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,  # L2 regularization (dropout alternative)
        'max_iter': 500,
        'random_state': 42
    },
    'config_2': {
        'hidden_layer_sizes': (128, 128, 128),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'max_iter': 500,
        'random_state': 43
    },
    'config_3': {
        'hidden_layer_sizes': (80, 80),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'max_iter': 500,
        'random_state': 44
    }
}"""
MLP_CONFIGURATIONS = {
    'config_1': {
        'name': 'Config_1',
        'hidden_layers': [64, 64],  # 2 hidden layers with 64 neurons each
        'activation': 'relu',
        'dropout_rate': 0.3,  # Dropout regularization
        'l2_reg': 0.01,  # L2 regularization
        'learning_rate': 0.001,
        'random_state': 42
    },
    'config_2': {
        'name': 'Config_2',
        # 3 hidden layers with 128 neurons each
        'hidden_layers': [128, 128, 128],
        'activation': 'relu',
        'dropout_rate': 0.3,  # Dropout regularization
        'l2_reg': 0.01,  # L2 regularization
        'learning_rate': 0.001,
        'random_state': 43
    },
    'config_3': {
        'name': 'Config_3',
        'hidden_layers': [80, 80],  # 2 hidden layers with 80 neurons each
        'activation': 'relu',
        'dropout_rate': 0.3,  # Dropout regularization
        'l2_reg': 0.01,  # L2 regularization
        'learning_rate': 0.001,
        'random_state': 44
    },
    'config_4': {
        'name': 'Config_4',
        # 3 hidden layers with 96 neurons each (NEW CONFIGURATION)
        'hidden_layers': [96, 96, 96],
        'activation': 'relu',
        'dropout_rate': 0.3,  # Dropout regularization
        'l2_reg': 0.01,  # L2 regularization
        'learning_rate': 0.001,
        'random_state': 45
    }
}

# Ensemble configuration - combines all 3 MLP configurations
ENSEMBLE_CONFIG = {
    'n_estimators': 3,
    'ensemble_method': 'weighted_average',  # TSA optimized weights
    'use_boosting': True,
    'boosting_algorithm': 'AdaBoost',
    'combination_strategy': 'strength_based'  # Combines strengths of all models
}

# Tunicate Swarm Algorithm configuration (from Table 3)
TUNICATE_CONFIG = {
    'population_size': 30,
    'max_iterations': 200,
    'search_space_bounds': [0, 1],  # for each weight
    'inertia_weight': 0.5,
    'adaptive_coefficient': 0.7,
    'convergence_threshold': 1e-5
}

# SHAP Explainability configuration
SHAP_CONFIG = {
    'explainer_type': 'kernel',  # 'deep', 'kernel', 'tree', 'linear'
    'background_samples': 100,  # Number of background samples for SHAP
    'max_evals': 1000,  # Maximum evaluations for kernel explainer
    # 'interventional' or 'tree_path_dependent'
    'feature_perturbation': 'interventional',
    'output_indices': None,  # Which outputs to explain (None for all)
    'silent': False,  # Suppress SHAP progress bars

    # Visualization settings
    'max_display': 20,  # Maximum features to display in plots
    'plot_size': (12, 8),  # Figure size for plots
    'show_values': True,  # Show SHAP values on plots
    'sort': True,  # Sort features by importance
    'color_bar': True,  # Show color bar in plots

    # Save settings
    'save_shap_values': True,  # Save SHAP values to file
    'save_plots': True,  # Save visualization plots
    'plot_format': 'png',  # Plot format: 'png', 'pdf', 'svg'
    'dpi': 300,  # Plot resolution

    # Individual explanation settings
    'n_individual_explanations': 10,  # Number of individual predictions to explain
    'individual_explanation_type': 'force',  # 'waterfall', 'force', 'decision'
}
