# Training file to select features
from config import *
from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.feature_selector import GeneticFeatureSelector
from modules.smote_processor import SMOTEProcessor
from modules.mlp_ensemble_copy import MLPEnsemble
from modules.model_evaluator import ModelEvaluator
from modules.shap_explainer import SHAPExplainer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


def main():
    # 1. Load data
    data_loader = DataLoader(
        DATA_CONFIG['csv_path'], DATA_CONFIG['target_column'])
    data_loader.load_data()
    X, y = data_loader.separate_features_target()

    # Store original feature names before any processing
    if hasattr(X, 'columns'):
        original_feature_names = list(X.columns)
    else:
        # If X is already a numpy array, create generic names
        original_feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    n_splits = 10  # 5 or 10 if you want finer granularity

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    all_selected_features = []

    for fold_num, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"\n{'='*30}\nFOLD {fold_num}/{n_splits}\n{'='*30}")

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 1️⃣ Clean data
        cleaner = DataCleaner()
        X_train_clean, clean_columns = cleaner.clean_pipeline(X_train)
        X_val_clean = cleaner.transform_pipeline(X_val)

        # 2️⃣ Feature selection
        try:
            # Try to use config values
            feature_selector = GeneticFeatureSelector(
                population_size=GA_CONFIG['population_size'],
                generations=GA_CONFIG['generations'],
                mutation_rate=GA_CONFIG['mutation_rate'],
                crossover_rate=GA_CONFIG['crossover_rate'],
                selection_method=GA_CONFIG['selection_method']
            )
            print("Using GA_CONFIG parameters from config file")
        except (NameError, KeyError):
            # Use default values if config not available
            feature_selector = GeneticFeatureSelector(
                population_size=50,
                generations=100,
                mutation_rate=0.05,
                crossover_rate=0.8,
                selection_method='tournament'
            )
            print("Using default GA parameters")

        X_train_selected, selected_indices = feature_selector.select_features(
            X_train_clean, y_train)
        # X_val_selected = X_val_clean[:, selected_indices]
        X_val_selected = X_val_clean.iloc[:, selected_indices]
        selected_feature_names = feature_selector.get_selected_feature_names(
            original_feature_names, selected_indices
        )
        all_selected_features.append(selected_feature_names)

        # 3️⃣ SMOTE
        # Map to the *new* indices after feature selection
        original_categorical = [11, 27, 28, 29, 30,
                                31, 32, 33]  # indices in original data
        categorical_features = [
            new_idx
            for new_idx, orig_idx in enumerate(selected_indices)
            if orig_idx in original_categorical
        ]
        print(
            f"Selected categorical feature indices for SMOTE: {categorical_features}")

        smote = SMOTEProcessor()
        X_train_balanced, y_train_balanced = smote.apply_smote(
            X_train_selected, y_train, categorical_features=categorical_features)

        X_train_balanced = X_train_selected
        y_train_balanced = y_train
        # 4️⃣ Train ensemble
        ensemble = MLPEnsemble()
        ensemble.train_with_boosting(X_train_balanced, y_train_balanced)

        # 5️⃣ Predict + softmax weight scaling
        train_predictions = ensemble.predict_individual(X_train_balanced)
        val_predictions = ensemble.predict_individual(X_val_selected)

        best_weights = ensemble.get_weights()
        alpha = 10
        weights_exp = np.exp(alpha * np.array(best_weights))
        best_weights_scaled = weights_exp / np.sum(weights_exp)

        final_predictions = ensemble.predict_ensemble_weighted(
            val_predictions, best_weights_scaled)

        # 6️⃣ Evaluate
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(y_val, final_predictions)
        fold_results.append(results)

        print(f"Fold {fold_num} results: {results}")

    # Average accuracy / F1 / whatever metrics you track
    all_accuracies = [res['accuracy'] for res in fold_results]
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)

    print(f"\nCross-validation accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Optionally save or print selected features per fold
    for i, feats in enumerate(all_selected_features, 1):
        print(f"Fold {i} selected features: {feats}")

     # 10. SHAP explainability analysis
    try:
        shap_explainer = SHAPExplainer(SHAP_CONFIG)
        ensemble_function = ensemble.get_ensemble_as_function(best_weights)
        shap_explainer.create_explainer(
            ensemble_function, X_train_selected, selected_feature_names)

        # 11. Calculate SHAP values
        shap_values = shap_explainer.calculate_shap_values(X_val_selected)

        # 12. Generate global feature importance
        importance_df = shap_explainer.global_feature_importance()

        # 13. Generate local explanations for sample predictions
        # First 10 samples or all if less
        sample_indices = np.arange(min(10, len(X_val_selected)))
        local_explanations = shap_explainer.local_explanations(
            X_val_selected, y_val, sample_indices=sample_indices)

        # 14. Analyze feature interactions
        interaction_results = shap_explainer.feature_interaction_analysis(
            X_val_selected)

        # 15. Create dependence plots for top features
        top_features = importance_df.head(5)['feature'].tolist()
        shap_explainer.dependence_plots(top_features, X_val_selected)

        # 16. Generate comprehensive explanation report
        shap_explainer.generate_explanation_report(X_val_selected, y_val)

        print("SHAP analysis completed successfully")

    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        print("Continuing without SHAP explanations...")

    # 17. Save results
    print("\nSaving results...")

    # Save selected features
    with open('results/selected_features.txt', 'w') as f:
        f.write("Selected Feature Indices:\n")
        f.write(','.join(map(str, selected_indices)) + '\n\n')
        f.write("Selected Feature Names:\n")
        f.write('\n'.join(selected_feature_names))

    # Save model weights
    import pickle
    with open('results/model_weights.pkl', 'wb') as f:
        pickle.dump(best_weights, f)

    # Save evaluation results
    with open('results/evaluation_results.txt', 'w') as f:
        f.write("Model Evaluation Results:\n")
        f.write(str(results))

    print("Results saved successfully!")
    print("Main pipeline completed.")


if __name__ == "__main__":
    main()
