# Test file with selected features
# Author: Sangeeth Suresh
import pickle
from config import *
from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.feature_selector import GeneticFeatureSelector
from modules.smote_processor import SMOTEProcessor
from modules.mlp_ensemble import MLPEnsemble
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
    # 1️⃣ Load trainval
    trainval_loader = DataLoader(
        r"data\pcos_train_val_union.csv",
        DATA_CONFIG['target_column']
    )
    trainval_loader.load_data()
    X_trainval, y_trainval = trainval_loader.separate_features_target()

    # 2️⃣ Load test
    test_loader = DataLoader(
        r"data\pcos_test_union.csv",
        DATA_CONFIG['target_column']
    )
    test_loader.load_data()
    X_test, y_test = test_loader.separate_features_target()

    # 3️⃣ Clean
    cleaner = DataCleaner()
    X_trainval_clean, cleaned_columns = cleaner.clean_pipeline(X_trainval)
    X_test_clean = cleaner.transform_pipeline(X_test)

    # 4️⃣ Feature selection
    X_train_selected = X_trainval_clean
    X_test_selected = X_test_clean
    selected_feature_names = cleaned_columns

    # 5️⃣ SMOTE
    categorical_features = [16, 17, 18, 19, 20]  # Adjust if needed
    # categorical_features = [5, 6, 7]
    smote = SMOTEProcessor()
    X_train_balanced, y_train_balanced = smote.apply_smote(
        X_train_selected, y_trainval, categorical_features=categorical_features
    )

    # 6️⃣ Train ensemble
    ensemble = MLPEnsemble()
    ensemble.train_with_boosting(X_train_balanced, y_train_balanced)

    # 7️⃣ Predict
    val_predictions = ensemble.predict_individual(X_test_selected)

    # 8️⃣ Ensemble weights
    best_weights = ensemble.get_weights()
    alpha = 10
    weights_exp = np.exp(alpha * np.array(best_weights))
    best_weights_scaled = weights_exp / np.sum(weights_exp)
    final_predictions = ensemble.predict_ensemble_weighted(
        val_predictions, best_weights_scaled
    )
    y_prob = ensemble.predict_proba_ensemble(X_test_selected)[:, 1]
    np.save('MLP_ensemble_probs.npy', y_prob)
    np.save('y_test.npy', y_test)

    # 9️⃣ Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(
        y_test, final_predictions, y_pred_proba=y_prob)
    print(f"✅ Test set results: {results}")

    # 🔟 SHAP explainability
    try:
        shap_explainer = SHAPExplainer(SHAP_CONFIG)

        X_train_df = pd.DataFrame(
            X_train_selected, columns=selected_feature_names)
        X_test_df = pd.DataFrame(
            X_test_selected, columns=selected_feature_names)

        ensemble_function = ensemble.get_ensemble_as_function(
            best_weights_scaled)

        shap_explainer.create_explainer(
            ensemble_function,
            X_train_df,
            feature_names=selected_feature_names
        )

        shap_values = shap_explainer.calculate_shap_values(X_test_df)
        importance_df = shap_explainer.global_feature_importance()

        # Local explanations
        sample_indices = np.arange(min(10, len(X_test_df)))
        shap_explainer.local_explanations(
            X_test_df, y_test, sample_indices=sample_indices
        )

        # Feature interactions
        shap_explainer.feature_interaction_analysis(X_test_df)

        # Dependence plots
        top_features = importance_df.head(5)['feature'].tolist()
        shap_explainer.dependence_plots(top_features, X_test_df)

        # Report
        shap_explainer.generate_explanation_report(X_test_df, y_test)
        print("✅ SHAP analysis completed successfully")

    except Exception as e:
        print(f"❌ SHAP analysis failed: {e}")

    # 1️⃣1️⃣ Save results
    os.makedirs('results', exist_ok=True)
    with open('results/model_weights.pkl', 'wb') as f:
        pickle.dump(best_weights_scaled, f)

    ensemble.save_models(filepath_prefix="results")

    with open('results/evaluation_results.txt', 'w') as f:
        f.write("Model Evaluation Results:\n")
        f.write(str(results))

    print("✅ Results saved successfully!")
    print("Main pipeline completed.")


if __name__ == "__main__":
    for i in range(10):
        main()
