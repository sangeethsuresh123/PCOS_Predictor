from datetime import datetime
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
import numpy as np
import pandas as pd


class ModelEvaluator:
    """Test and evaluate the final model"""

    def __init__(self):
        """Initialize evaluation metrics"""
        self.metrics = {}
        self.confusion_mat = None
        self.classification_rep = None

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        # Ensure shapes
        if y_pred_proba is not None and y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(
            y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        auc_score = None
        if y_pred_proba is not None:
            try:
                auc_score = roc_auc_score(y_true, y_pred_proba)
            except ValueError as e:
                print(f"AUC could not be computed: {e}")

        self.metrics = {
            'accuracy': round(accuracy, 4),
            'precision_weighted': round(precision_weighted, 4),
            'recall_weighted': round(recall_weighted, 4),
            'f1_weighted': round(f1_weighted, 4),
            'precision_macro': round(precision_macro, 4),
            'recall_macro': round(recall_macro, 4),
            'f1_macro': round(f1_macro, 4),
            'auc_score': round(auc_score, 4) if auc_score is not None else None,
            'total_samples': len(y_true),
            'evaluation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Print summary
        print("\nModel Evaluation Results:")
        for k, v in self.metrics.items():
            if k != "evaluation_timestamp":
                print(f"{k}: {v}")

        return self.metrics

    def generate_confusion_matrix(self, y_true, y_pred):
        """
        Create confusion matrix
        Return confusion matrix
        """
        try:
            # Calculate confusion matrix
            self.confusion_mat = confusion_matrix(y_true, y_pred)

            # Get unique labels
            labels = sorted(list(set(y_true) | set(y_pred)))

            # Create DataFrame for better visualization
            cm_df = pd.DataFrame(
                self.confusion_mat,
                index=[f'True_{label}' for label in labels],
                columns=[f'Pred_{label}' for label in labels]
            )

            print("\nConfusion Matrix:")
            print(cm_df)

            return self.confusion_mat

        except Exception as e:
            print(f"Error generating confusion matrix: {str(e)}")
            return None

    def generate_classification_report(self, y_true, y_pred):
        """
        Generate detailed classification report
        Return report
        """
        try:
            # Generate classification report
            self.classification_rep = classification_report(
                y_true, y_pred,
                output_dict=True,
                zero_division=0
            )

            # Also get string version for display
            report_str = classification_report(
                y_true, y_pred,
                zero_division=0
            )

            print("\nClassification Report:")
            print(report_str)

            return self.classification_rep

        except Exception as e:
            print(f"Error generating classification report: {str(e)}")
            return None

    def save_results(self, results, filepath):
        """
        Save evaluation results
        Return success status
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Prepare comprehensive results
            comprehensive_results = {
                'evaluation_metrics': results,
                'confusion_matrix': self.confusion_mat.tolist() if self.confusion_mat is not None else None,
                'classification_report': self.classification_rep,
                'summary': self._generate_summary()
            }

            # Save as both JSON and text
            base_path = filepath.rsplit('.', 1)[0]

            # Save JSON version
            json_path = f"{base_path}.json"
            with open(json_path, 'w') as f:
                json.dump(comprehensive_results, f, indent=2)

            # Save text version
            txt_path = f"{base_path}.txt"
            with open(txt_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("MODEL EVALUATION RESULTS\n")
                f.write("="*60 + "\n\n")

                # Write metrics
                f.write("PERFORMANCE METRICS:\n")
                f.write("-"*30 + "\n")
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")

                # Write confusion matrix
                if self.confusion_mat is not None:
                    f.write(f"\nCONFUSION MATRIX:\n")
                    f.write("-"*30 + "\n")
                    f.write(str(self.confusion_mat) + "\n")

                # Write classification report
                if self.classification_rep is not None:
                    f.write(f"\nCLASSIFICATION REPORT:\n")
                    f.write("-"*30 + "\n")
                    report_str = classification_report(
                        [], [], output_dict=False)
                    # Convert dict back to string format
                    for class_name, metrics in self.classification_rep.items():
                        if isinstance(metrics, dict):
                            f.write(f"{class_name}:\n")
                            for metric, value in metrics.items():
                                f.write(f"  {metric}: {value}\n")
                        else:
                            f.write(f"{class_name}: {metrics}\n")

                # Write summary
                f.write(f"\nSUMMARY:\n")
                f.write("-"*30 + "\n")
                f.write(self._generate_summary())

            print(f"\nResults saved successfully:")
            print(f"JSON: {json_path}")
            print(f"Text: {txt_path}")

            return True

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False

    def _generate_summary(self):
        """Generate a summary of the evaluation results"""
        if not self.metrics:
            return "No evaluation metrics available."

        summary = []
        summary.append(
            f"Model evaluated on {self.metrics.get('total_samples', 'N/A')} samples")
        summary.append(
            f"Overall Accuracy: {self.metrics.get('accuracy', 'N/A')}")
        summary.append(
            f"Weighted F1-Score: {self.metrics.get('f1_weighted', 'N/A')}")

        if self.metrics.get('auc_score'):
            summary.append(
                f"AUC Score: {self.metrics.get('auc_score', 'N/A')}")

        # Performance assessment
        f1_score = self.metrics.get('f1_weighted', 0)
        if f1_score >= 0.9:
            performance = "Excellent"
        elif f1_score >= 0.8:
            performance = "Good"
        elif f1_score >= 0.7:
            performance = "Fair"
        else:
            performance = "Needs Improvement"

        summary.append(f"Performance Assessment: {performance}")
        summary.append(
            f"Evaluation completed at: {self.metrics.get('evaluation_timestamp', 'N/A')}")

        return "\n".join(summary)
