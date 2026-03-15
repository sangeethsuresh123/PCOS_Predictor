from collections import Counter
from imblearn.over_sampling import SMOTE, SMOTENC
import numpy as np
import pandas as pd


class SMOTEProcessor:
    """Handle SMOTE/SMOTENC oversampling depending on feature types."""

    def __init__(self, random_state=42, k_neighbors=5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.smote = None
        self.original_distribution = None
        self.resampled_distribution = None

    def _identify_categorical_indices(self, X):
        """
        Identify indices of categorical columns if X is a DataFrame.
        Otherwise return empty list for NumPy arrays.
        """
        if isinstance(X, pd.DataFrame):
            return list(X.select_dtypes(include=['category', 'object']).columns)
        else:
            return []  # NumPy arrays are assumed to be already numeric

    def apply_smote(self, X, y, categorical_features=None):
        """
        Apply SMOTE or SMOTENC oversampling.

        Args:
            X (pd.DataFrame or np.ndarray): feature matrix
            y (pd.Series or np.ndarray): target labels
            categorical_features (list of int): column indices that are categorical

        Returns:
            tuple: (X_resampled, y_resampled)
        """
        # Store original distribution
        self.original_distribution = self.get_class_distribution(y)
        print(f"Original class distribution: {self.original_distribution}")

        # Determine categorical indices
        cat_indices = []
        if categorical_features is not None:
            cat_indices = categorical_features
        elif isinstance(X, pd.DataFrame):
            cat_indices = self._identify_categorical_indices(X)

        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        k_neighbors_adjusted = max(
            1, min(self.k_neighbors, min_class_count - 1))

        if k_neighbors_adjusted != self.k_neighbors:
            print(
                f"Warning: Adjusting k_neighbors to {k_neighbors_adjusted} due to small class size ({min_class_count})")

        # Choose SMOTENC if we have categorical indices
        if len(cat_indices) > 0:
            print(
                f"Using SMOTENC for categorical features at indices: {cat_indices}")
            self.smote = SMOTENC(
                categorical_features=cat_indices,
                random_state=self.random_state,
                k_neighbors=k_neighbors_adjusted
            )
        else:
            print("Using plain SMOTE")
            self.smote = SMOTE(
                random_state=self.random_state,
                k_neighbors=k_neighbors_adjusted
            )

        # Fit and resample
        X_resampled, y_resampled = self.smote.fit_resample(X, y)

        # Store new class distribution
        self.resampled_distribution = self.get_class_distribution(y_resampled)
        self._print_oversampling_summary()

        return X_resampled, y_resampled

    def get_class_distribution(self, y):
        """Analyze class distribution as a dictionary."""
        class_counts = Counter(y)
        total = len(y)
        return {
            label: {'count': count, 'percentage': round((count/total)*100, 2)}
            for label, count in class_counts.items()
        }

    def _print_oversampling_summary(self):
        """Pretty-print summary of class distributions before/after SMOTE."""
        if self.original_distribution and self.resampled_distribution:
            print("\n=== SMOTE OVERSAMPLING SUMMARY ===")
            for label in self.original_distribution:
                orig_count = self.original_distribution[label]['count']
                new_count = self.resampled_distribution[label]['count']
                print(
                    f"Class {label}: {orig_count} -> {new_count} (+{new_count-orig_count})")
            print("===================================\n")

    def get_oversampling_info(self):
        return {
            'original_distribution': self.original_distribution,
            'resampled_distribution': self.resampled_distribution,
            'k_neighbors_used': self.smote.k_neighbors if self.smote else None
        }
