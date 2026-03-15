import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging


class DataLoader:
    """Handle CSV data loading and basic preprocessing"""

    def __init__(self, file_path, target_column):
        """Initialize parameters"""
        self.file_path = file_path
        self.target_column = target_column
        self.data = None
        self.X = None
        self.y = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Load CSV file
        Return pandas DataFrame
        """
        try:
            self.data = pd.read_csv(self.file_path)
            self.logger.info(f"Data loaded successfully: {self.data.shape}")
            return self.data

        except FileNotFoundError:
            self.logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def separate_features_target(self):
        """
        Separate X and y
        Return X, y
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Separate features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

        self.logger.info(f"Features shape: {self.X.shape}")
        self.logger.info(f"Target shape: {self.y.shape}")

        return self.X, self.y

    def train_test_split(self, test_size=0.3):
        """
        Apply 70-30 split
        Return X_train, X_test, y_train, y_test
        """
        if self.X is None or self.y is None:
            raise ValueError(
                "Features and target not separated. Call separate_features_target() first.")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y
        )

        self.logger.info(
            f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test
