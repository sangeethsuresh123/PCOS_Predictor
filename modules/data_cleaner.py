import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataCleaner:
    """Handle data cleaning, encoding, imputation, and scaling."""

    def __init__(self):
        self.scaler = None
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.label_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def fix_data_types(self, df):
        """Convert known numeric columns that may be loaded as strings to numeric types."""
        df = df.copy()
        # add any other columns that need fixing
        numeric_columns = ['II    beta-HCG(mIU/mL)', 'AMH(ng/mL)']

        for col in numeric_columns:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                self.logger.info(
                    f"Converted {col} from {original_type} to numeric.")

        # Drop unnecessary columns if present
        drop_cols = ['Unnamed: 44']
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                self.logger.info(f"Dropped column: {col}")

        return df

    def handle_missing_values(self, X):
        """Impute missing numeric and categorical values separately."""
        X = X.copy()
        self.numeric_cols = X.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(
            include=['object', 'category']).columns.tolist()

        if not self.is_fitted:
            # Fit imputers
            if self.numeric_cols:
                self.numeric_imputer.fit(X[self.numeric_cols])
            if self.categorical_cols:
                self.categorical_imputer.fit(X[self.categorical_cols])

        # Transform
        if self.numeric_cols:
            X[self.numeric_cols] = self.numeric_imputer.transform(
                X[self.numeric_cols])
        if self.categorical_cols:
            X[self.categorical_cols] = self.categorical_imputer.transform(
                X[self.categorical_cols])

        self.logger.info("Missing values handled successfully.")
        return X

    def encode_categorical_variables(self, X, is_training=True):
        """Encode categorical columns as integer labels."""
        X = X.copy()

        if not self.categorical_cols:
            self.logger.info("No categorical columns to encode.")
            return X

        if is_training:
            for col in self.categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                self.logger.info(f"Encoded column: {col}")
        else:
            for col in self.categorical_cols:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unknowns
                    X_col = X[col].astype(str)
                    mask = X_col.isin(le.classes_)
                    unseen = set(X_col[~mask].unique()) - {'nan'}
                    if unseen:
                        self.logger.warning(
                            f"Unseen categories in {col}: {unseen}")
                    X[col] = -1
                    X.loc[mask, col] = le.transform(X_col[mask])
        return X

    def scale_features_fit(self, X):
        """Fit scaler on numeric columns and scale them."""
        self.scaler = StandardScaler()
        X_scaled = X.copy()

        if self.numeric_cols:
            X_scaled[self.numeric_cols] = self.scaler.fit_transform(
                X_scaled[self.numeric_cols])
            self.logger.info(f"Scaled numeric columns: {self.numeric_cols}")
        return X_scaled

    def scale_features_transform(self, X):
        """Scale numeric columns using previously fitted scaler."""
        X_scaled = X.copy()

        if self.numeric_cols and self.scaler:
            X_scaled[self.numeric_cols] = self.scaler.transform(
                X_scaled[self.numeric_cols])
            self.logger.info(f"Scaled numeric columns: {self.numeric_cols}")
        return X_scaled

    def clean_pipeline(self, X_train):
        self.logger.info("Starting data cleaning pipeline (training)")
        X_train = self.fix_data_types(X_train)
        self.is_fitted = False
        X_train = self.handle_missing_values(X_train)
        X_train = self.encode_categorical_variables(X_train, is_training=True)
        X_train = self.scale_features_fit(X_train)
        # Capture column names after all transformations
        self.output_columns = X_train.columns.tolist()
        self.logger.info(
            f"Training data cleaning completed with columns: {self.output_columns}")
        return X_train, self.output_columns

    def transform_pipeline(self, X):
        self.logger.info("Starting data transformation pipeline")
        X = self.fix_data_types(X)
        self.is_fitted = True
        X = self.handle_missing_values(X)
        X = self.encode_categorical_variables(X, is_training=False)
        X = self.scale_features_transform(X)
        self.logger.info(
            f"Data transformation completed. Columns: {self.output_columns}")
        # returns a DataFrame
        return pd.DataFrame(X, columns=self.output_columns)
