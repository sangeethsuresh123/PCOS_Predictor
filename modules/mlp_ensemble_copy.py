import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from keras.metrics import Precision, Recall
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings
import os

# Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("Running on CPU")


class MLPEnsemble(BaseEstimator, ClassifierMixin):
    """Create ensemble of 4 MLPs using Keras/TensorFlow with LogitBoost and weighted averaging"""

    def __init__(self, configurations=None, epochs=1000, batch_size=32, verbose=0):
        """
        Initialize with 4 specific MLP configurations
        Config 1: 2 hidden layers with 64 neurons each + Dropout (F1: 91.23%)
        Config 2: 3 hidden layers with 128 neurons each + Dropout (F1: 91.64%)
        Config 3: 2 hidden layers with 80 neurons each + Dropout (F1: 90.39%)
        Config 4: 3 hidden layers with 96 neurons each + Dropout (New configuration)
        """
        self.configurations = configurations or self._get_default_configurations()
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.models = []
        self.model_weights = []
        self.sample_weights = None
        self.working_response = None
        self.working_weights = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_classes_ = None
        self.is_fitted = False
        self.label_encoder = LabelEncoder()
        self.history = []

    def _get_default_configurations(self):
        """Get the 4 MLP configurations including the new 4th configuration"""
        return [
            {
                'name': 'Config_1',
                # 2 hidden layers with 64 neurons each
                'hidden_layers': [64, 64],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'l2_reg': 0.01,
                'learning_rate': 0.001,
                'random_state': 42
            },
            {
                'name': 'Config_2_Deeper',
                # 3 hidden layers with 128 neurons each
                'hidden_layers': [128, 128, 128],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'l2_reg': 0.01,
                'learning_rate': 0.001,
                'random_state': 43
            },
            {
                'name': 'Config_3_LeakyReLU',
                # 2 hidden layers with 80 neurons each, using Leaky ReLU
                'hidden_layers': [80, 80],
                'activation': 'leaky_relu',
                'dropout_rate': 0.3,
                'l2_reg': 0.01,
                'learning_rate': 0.001,
                'random_state': 44
            },
            {
                'name': 'Config_4_LowerReg',
                # 2 hidden layers with 96 neurons each, lower L2 regularization
                'hidden_layers': [96, 96],
                'activation': 'relu',
                'dropout_rate': 0.2,
                'l2_reg': 0.001,
                'learning_rate': 0.001,
                'random_state': 45
            }
        ]

    def validate_and_complete_config(self, config):
        """Ensure config has all required keys with sensible defaults"""
        defaults = {
            'name': 'mlp_model',
            'random_state': 42,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'hidden_layers': [64, 32],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'l2_reg': 0.001,
            'patience': 10,
            'min_delta': 0.001,
            'verbose': 0
        }

        # Create a copy to avoid modifying the original
        if isinstance(config, dict):
            complete_config = config.copy()
        else:
            # If config is not a dict, create a new one
            complete_config = {}

        # Add missing keys with defaults
        for key, default_value in defaults.items():
            if key not in complete_config or complete_config[key] is None:
                complete_config[key] = default_value

        return complete_config

    def _create_mlp_model(self, config, input_dim, n_classes):
        '''Create a single MLP model based on configuration'''
        # Set random seed for reproducibility with default values
        random_state = config.get('random_state', 42)
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        # Get configuration values with defaults
        # Generate unique name
        name = config.get('name', f'mlp_model_{random_state}')
        hidden_layers = config.get('hidden_layers', [64, 32])
        activation = config.get('activation', 'relu')
        l2_reg = config.get('l2_reg', 0.001)
        dropout_rate = config.get('dropout_rate', 0.2)
        learning_rate = config.get('learning_rate', 0.001)

        model = keras.Sequential(name=name)

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers with dropout and L2 regularization
        for i, neurons in enumerate(hidden_layers):
            model.add(layers.Dense(
                neurons,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f"{name}_hidden_{i+1}"
            ))
            model.add(layers.Dropout(
                dropout_rate, name=f"{name}_dropout_{i+1}"))

        # Output layer - always sigmoid for LogitBoost (even for multiclass)
        if n_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid',
                                   name=f"{name}_output"))
            loss = 'binary_crossentropy'
            # metrics = ['accuracy', 'precision', 'recall']

            metrics = ['accuracy', Precision(), Recall()]

        else:
            # Multi-class classification - use softmax but we'll handle LogitBoost logic separately
            model.add(layers.Dense(n_classes, activation='softmax',
                                   name=f"{name}_output"))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy', 'sparse_categorical_accuracy']

        # Compile model with Adam optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return model

    def create_individual_mlps(self, input_dim, n_classes):
        '''Create individual MLP models with different architectures'''
        mlps = []

        for i, config in enumerate(self.configurations):
            # Validate and complete the config with defaults
            complete_config = self.validate_and_complete_config(config)

            # Ensure unique name if not provided
            if 'name' not in config or config.get('name') is None:
                complete_config['name'] = f'mlp_model_{i}'

            model = self._create_mlp_model(
                complete_config, input_dim, n_classes)
            mlps.append(model)

            # Safe logging with get() method
            name = complete_config.get('name', f'model_{i}')
            hidden_layers = complete_config.get('hidden_layers', 'unknown')
            print(f"Created {name}: {hidden_layers} architecture")

            if self.verbose > 0:
                model.summary()

        return mlps

    def _initialize_logitboost_variables(self, X, y_encoded):
        """Initialize variables for LogitBoost algorithm"""
        n_samples = X.shape[0]

        if self.n_classes_ == 2:
            # Binary case: Initialize with log-odds of class proportions
            p1 = np.mean(y_encoded)
            p0 = 1 - p1
            # Avoid log(0) by adding small epsilon
            epsilon = 1e-15
            p1 = max(p1, epsilon)
            p0 = max(p0, epsilon)

            initial_f = np.log(p1 / p0)
            self.ensemble_predictions = np.full(n_samples, initial_f)

            # Initialize working response and weights
            prob = 1.0 / (1.0 + np.exp(-self.ensemble_predictions))
            prob = np.clip(prob, epsilon, 1 - epsilon)

            self.working_response = (y_encoded - prob) / (prob * (1 - prob))
            self.working_weights = prob * (1 - prob)

        else:
            # Multi-class case: Initialize with class proportions
            self.ensemble_predictions = np.zeros((n_samples, self.n_classes_))
            class_probs = np.bincount(y_encoded) / n_samples

            for k in range(self.n_classes_):
                self.ensemble_predictions[:, k] = np.log(
                    max(class_probs[k], 1e-15))

            # Initialize working response and weights for multi-class
            probabilities = self._softmax(self.ensemble_predictions)
            self.working_response = np.zeros((n_samples, self.n_classes_))
            self.working_weights = np.zeros((n_samples, self.n_classes_))

            for k in range(self.n_classes_):
                y_k = (y_encoded == k).astype(float)
                prob_k = probabilities[:, k]
                prob_k = np.clip(prob_k, 1e-15, 1 - 1e-15)

                self.working_response[:, k] = (y_k - prob_k) / prob_k
                self.working_weights[:, k] = prob_k * (1 - prob_k)

    def _softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _update_logitboost_variables(self, X, y_encoded, model_predictions):
        """Update working response and weights for LogitBoost"""
        epsilon = 1e-15

        if self.n_classes_ == 2:
            # Update ensemble predictions
            self.ensemble_predictions += 0.5 * model_predictions.flatten()

            # Update probabilities
            prob = 1.0 / (1.0 + np.exp(-self.ensemble_predictions))
            prob = np.clip(prob, epsilon, 1 - epsilon)

            # Update working response and weights
            self.working_response = (y_encoded - prob) / (prob * (1 - prob))
            self.working_weights = prob * (1 - prob)

        else:
            # Multi-class case
            # Update ensemble predictions
            self.ensemble_predictions += 0.5 * model_predictions

            # Update probabilities
            probabilities = self._softmax(self.ensemble_predictions)

            # Update working response and weights
            for k in range(self.n_classes_):
                y_k = (y_encoded == k).astype(float)
                prob_k = probabilities[:, k]
                prob_k = np.clip(prob_k, epsilon, 1 - epsilon)

                self.working_response[:, k] = (y_k - prob_k) / prob_k
                self.working_weights[:, k] = prob_k * (1 - prob_k)

    def _convert_predictions_to_working_response(self, model_predictions, y_encoded):
        """Convert model predictions to format suitable for LogitBoost updates"""
        if self.n_classes_ == 2:
            # For binary classification, convert probabilities to log-odds
            prob = np.clip(model_predictions.flatten(), 1e-15, 1 - 1e-15)
            return np.log(prob / (1 - prob))
        else:
            # For multi-class, convert probabilities to log scale
            prob = np.clip(model_predictions, 1e-15, 1 - 1e-15)
            return np.log(prob)

    def train_with_boosting(self, X, y):
        """Train ensemble using LogitBoost algorithm with Keras models"""
        X, y = check_X_y(X, y)

        # Encode labels if they're not already numeric
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)

        print("Training Keras MLP Ensemble with LogitBoost...")
        print(f"Training data shape: {X.shape}")
        print(f"Classes: {self.classes_}")
        print(f"Number of classes: {self.n_classes_}")

        # Initialize LogitBoost variables
        self._initialize_logitboost_variables(X, y_encoded)

        # Create individual MLPs
        self.models = self.create_individual_mlps(
            self.n_features_in_, self.n_classes_)
        self.model_weights = []
        self.history = []

        # Setup callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )

        callbacks = [early_stopping, reduce_lr]

        # Process configurations to ensure they have all required keys
        processed_configs = []
        for i, config in enumerate(self.configurations):
            processed_config = self.validate_and_complete_config(config)
            # Ensure name is set
            if 'name' not in config or config.get('name') is None:
                processed_config['name'] = f'mlp_model_{i}'
            processed_configs.append(processed_config)

        for i, (model, config) in enumerate(zip(self.models, processed_configs)):
            print(f"\nTraining Model {i+1}: {config['name']}")

            # Prepare sample weights for LogitBoost
            if self.n_classes_ == 2:
                # Use working weights as sample weights
                sample_weights_normalized = self.working_weights / \
                    np.sum(self.working_weights) * len(self.working_weights)
                y_train = y_encoded.astype(np.float32)
            else:
                # For multi-class, use average of working weights across classes
                sample_weights_normalized = np.mean(
                    self.working_weights, axis=1)
                sample_weights_normalized = sample_weights_normalized / \
                    np.sum(sample_weights_normalized) * \
                    len(sample_weights_normalized)
                y_train = y_encoded

            # Train model
            history = model.fit(
                X.astype(np.float32),
                y_train,
                sample_weight=sample_weights_normalized,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=self.verbose
            )

            self.history.append(history)

            # Get predictions for LogitBoost updates
            if self.n_classes_ == 2:
                y_pred_proba = model.predict(X.astype(np.float32), verbose=0)
                model_predictions_logit = self._convert_predictions_to_working_response(
                    y_pred_proba, y_encoded)
            else:
                y_pred_proba = model.predict(X.astype(np.float32), verbose=0)
                model_predictions_logit = self._convert_predictions_to_working_response(
                    y_pred_proba, y_encoded)

            # Calculate LogitBoost model weight based on weighted least squares fit
            if self.n_classes_ == 2:
                # Calculate correlation between predictions and working response
                weighted_corr = np.sum(
                    self.working_weights * self.working_response * model_predictions_logit.flatten())
                weighted_var = np.sum(
                    self.working_weights * model_predictions_logit.flatten() ** 2)

                if weighted_var > 1e-10:
                    model_weight = abs(weighted_corr / weighted_var)
                else:
                    model_weight = 0.1
            else:
                # For multi-class, average the weights across classes
                model_weight = 0.0
                for k in range(self.n_classes_):
                    weighted_corr = np.sum(
                        self.working_weights[:, k] * self.working_response[:, k] * model_predictions_logit[:, k])
                    weighted_var = np.sum(
                        self.working_weights[:, k] * model_predictions_logit[:, k] ** 2)

                    if weighted_var > 1e-10:
                        model_weight += abs(weighted_corr / weighted_var)

                model_weight /= self.n_classes_

            # Ensure reasonable bounds on model weight
            model_weight = max(min(model_weight, 2.0), 0.1)
            self.model_weights.append(model_weight)

            print(f"Model {i+1} Weight: {model_weight:.4f}")
            print(
                f"Model {i+1} Final Val Loss: {history.history['val_loss'][-1]:.4f}")

            # Update LogitBoost variables for next iteration
            if i < len(self.models) - 1:  # Don't update after last model
                self._update_logitboost_variables(
                    X, y_encoded, model_predictions_logit)

        # Normalize model weights
        total_weight = sum(self.model_weights)
        if total_weight > 0:
            self.model_weights = [w / total_weight for w in self.model_weights]
        else:
            self.model_weights = [1/len(self.models)] * len(self.models)

        print(
            f"\nFinal Model Weights: {[f'{w:.4f}' for w in self.model_weights]}")
        self.is_fitted = True

        return self.models

    def get_weights(self):
        """
        Get the calculated model weights after training.

        Returns:
            list: Normalized model weights for each model in the ensemble.
                 Returns None if models haven't been trained yet.

        Example:
            ensemble = MLPEnsemble()
            ensemble.train_with_boosting(X_train_balanced, y_train_balanced)
            best_weights = ensemble.get_weights()
            print(f"Model weights: {best_weights}")
        """
        if not self.is_fitted:
            print(
                "Warning: Models have not been trained yet. Call train_with_boosting() first.")
            return None

        # Return a copy to prevent accidental modification
        return self.model_weights.copy()

    def predict_individual(self, X):
        """Get predictions from all 4 individual models"""
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction")

        X = check_array(X).astype(np.float32)
        individual_predictions = []

        for i, model in enumerate(self.models):
            if self.n_classes_ == 2:
                pred_proba = model.predict(X, verbose=0)
                pred = (pred_proba > 0.5).astype(int).flatten()
                # Convert back to original labels
                pred = self.label_encoder.inverse_transform(pred)
            else:
                pred_proba = model.predict(X, verbose=0)
                pred = np.argmax(pred_proba, axis=1)
                # Convert back to original labels
                pred = self.label_encoder.inverse_transform(pred)

            individual_predictions.append(pred)

        # Shape: (n_samples, n_models)
        return np.array(individual_predictions).T

    def predict_ensemble_weighted(self, X, weights=None):
        """Make weighted ensemble predictions using optimized weights"""
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction")

        # Check if X contains pre-computed predictions or raw features
        if X.shape[1] == len(self.models):
            # X contains pre-computed individual predictions
            individual_predictions = X
            return self._combine_predictions_weighted(individual_predictions, weights)
        else:
            # X contains raw features
            X = check_array(X).astype(np.float32)

            # Use provided weights or default model weights
            if weights is None:
                weights = self.model_weights

            # Ensure weights sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)

            # Get probability predictions from each model
            ensemble_proba = np.zeros((X.shape[0], self.n_classes_))

            for i, (model, weight) in enumerate(zip(self.models, weights)):
                if self.n_classes_ == 2:
                    model_proba = model.predict(X, verbose=0)
                    # Convert to 2-class probability format
                    proba_2d = np.column_stack(
                        [1 - model_proba.flatten(), model_proba.flatten()])
                else:
                    proba_2d = model.predict(X, verbose=0)

                ensemble_proba += weight * proba_2d

            # Convert probabilities to class predictions
            if self.n_classes_ == 2:
                predictions_encoded = (ensemble_proba[:, 1] > 0.5).astype(int)
            else:
                predictions_encoded = np.argmax(ensemble_proba, axis=1)

            # Convert back to original labels
            predictions = self.label_encoder.inverse_transform(
                predictions_encoded)

            return predictions

    def predict_proba_ensemble(self, X, weights=None):
        """Return prediction probabilities for SHAP explainer"""
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction")

        # Check if X contains pre-computed predictions or raw features
        if X.shape[1] == len(self.models):
            # X contains pre-computed individual predictions
            individual_predictions = X
            return self._convert_predictions_to_proba(individual_predictions, weights)
        else:
            # X contains raw features
            X = check_array(X).astype(np.float32)

            # Use provided weights or default model weights
            if weights is None:
                weights = self.model_weights

            # Ensure weights sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)

            # Get weighted probability predictions
            ensemble_proba = np.zeros((X.shape[0], self.n_classes_))

            for i, (model, weight) in enumerate(zip(self.models, weights)):
                if self.n_classes_ == 2:
                    model_proba = model.predict(X, verbose=0)
                    # Convert to 2-class probability format
                    proba_2d = np.column_stack(
                        [1 - model_proba.flatten(), model_proba.flatten()])
                else:
                    proba_2d = model.predict(X, verbose=0)

                ensemble_proba += weight * proba_2d

            return ensemble_proba

    def get_ensemble_as_function(self, weights=None):
        """Return ensemble as a callable function for SHAP"""
        if not self.is_fitted:
            raise ValueError(
                "Models must be trained before creating ensemble function"
            )

        def ensemble_function(X):
            """Ensemble prediction function for SHAP"""
            # Convert to NumPy if it's a DataFrame
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self.predict_proba_ensemble(X, weights)

        return ensemble_function

    def predict(self, X):
        """Standard predict method using weighted ensemble"""
        return self.predict_ensemble_weighted(X)

        def predict_proba(self, X):
            """Standard predict_proba method using weighted ensemble"""
        return self.predict_proba_ensemble(X)

    def _combine_predictions_weighted(self, individual_predictions, weights=None):
        """Combine pre-computed individual predictions using weights"""
        if weights is None:
            weights = self.model_weights

        # Ensure weights sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # For each sample, compute weighted vote
        n_samples = individual_predictions.shape[0]
        ensemble_predictions = []

        for i in range(n_samples):
            # Get predictions from all models for this sample
            sample_predictions = individual_predictions[i]

            # Calculate weighted votes for each class
            class_votes = np.zeros(len(self.classes_))
            for class_idx, class_label in enumerate(self.classes_):
                for model_idx, pred in enumerate(sample_predictions):
                    if pred == class_label:
                        class_votes[class_idx] += weights[model_idx]

            # Choose class with highest weighted vote
            predicted_class = self.classes_[np.argmax(class_votes)]
            ensemble_predictions.append(predicted_class)

        return np.array(ensemble_predictions)

    def _convert_predictions_to_proba(self, individual_predictions, weights=None):
        """Convert individual predictions to ensemble probabilities"""
        if weights is None:
            weights = self.model_weights

        # Ensure weights sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        n_samples = individual_predictions.shape[0]
        ensemble_proba = np.zeros((n_samples, len(self.classes_)))

        for i in range(n_samples):
            # Get predictions from all models for this sample
            sample_predictions = individual_predictions[i]

            # Calculate weighted probabilities for each class
            for class_idx, class_label in enumerate(self.classes_):
                for model_idx, pred in enumerate(sample_predictions):
                    if pred == class_label:
                        ensemble_proba[i, class_idx] += weights[model_idx]

        # Normalize probabilities
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        ensemble_proba = ensemble_proba / row_sums

        return ensemble_proba

    def get_model_info(self):
        """Get information about the ensemble models"""
        if not self.is_fitted:
            return "Models not trained yet"

        info = {
            'n_models': len(self.models),
            'model_weights': self.model_weights,
            'configurations': [config.get('name', f'model_{i}') for i, config in enumerate(self.configurations)],
            'architectures': [config.get('hidden_layers', 'unknown') for config in self.configurations],
            'classes': self.classes_,
            'n_features': self.n_features_in_,
            'n_classes': self.n_classes_,
            'framework': 'TensorFlow/Keras',
            'boosting_algorithm': 'LogitBoost'
        }

        return info

    def get_individual_predictions_for_tsa(self, X_train, y_train, X_test=None):
        """Get individual model predictions formatted for TSA optimizer"""
        if not self.is_fitted:
            raise ValueError(
                "Models must be trained before getting predictions")

        # Get individual predictions for training set
        train_predictions = self.predict_individual(X_train)

        result = {
            'train_predictions': train_predictions,
            'y_train': y_train,
            'n_models': len(self.models),
            'model_names': [config.get('name', f'model_{i}') for i, config in enumerate(self.configurations)]
        }

        # Get test predictions if provided
        if X_test is not None:
            test_predictions = self.predict_individual(X_test)
            result['test_predictions'] = test_predictions

        return result

    def save_models(self, filepath_prefix):
        """Save individual models"""
        if not self.is_fitted:
            raise ValueError("Models must be trained before saving")

        for i, (model, config) in enumerate(zip(self.models, self.configurations)):
            model_name = config.get('name', f'model_{i}')
            model_path = f"{filepath_prefix}\\{model_name}.keras"
            model.save(model_path)
            print(f"Saved {model_name} to {model_path}")

    def load_models(self, filepath_prefix):
        """Load individual models"""
        self.models = []
        for i, config in enumerate(self.configurations):
            model_name = config.get('name', f'model_{i}')
            model_path = f"{filepath_prefix}_{model_name}.keras"
            model = keras.models.load_model(model_path)
            self.models.append(model)
            print(f"Loaded {model_name} from {model_path}")

        self.is_fitted = True
