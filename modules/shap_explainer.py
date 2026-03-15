import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """SHAP (Shapley Additive Explanations) for model interpretability"""

    def __init__(self, config):
        """
        Initialize SHAP explainer with configuration from config.py

        Args:
            config: Dictionary containing SHAP configuration parameters
        """
        self.config = config
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.X_background = None
        self.model_function = None

        # Create results directories if they don't exist
        os.makedirs('results/shap_waterfall_plots', exist_ok=True)
        os.makedirs('results/shap_force_plots', exist_ok=True)
        # os.makedirs(
        #   '/content/drive/MyDrive/Research/implementation/results/shap_waterfall_plots', exist_ok=True)
        # os.makedirs(
        #   '/content/drive/MyDrive/Research/implementation/results/shap_force_plots', exist_ok=True)
        # Set SHAP configuration
        if not self.config.get('silent', False):
            shap.initjs()  # Initialize JavaScript for force plots

    def create_explainer(self, model_function, X_background, feature_names=None):
        """
        Create SHAP explainer based on configuration.

        Args:
            model_function: Callable ensemble model function
            X_background: Background data for SHAP explainer
            feature_names: List of feature names for interpretability
        """
        print(f"Creating SHAP {self.config['explainer_type']} explainer...")

        self.model_function = model_function
        self.feature_names = feature_names

        # ✅ Convert background to NumPy
        if isinstance(X_background, pd.DataFrame):
            X_background = X_background.values

        # Sample background data if needed
        n_background = min(
            self.config['background_samples'], len(X_background))
        if n_background < len(X_background):
            background_idx = np.random.choice(
                len(X_background), n_background, replace=False
            )
            self.X_background = X_background[background_idx]
        else:
            self.X_background = X_background

        # Create appropriate explainer
        explainer_type = self.config['explainer_type'].lower()

        if explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(
                model_function, self.X_background)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                model_function,
                self.X_background,
                feature_perturbation=self.config.get(
                    'feature_perturbation', 'interventional')
            )
        elif explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(model_function)
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(
                model_function, self.X_background)
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")

        print(
            f"SHAP explainer created successfully with {len(self.X_background)} background samples"
        )

    def calculate_shap_values(self, X_explain):
        """
        Calculate SHAP values for given dataset

        Args:
            X_explain: Dataset to explain

        Returns:
            shap_values: SHAP values array
        """
        if self.explainer is None:
            raise ValueError(
                "Explainer not created. Call create_explainer() first.")

        print(f"Calculating SHAP values for {len(X_explain)} samples...")

        # Calculate SHAP values based on explainer type
        if self.config['explainer_type'].lower() == 'kernel':
            # For KernelExplainer, use max_evals parameter
            self.shap_values = self.explainer.shap_values(
                X_explain,
                nsamples=self.config.get('max_evals', 1000),
                silent=self.config.get('silent', False)
            )
        else:
            # For other explainers
            self.shap_values = self.explainer.shap_values(X_explain)

        # Handle different output formats
        if isinstance(self.shap_values, list):
            # Multi-class output - take the positive class for binary classification
            self.shap_values = self.shap_values[1] if len(
                self.shap_values) == 2 else self.shap_values[0]

        print(f"SHAP values calculated. Shape: {self.shap_values.shape}")

        # Save SHAP values if configured
        if self.config.get('save_shap_values', True):
            self.save_shap_values()

        return self.shap_values

    def global_feature_importance(self, save_plot=None):
        """
        Calculate and visualize global feature importance

        Args:
            save_plot: Path to save plot (optional)

        Returns:
            importance_df: DataFrame with feature importance scores
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first.")

        # Debug: Print SHAP values shape and type
        print(f"SHAP values shape: {self.shap_values.shape}")
        print(f"SHAP values type: {type(self.shap_values)}")

        # Handle different SHAP value formats
        shap_vals = self.shap_values

        # If SHAP values are 3D (common with some explainers), take the appropriate slice
        if len(shap_vals.shape) == 3:
            print(f"3D SHAP values detected: {shap_vals.shape}")
            # For binary classification, typically take the positive class (index 1)
            if shap_vals.shape[2] == 2:
                shap_vals = shap_vals[:, :, 1]  # Take positive class
            else:
                shap_vals = shap_vals[:, :, 0]  # Take first class
            print(f"After slicing: {shap_vals.shape}")

        # Ensure we have 2D array (samples x features)
        if len(shap_vals.shape) != 2:
            raise ValueError(
                f"Unexpected SHAP values shape: {shap_vals.shape}")

        # Calculate mean absolute SHAP values for global importance
        mean_shap_values = np.abs(shap_vals).mean(axis=0)

        # Ensure mean_shap_values is 1D
        if len(mean_shap_values.shape) > 1:
            mean_shap_values = mean_shap_values.flatten()

        print(f"Mean SHAP values shape: {mean_shap_values.shape}")

        # Create importance DataFrame
        if self.feature_names is not None:
            if len(self.feature_names) != len(mean_shap_values):
                print(
                    f"Warning: Feature names length ({len(self.feature_names)}) doesn't match SHAP values ({len(mean_shap_values)})")
                # Use available feature names or generate names
                if len(self.feature_names) > len(mean_shap_values):
                    feature_names = self.feature_names[:len(mean_shap_values)]
                else:
                    feature_names = self.feature_names + \
                        [f'Feature_{i}' for i in range(
                            len(self.feature_names), len(mean_shap_values))]
            else:
                feature_names = self.feature_names

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap_values
            })
        else:
            importance_df = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(len(mean_shap_values))],
                'importance': mean_shap_values
            })

        # Sort by importance
        importance_df = importance_df.sort_values(
            'importance', ascending=False)

        # Create and save summary plot
        try:
            plt.figure(figsize=self.config['plot_size'])
            shap.summary_plot(
                shap_vals,
                features=None,  # Don't pass features to avoid dimension mismatch
                feature_names=self.feature_names,
                max_display=self.config.get('max_display', 20),
                show=False
            )

            if save_plot or self.config.get('save_plots', True):
                save_path = save_plot or 'results/shap_summary_plot.png'
                # save_path = save_plot or '/content/drive/MyDrive/Research/implementation/results/shap_summary_plot.png'
                plt.savefig(save_path, dpi=self.config.get(
                    'dpi', 300), bbox_inches='tight')
                print(f"Global importance plot saved to {save_path}")

            plt.close()
        except Exception as e:
            print(f"Warning: Could not create summary plot: {str(e)}")
            plt.close()

        # Create bar plot
        try:
            plt.figure(figsize=self.config['plot_size'])
            top_features = importance_df.head(
                self.config.get('max_display', 20))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Global Feature Importance')
            plt.gca().invert_yaxis()

            if self.config.get('save_plots', True):
                plt.savefig('results/feature_importance.png',
                            dpi=self.config.get('dpi', 300), bbox_inches='tight')
                # plt.savefig('/content/drive/MyDrive/Research/implementation/results/feature_importance.png',
                #           dpi=self.config.get('dpi', 300), bbox_inches='tight')
                print(
                    "Feature importance bar plot saved to results/feature_importance.png")

            plt.close()
        except Exception as e:
            print(f"Warning: Could not create bar plot: {str(e)}")
            plt.close()

        return importance_df

    def local_explanations(self, X_sample, y_sample=None, sample_indices=None):
        """
        Generate local explanations for individual predictions

        Args:
            X_sample: Sample data to explain
            y_sample: True labels (optional)
            sample_indices: Specific indices to explain (optional)

        Returns:
            explanations: List of individual explanation results
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first.")

        n_explanations = min(
            self.config.get('n_individual_explanations', 10),
            len(X_sample)
        )

        if sample_indices is None:
            sample_indices = np.random.choice(
                len(X_sample), n_explanations, replace=False)

        # Convert sample_indices to list for easier handling
        sample_indices = list(sample_indices)

        explanations = []
        explanation_type = self.config.get(
            'individual_explanation_type', 'force')

        print(f"Generating {len(sample_indices)} local explanations...")

        for i, idx in enumerate(sample_indices):
            try:
                # Calculate SHAP values for this specific sample
                sample_data = X_sample[idx:idx+1]
                sample_shap_values = self.explainer.shap_values(sample_data)

                # Debug: Print shape information
                print(
                    f"Sample {idx} SHAP values shape: {np.array(sample_shap_values).shape}")

                # Handle different SHAP value formats
                if isinstance(sample_shap_values, list):
                    # Multi-class output - take the positive class for binary classification
                    if len(sample_shap_values) == 2:
                        # Positive class, first sample
                        sample_shap = sample_shap_values[1][0]
                    else:
                        # First class, first sample
                        sample_shap = sample_shap_values[0][0]
                else:
                    # Handle numpy array
                    if len(sample_shap_values.shape) == 3:
                        # 3D: (samples, features, classes)
                        if sample_shap_values.shape[2] == 2:
                            # Positive class
                            sample_shap = sample_shap_values[0, :, 1]
                        else:
                            # First class
                            sample_shap = sample_shap_values[0, :, 0]
                    elif len(sample_shap_values.shape) == 2:
                        if sample_shap_values.shape[0] == 1:
                            # 2D: (1, features) - single sample
                            sample_shap = sample_shap_values[0]
                        elif sample_shap_values.shape[1] == 2:
                            # 2D: (features, classes) - single sample with classes
                            # Positive class
                            sample_shap = sample_shap_values[:, 1]
                        else:
                            # 2D: (features,) - already correct format
                            sample_shap = sample_shap_values
                    else:
                        # 1D: already correct
                        sample_shap = sample_shap_values

                # Ensure sample_shap is 1D and float64
                if len(sample_shap.shape) > 1:
                    sample_shap = sample_shap.flatten()
                sample_shap = sample_shap.astype(np.float64)

                print(f"Final sample {idx} SHAP shape: {sample_shap.shape}")

                # Calculate expected value (baseline)
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    if len(expected_value) == 2:
                        # Positive class for binary
                        expected_value = float(expected_value[1])
                    else:
                        expected_value = float(expected_value[0]) if len(
                            expected_value) > 0 else 0.0
                elif isinstance(expected_value, list):
                    if len(expected_value) == 2:
                        # Positive class for binary
                        expected_value = float(expected_value[1])
                    else:
                        expected_value = float(expected_value[0]) if len(
                            expected_value) > 0 else 0.0
                else:
                    expected_value = float(expected_value)

                # **FIX: Ensure sample data is numeric and properly formatted**
                # sample_features = sample_data[0] if len(sample_data) > 0 else None
                # ✅ FIX: Get first row's feature vector
                if isinstance(sample_data, pd.DataFrame):
                    sample_features = sample_data.iloc[0].values if len(
                        sample_data) > 0 else None
                else:
                    sample_features = sample_data[0] if len(
                        sample_data) > 0 else None

                # Convert to numeric, handling any string values
                if sample_features is not None:
                    # Create a copy to avoid modifying original data
                    sample_features_clean = np.array(
                        sample_features, dtype=object)

                    # Convert each feature to float, handling non-numeric values
                    # Changed variable name to avoid conflict
                    for j in range(len(sample_features_clean)):
                        try:
                            # Try to convert to float
                            sample_features_clean[j] = float(
                                sample_features_clean[j])
                        except (ValueError, TypeError):
                            # If conversion fails, use 0 or a default value
                            print(
                                f"Warning: Non-numeric value found in feature {j}: {sample_features_clean[j]}")
                            sample_features_clean[j] = 0.0

                    # Convert final array to float64
                    sample_features_clean = sample_features_clean.astype(
                        np.float64)
                else:
                    sample_features_clean = np.array([])

                # Create explanation based on type
                if explanation_type == 'waterfall':
                    plt.figure(figsize=self.config['plot_size'])

                    try:
                        # **FIX: Create SHAP Explanation object with clean numeric data**
                        explanation = shap.Explanation(
                            values=sample_shap,                     # Already float64
                            base_values=expected_value,             # Already float
                            data=sample_features_clean,             # Clean numeric data
                            feature_names=self.feature_names
                        )

                        shap.waterfall_plot(explanation, show=False)

                        if self.config.get('save_plots', True):
                            save_path = f'results/shap_waterfall_plots/waterfall_sample_{idx}.png'
                            # save_path = f'/content/drive/MyDrive/Research/implementation/results/shap_waterfall_plots/waterfall_sample_{idx}.png'
                            plt.savefig(save_path, dpi=self.config.get(
                                'dpi', 300), bbox_inches='tight')
                            print(f"Waterfall plot saved for sample {idx}")

                    except Exception as plot_error:
                        print(
                            f"Error creating waterfall plot for sample {idx}: {str(plot_error)}")
                        # Create alternative bar plot
                        plt.clf()  # Clear the figure

                        # Create simple bar plot as fallback
                        feature_names_display = self.feature_names if self.feature_names else [
                            f'Feature_{k}' for k in range(len(sample_shap))]

                        # Sort features by absolute SHAP value for better visualization
                        sorted_indices = np.argsort(
                            np.abs(sample_shap))[-10:]  # Top 10 features

                        plt.barh(range(len(sorted_indices)),
                                 sample_shap[sorted_indices])
                        plt.yticks(range(len(sorted_indices)),
                                   [feature_names_display[k] for k in sorted_indices])
                        plt.xlabel('SHAP Value')
                        plt.title(f'Feature Contributions - Sample {idx}')
                        plt.axvline(x=0, color='black',
                                    linestyle='--', alpha=0.5)

                        if self.config.get('save_plots', True):
                            save_path = f'results/shap_waterfall_plots/bar_sample_{idx}.png'
                            # save_path = f'/content/drive/MyDrive/Research/implementation/results/shap_waterfall_plots/bar_sample_{idx}.png'
                            plt.savefig(save_path, dpi=self.config.get(
                                'dpi', 300), bbox_inches='tight')
                            print(
                                f"Alternative bar plot saved for sample {idx}")

                    plt.close()

                elif explanation_type == 'force':
                    # Force plot (saved as HTML)
                    try:
                        force_plot = shap.force_plot(
                            expected_value,
                            sample_shap,
                            sample_features_clean,
                            feature_names=self.feature_names,
                            show=False
                        )

                        if self.config.get('save_plots', True):
                            save_path = f'results/shap_force_plots/force_sample_{idx}.png'
                            # save_path = f'/content/drive/MyDrive/Research/implementation/results/shap_force_plots/force_sample_{idx}.html'
                            shap.save_html(save_path, force_plot)
                            print(f"Force plot saved for sample {idx}")

                    except Exception as force_error:
                        print(
                            f"Error creating force plot for sample {idx}: {str(force_error)}")

                # **FIX: Use idx directly for y_sample access**
                true_label = None
                if y_sample is not None:
                    try:
                        # Try direct index access first
                        if hasattr(y_sample, 'iloc'):
                            # If it's a pandas Series, use iloc for positional indexing
                            true_label = y_sample.iloc[idx]
                        else:
                            # If it's a numpy array or list, use direct indexing
                            true_label = y_sample[idx]
                    except (IndexError, KeyError) as e:
                        print(
                            f"Warning: Could not access true label for sample {idx}: {e}")
                        true_label = None

                # Store explanation data
                explanation_data = {
                    'sample_index': idx,
                    'shap_values': sample_shap,
                    'feature_values': sample_features_clean,
                    'expected_value': expected_value,
                    'prediction': self.model_function(sample_data)[0] if hasattr(self, 'model_function') else None,
                    'true_label': true_label
                }

                explanations.append(explanation_data)

            except Exception as e:
                print(
                    f"Error generating explanation for sample {idx}: {str(e)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                continue

        print(f"Generated {len(explanations)} local explanations")
        return explanations

    def feature_interaction_analysis(self, X_data, max_interactions=10):
        """
        Analyze feature interactions using SHAP interaction values

        Args:
            X_data: Data for interaction analysis
            max_interactions: Maximum number of interactions to analyze

        Returns:
            interaction_results: Dictionary containing interaction analysis results
        """
        print("Analyzing feature interactions...")

        try:
            # Calculate interaction values (if supported by explainer)
            if hasattr(self.explainer, 'shap_interaction_values'):
                interaction_values = self.explainer.shap_interaction_values(
                    X_data[:100])  # Limit for performance

                # Calculate mean interaction strengths
                mean_interactions = np.abs(interaction_values).mean(axis=0)

                # Find top interactions
                n_features = mean_interactions.shape[0]
                interactions = []

                for i in range(n_features):
                    for j in range(i+1, n_features):
                        interaction_strength = mean_interactions[i, j]
                        feature_i = self.feature_names[
                            i] if self.feature_names else f'Feature_{i}'
                        feature_j = self.feature_names[
                            j] if self.feature_names else f'Feature_{j}'

                        interactions.append({
                            'feature_1': feature_i,
                            'feature_2': feature_j,
                            'interaction_strength': interaction_strength
                        })

                # Sort by interaction strength
                interactions = sorted(
                    interactions, key=lambda x: x['interaction_strength'], reverse=True)
                top_interactions = interactions[:max_interactions]

                # Create interaction heatmap
                plt.figure(figsize=self.config['plot_size'])
                feature_names_display = self.feature_names[:20] if self.feature_names else [
                    f'F{i}' for i in range(20)]
                display_interactions = mean_interactions[:20, :20]

                sns.heatmap(
                    display_interactions,
                    xticklabels=feature_names_display,
                    yticklabels=feature_names_display,
                    annot=False,
                    cmap='coolwarm',
                    center=0
                )
                plt.title('Feature Interaction Heatmap')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)

                if self.config.get('save_plots', True):
                    plt.savefig('results/feature_interactions.png',
                                dpi=self.config.get('dpi', 300), bbox_inches='tight')
                    # plt.savefig('/content/drive/MyDrive/Research/implementation/results/feature_interactions.png',
                    #            dpi=self.config.get('dpi', 300), bbox_inches='tight')
                    print(
                        "Feature interaction heatmap saved to results/feature_interactions.png")
                    # print(
                    #   "Feature interaction heatmap saved to /content/drive/MyDrive/Research/implementation/results/feature_interactions.png")

                plt.close()

                return {
                    'top_interactions': top_interactions,
                    'interaction_matrix': mean_interactions,
                    'interaction_values': interaction_values
                }

            else:
                print("Interaction values not supported by current explainer type")
                return {'message': 'Interaction analysis not supported'}

        except Exception as e:
            print(f"Error in interaction analysis: {str(e)}")
            return {'error': str(e)}

    def dependence_plots(self, feature_names, X_data):
        """
        Create SHAP dependence plots showing feature effects

        Args:
            feature_names: List of feature names to create plots for
            X_data: Data for dependence plots
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first.")

        print(
            f"Creating dependence plots for {len(feature_names)} features...")

        for feature_name in feature_names:
            try:
                # Find feature index
                if self.feature_names:
                    if feature_name in self.feature_names:
                        feature_idx = self.feature_names.index(feature_name)
                    else:
                        print(
                            f"Feature '{feature_name}' not found in feature names")
                        continue
                else:
                    # Assume feature_name is an index
                    feature_idx = int(feature_name) if str(
                        feature_name).isdigit() else 0

                # Create dependence plot
                plt.figure(figsize=self.config['plot_size'])
                shap.dependence_plot(
                    feature_idx,
                    self.shap_values,
                    X_data,
                    feature_names=self.feature_names,
                    show=False
                )

                if self.config.get('save_plots', True):
                    safe_name = str(feature_name).replace(
                        '/', '_').replace(' ', '_')
                    save_path = f'results/dependence_plot_{safe_name}.png'
                    # save_path = f'/content/drive/MyDrive/Research/implementation/results/dependence_plot_{safe_name}.png'
                    plt.savefig(save_path, dpi=self.config.get(
                        'dpi', 300), bbox_inches='tight')
                    print(f"Dependence plot saved for {feature_name}")

                plt.close()

            except Exception as e:
                print(
                    f"Error creating dependence plot for {feature_name}: {str(e)}")
                continue

    def clustering_explanations(self, X_data, n_clusters=5):
        """
        Cluster similar explanations together

        Args:
            X_data: Data for clustering
            n_clusters: Number of clusters

        Returns:
            cluster_results: Dictionary containing clustering results
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first.")

        print(f"Clustering explanations into {n_clusters} groups...")

        try:
            # Standardize SHAP values for clustering
            scaler = StandardScaler()
            shap_scaled = scaler.fit_transform(self.shap_values)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(shap_scaled)

            # Analyze clusters
            cluster_results = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_shap = self.shap_values[cluster_mask]
                cluster_data = X_data[cluster_mask]

                # Calculate cluster statistics
                mean_shap = cluster_shap.mean(axis=0)
                std_shap = cluster_shap.std(axis=0)

                cluster_results[cluster_id] = {
                    'size': np.sum(cluster_mask),
                    'mean_shap': mean_shap,
                    'std_shap': std_shap,
                    'representative_idx': np.argmin(np.sum((cluster_shap - mean_shap)**2, axis=1))
                }

            # Create cluster visualization
            plt.figure(figsize=self.config['plot_size'])

            # Use PCA for 2D visualization if needed
            if self.shap_values.shape[1] > 2:
                pca = PCA(n_components=2)
                shap_2d = pca.fit_transform(self.shap_values)
            else:
                shap_2d = self.shap_values

            scatter = plt.scatter(
                shap_2d[:, 0], shap_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
            plt.xlabel('SHAP Component 1')
            plt.ylabel('SHAP Component 2')
            plt.title(f'SHAP Value Clusters (n={n_clusters})')

            if self.config.get('save_plots', True):
                plt.savefig('results/shap_clusters.png',
                            dpi=self.config.get('dpi', 300), bbox_inches='tight')
                # plt.savefig('/content/drive/MyDrive/Research/implementation/results/shap_clusters.png',
                #           dpi=self.config.get('dpi', 300), bbox_inches='tight')
                print("Clustering plot saved to results/shap_clusters.png")
                # print("Clustering plot saved to /content/drive/MyDrive/Research/implementation/results/shap_clusters.png")

            plt.close()

            return cluster_results

        except Exception as e:
            print(f"Error in clustering analysis: {str(e)}")
            return {'error': str(e)}

    def save_shap_values(self, filepath='results/shap_values.pkl'):
        # def save_shap_values(self, filepath='/content/drive/MyDrive/Research/implementation/results/shap_values.pkl'):
        """
        Save calculated SHAP values to file

        Args:
            filepath: Path to save SHAP values
        """
        if self.shap_values is None:
            print("No SHAP values to save")
            return

        try:
            save_data = {
                'shap_values': self.shap_values,
                'feature_names': self.feature_names,
                'config': self.config,
                'explainer_type': self.config['explainer_type']
            }

            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"SHAP values saved to {filepath}")

        except Exception as e:
            print(f"Error saving SHAP values: {str(e)}")

    # def load_shap_values(self, filepath='results/shap_values.pkl'):
    def load_shap_values(self, filepath='/content/drive/MyDrive/Research/implementation/results/shap_values.pkl'):
        """
        Load previously calculated SHAP values

        Args:
            filepath: Path to load SHAP values from
        """
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.shap_values = save_data['shap_values']
            self.feature_names = save_data.get('feature_names')

            print(f"SHAP values loaded from {filepath}")
            print(f"Shape: {self.shap_values.shape}")

        except Exception as e:
            print(f"Error loading SHAP values: {str(e)}")

    def generate_explanation_report(self, X_data, y_data=None):
        """
        Generate comprehensive explanation report

        Args:
            X_data: Data for explanation
            y_data: True labels (optional)
        """
        print("Generating comprehensive explanation report...")

        try:
            # Calculate global importance if not done
            importance_df = self.global_feature_importance()

            # Generate local explanations
            local_explanations = self.local_explanations(X_data, y_data)

            # Feature interactions
            interaction_results = self.feature_interaction_analysis(X_data)

            # Create report
            report = {
                'summary': {
                    'n_samples_explained': len(X_data),
                    'n_features': self.shap_values.shape[1] if self.shap_values is not None else 0,
                    'explainer_type': self.config['explainer_type'],
                    'background_samples': self.config['background_samples']
                },
                'global_importance': importance_df.to_dict('records'),
                'local_explanations': local_explanations,
                'interactions': interaction_results
            }

            # Save report as JSON
            import json
            with open('results/explanation_report.json', 'w') as f:
                # with open('/content/drive/MyDrive/Research/implementation/results/explanation_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(
                "Comprehensive explanation report saved to results/explanation_report.json")

            return report

        except Exception as e:
            print(f"Error generating explanation report: {str(e)}")
            return {'error': str(e)}

    def compare_predictions(self, X_sample1, X_sample2):
        """
        Compare SHAP explanations between two samples

        Args:
            X_sample1: First sample
            X_sample2: Second sample

        Returns:
            comparison: Dictionary containing comparison results
        """
        if self.explainer is None:
            raise ValueError(
                "Explainer not created. Call create_explainer() first.")

        try:
            # Calculate SHAP values for both samples
            shap1 = self.explainer.shap_values(X_sample1.reshape(1, -1))
            shap2 = self.explainer.shap_values(X_sample2.reshape(1, -1))

            # Handle list output
            if isinstance(shap1, list):
                shap1 = shap1[1] if len(shap1) == 2 else shap1[0]
            if isinstance(shap2, list):
                shap2 = shap2[1] if len(shap2) == 2 else shap2[0]

            shap1, shap2 = shap1[0], shap2[0]

            # Calculate differences
            shap_diff = shap2 - shap1
            feature_diff = X_sample2 - X_sample1

            # Create comparison visualization
            plt.figure(figsize=self.config['plot_size'])

            # Plot SHAP differences
            feature_names_display = self.feature_names if self.feature_names else [
                f'F{i}' for i in range(len(shap_diff))]

            plt.barh(range(len(shap_diff)), shap_diff)
            plt.yticks(range(len(shap_diff)), feature_names_display)
            plt.xlabel('SHAP Value Difference (Sample 2 - Sample 1)')
            plt.title('SHAP Value Comparison Between Samples')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

            if self.config.get('save_plots', True):
                plt.savefig('results/shap_comparison.png',
                            dpi=self.config.get('dpi', 300), bbox_inches='tight')
                # plt.savefig('/content/drive/MyDrive/Research/implementation/results/shap_comparison.png',
                #           dpi=self.config.get('dpi', 300), bbox_inches='tight')
                print("SHAP comparison plot saved to results/shap_comparison.png")
                # print(
                #    "SHAP comparison plot saved to /content/drive/MyDrive/Research/implementation/results/shap_comparison.png")

            plt.close()

            return {
                'shap_values_1': shap1,
                'shap_values_2': shap2,
                'shap_difference': shap_diff,
                'feature_difference': feature_diff,
                'most_different_features': np.argsort(np.abs(shap_diff))[-5:][::-1]
            }

        except Exception as e:
            print(f"Error comparing predictions: {str(e)}")
            return {'error': str(e)}

    def plot_shap_values(self, plot_type='summary', feature_names=None):
        """
        Create various SHAP visualization plots

        Args:
            plot_type: Type of plot ('summary', 'bar', 'violin', 'heatmap')
            feature_names: Feature names for plotting
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first.")

        feature_names = feature_names or self.feature_names

        plt.figure(figsize=self.config['plot_size'])

        try:
            if plot_type == 'summary':
                shap.summary_plot(
                    self.shap_values,
                    feature_names=feature_names,
                    max_display=self.config.get('max_display', 20),
                    show=False
                )
            elif plot_type == 'bar':
                shap.summary_plot(
                    self.shap_values,
                    feature_names=feature_names,
                    plot_type='bar',
                    max_display=self.config.get('max_display', 20),
                    show=False
                )
            elif plot_type == 'violin':
                shap.summary_plot(
                    self.shap_values,
                    feature_names=feature_names,
                    plot_type='violin',
                    max_display=self.config.get('max_display', 20),
                    show=False
                )
            elif plot_type == 'heatmap':
                # Create heatmap of SHAP values
                shap_df = pd.DataFrame(self.shap_values, columns=feature_names)
                sns.heatmap(shap_df.T, cmap='coolwarm', center=0, cbar=True)
                plt.title('SHAP Values Heatmap')

            if self.config.get('save_plots', True):
                save_path = f'results/shap_{plot_type}_plot.png'
                # save_path = f'/content/drive/MyDrive/Research/implementation/results/shap_{plot_type}_plot.png'
                plt.savefig(save_path, dpi=self.config.get(
                    'dpi', 300), bbox_inches='tight')
                print(f"SHAP {plot_type} plot saved to {save_path}")

            plt.close()

        except Exception as e:
            print(f"Error creating {plot_type} plot: {str(e)}")
            plt.close()
