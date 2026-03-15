import json
import numpy as np
import matplotlib.pyplot as plt
import shap

# 🔍 Load JSON file
with open('explanation_report.json', 'r') as f:
    data = json.load(f)

# 🔍 Fetch feature names and explanation info
feature_names = [item['feature'] for item in data['global_importance']]
explanations = data['local_explanations']

# 🔧 Select the sample you want to visualize — example using sample_index == 28
sample_idx = 28
explanation = next((e for e in explanations if int(
    e['sample_index']) == sample_idx), None)

if explanation is None:
    raise ValueError(f"Sample {sample_idx} not found!")

# 🔍 Parse SHAP values and feature values as NumPy arrays
shap_values = np.fromstring(explanation['shap_values'].replace(
    '[', '').replace(']', ''), sep=' ')
feature_values = np.fromstring(
    explanation['feature_values'].replace('[', '').replace(']', ''), sep=' ')
expected_value = float(explanation['expected_value'])

# ✅ Convert to float and ensure shapes align
shap_values = shap_values.astype(np.float64)
feature_values = feature_values.astype(np.float64)

# 🧮 Prepare shap.Explanation object for waterfall
shap_expl = shap.Explanation(
    values=shap_values,
    base_values=expected_value,
    data=feature_values,
    feature_names=feature_names
)

# 🎨 1️⃣ Waterfall Plot
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_expl, show=False)
plt.title(f"Waterfall Plot for sample {sample_idx}")
plt.tight_layout()
plt.savefig(f'waterfall_sample_{sample_idx}.png', dpi=200)
plt.close()

print(f"✅ Saved waterfall plot as waterfall_sample_{sample_idx}.png")

# 🎨 2️⃣ Bar Plot of top feature contributions
top_n = 10
# Sort features by absolute SHAP value
top_indices = np.argsort(np.abs(shap_values))[-top_n:]
top_features = [feature_names[i] for i in top_indices]
top_vals = shap_values[top_indices]

plt.figure(figsize=(8, 6))
plt.barh(range(len(top_indices)), top_vals, color=[
         'red' if v < 0 else 'green' for v in top_vals])
plt.yticks(range(len(top_indices)), top_features)
plt.xlabel('SHAP Value')
plt.title(f'Top {top_n} Feature Contributions (sample {sample_idx})')
plt.axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig(f'barplot_sample_{sample_idx}.png', dpi=200)
plt.close()

print(f"✅ Saved bar plot as barplot_sample_{sample_idx}.png")
