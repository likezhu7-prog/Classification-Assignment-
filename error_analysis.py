#!/usr/bin/env python3
"""
Error Analysis + SHAP Analysis for Cancer Classification

Core Story: Can the model still classify cancer types without disease names?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap

# =============================================================================
# Part 1: Import Libraries
# =============================================================================
print("=" * 60)
print("Part 1: Importing Libraries")
print("=" * 60)

print("Libraries imported successfully!")

# =============================================================================
# Part 2: Load Data
# =============================================================================
print("\n" + "=" * 60)
print("Part 2: Loading Data")
print("=" * 60)

# Load original data (with disease names)
df_original = pd.read_csv("merged_cancer.csv")
print(f"Original data shape: {df_original.shape}")

# Load data with keywords removed
df_no_keyword = pd.read_csv("merged_cancer_no_keywords.csv")
print(f"No-keyword data shape: {df_no_keyword.shape}")

# Define label names 
label_names = {
    0: "Colon Cancer",
    1: "Liver Cancer",
    2: "Lung Cancer",
    3: "Stomach Cancer",
    4: "Thyroid Cancer"
}

# Extract features and labels
X_orig = df_original["cleaned_text"].astype(str)
X_no_kw = df_no_keyword["cleaned_text"].astype(str)
y = df_original["label"]

print(f"Data loaded: {len(X_orig)} documents, {y.nunique()} classes")

# =============================================================================
# Part 3: Split Train/Test Sets
# =============================================================================
print("\n" + "=" * 60)
print("Part 3: Splitting Data")
print("=" * 60)

# Use same random_state 
X_train, X_test, y_train, y_test = train_test_split(
    X_orig, y, test_size=0.2, stratify=y, random_state=42
)

# For no-keyword data, use same indices
X_train_no_kw = X_no_kw.iloc[X_train.index]
X_test_no_kw = X_no_kw.iloc[X_test.index]

# Verify index alignment
assert X_train_no_kw.index.equals(X_train.index), "Index mismatch!"
assert X_test_no_kw.index.equals(X_test.index), "Index mismatch!"

print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
print("Index verification: PASSED")

# =============================================================================
# Part 4: Define TF-IDF Pipeline Function
# =============================================================================
print("\n" + "=" * 60)
print("Part 4: Defining Pipeline")
print("=" * 60)


def make_tfidf_pipeline(clf):
    """
    Create TF-IDF + Classifier Pipeline
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", clf)
    ])


print("Pipeline function defined!")

# =============================================================================
# Part 5: Train Original Model (with keywords)
# =============================================================================
print("\n" + "=" * 60)
print("Part 5: Training Original Model")
print("=" * 60)

# Create champion model (SVM + TF-IDF)
pipeline_orig = make_tfidf_pipeline(LinearSVC())

# Train model
print("Training original model (with keywords)...")
pipeline_orig.fit(X_train, y_train)

# Predict
y_pred_orig = pipeline_orig.predict(X_test)

# Calculate accuracy
accuracy_original = accuracy_score(y_test, y_pred_orig)
print(f"\nOriginal model training completed!")
print(f"Test accuracy: {accuracy_original:.4f} ({accuracy_original*100:.2f}%)")

# =============================================================================
# Part 6: Error Analysis - Confusion Matrix
# =============================================================================
print("\n" + "=" * 60)
print("Part 6: Confusion Matrix")
print("=" * 60)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_orig)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[label_names[i] for i in range(5)],
    yticklabels=[label_names[i] for i in range(5)]
)
plt.title('Confusion Matrix - Original Data (With Keywords)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_original.png', dpi=150, bbox_inches='tight')
plt.show()

# Print classification report
print("\nClassification Report (Original Data):")
print(classification_report(
    y_test, y_pred_orig,
    target_names=[label_names[i] for i in range(5)]
))

# =============================================================================
# Part 7: Error Analysis - Misclassified Samples
# =============================================================================
print("\n" + "=" * 60)
print("Part 7: Error Analysis")
print("=" * 60)

# Find misclassified samples
error_mask = y_test != y_pred_orig
errors_X = X_test[error_mask]
errors_y_true = y_test[error_mask]
errors_y_pred = y_pred_orig[error_mask]

print(f"Misclassified samples: {len(errors_X)} / {len(X_test)} ({len(errors_X)/len(X_test)*100:.1f}%)")

# Count confusion pairs
print("\nClass Confusion Statistics:")
for true_label in range(5):
    for pred_label in range(5):
        count = ((errors_y_true == true_label) & (errors_y_pred == pred_label)).sum()
        if count > 0:
            print(f"   {label_names[true_label]} -> {label_names[pred_label]}: {count} times")

# =============================================================================
# Part 8: SHAP Analysis (Original Data)
# =============================================================================
print("\n" + "=" * 60)
print("Part 8: SHAP Analysis")
print("=" * 60)

# Extract TF-IDF vectorizer and model
vectorizer_orig = pipeline_orig.named_steps["tfidf"]
clf_orig = pipeline_orig.named_steps["clf"]

# Transform test set
X_test_tfidf_orig = vectorizer_orig.transform(X_test)
print(f"TF-IDF feature matrix shape: {X_test_tfidf_orig.shape}")

# Get feature names
feature_names = vectorizer_orig.get_feature_names_out()

# Create SHAP Explainer
print("\nCalculating SHAP values (this may take a moment)...")
explainer_orig = shap.LinearExplainer(clf_orig, X_test_tfidf_orig)
shap_values_orig = explainer_orig.shap_values(X_test_tfidf_orig)

# =============================================================================
# Part 8: SHAP Analysis
# =============================================================================
print("\n" + "=" * 60)
print("Part 8: SHAP Analysis")
print("=" * 60)

# Debug: check shap_values shape
print(f"shap_values_orig shape: {shap_values_orig.shape}")

# -----------------------------------------------------------------------------
# Part 8a: SHAP Summary Plot
# -----------------------------------------------------------------------------
print("\nGenerating SHAP Summary Plot...")
plt.figure(figsize=(16, 12))
shap.summary_plot(
    shap_values_orig, X_test_tfidf_orig,
    feature_names=feature_names,
    show=False,
    max_display=20
)
plt.title('SHAP Summary Plot - Original Data (All Classes)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_summary_original.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP Summary Plot saved: shap_summary_original.png")

# -----------------------------------------------------------------------------
# Part 8b: SHAP Feature Importance (Simple Bar Chart)
# -----------------------------------------------------------------------------
print("\nGenerating SHAP Feature Importance...")

# Calculate mean absolute SHAP value for each feature
# shap_values_orig shape: (n_samples, n_features, n_classes)
mean_abs_shap = np.abs(shap_values_orig).mean(axis=0)  # Average over samples
mean_abs_shap = np.mean(mean_abs_shap, axis=1)  # Average over classes

print(f"mean_abs_shap shape: {mean_abs_shap.shape}")

# Get top 20 features
top_n = 20
top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_values = mean_abs_shap[top_indices]

# Plot feature importance
plt.figure(figsize=(12, 8))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
y_pos = np.arange(top_n)
bars = plt.barh(y_pos, top_values[::-1], color=colors)
plt.yticks(y_pos, top_features[::-1])
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('SHAP Feature Importance (Top 20)', fontsize=14, pad=20)

# Add value labels
for bar, val in zip(bars, top_values[::-1]):
    plt.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("SHAP Feature Importance saved: shap_feature_importance.png")

print(f"\nTop 10 Most Important Features:")
for i in range(min(10, len(top_features))):
    print(f"  {i+1}. {top_features[i]}: {top_values[i]:.4f}")

print("\nSHAP analysis completed!")

# =============================================================================
# Part 9: Top Features per Class
# =============================================================================
print("\n" + "=" * 60)
print("Part 9: Top Features per Class")
print("=" * 60)

print("\nTop Features per Class (SHAP values):\n")

for label in range(5):
    # Calculate mean SHAP value for this class
    mean_shap = np.mean(shap_values_orig[:, :, label], axis=0)

    # Get positive features (promote this class)
    top_positive_idx = np.argsort(mean_shap)[-5:][::-1]
    # Get negative features (prevent this class)
    top_negative_idx = np.argsort(mean_shap)[:5]

    print(f"=== {label_names[label]} ===")
    print(f"  Positive features (promote prediction):")
    for idx in top_positive_idx:
        print(f"    - {feature_names[idx]}: {mean_shap[idx]:.4f}")
    print(f"  Negative features (prevent prediction):")
    for idx in top_negative_idx:
        print(f"    - {feature_names[idx]}: {mean_shap[idx]:.4f}")
    print()

# =============================================================================
# Part 10: Remove Keywords Experiment (Core Story)
# =============================================================================
print("\n" + "=" * 60)
print("Part 10: Remove Keywords Experiment (Core Story)")
print("=" * 60)

# Train model on data without keywords
pipeline_no_kw = make_tfidf_pipeline(LinearSVC())

print("Training model without keywords...")
pipeline_no_kw.fit(X_train_no_kw, y_train)

# Predict
y_pred_no_kw = pipeline_no_kw.predict(X_test_no_kw)

# Calculate accuracy
accuracy_no_keyword = accuracy_score(y_test, y_pred_no_kw)
accuracy_drop = accuracy_original - accuracy_no_keyword
accuracy_drop_percent = (accuracy_drop / accuracy_original) * 100

print(f"\nNo-keyword model training completed!")
print(f"   Original accuracy (with keywords):    {accuracy_original:.4f} ({accuracy_original*100:.2f}%)")
print(f"   After removing keywords:              {accuracy_no_keyword:.4f} ({accuracy_no_keyword*100:.2f}%)")
print(f"   Accuracy drop:                        {accuracy_drop:.4f} ({accuracy_drop_percent:.1f}%)")

# =============================================================================
# Part 11: Accuracy Comparison Visualization
# =============================================================================
print("\n" + "=" * 60)
print("Part 11: Accuracy Comparison")
print("=" * 60)

# Overall accuracy comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Overall accuracy
ax1 = axes[0]
accuracies = [accuracy_original * 100, accuracy_no_keyword * 100]
bars = ax1.bar(
    ['Original\n(With Keywords)', 'No Keywords\n(Removed)'],
    accuracies, color=['steelblue', 'coral']
)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy Comparison: With vs Without Keywords', fontsize=14)
ax1.set_ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', fontsize=11)

# Right: Accuracy drop
ax2 = axes[1]
ax2.bar(['Accuracy Drop'], [accuracy_drop_percent], color=['indianred'])
ax2.set_ylabel('Drop (%)', fontsize=12)
ax2.set_title(f'Accuracy Drop: -{accuracy_drop_percent:.1f}%', fontsize=14)
ax2.set_ylim(0, max(accuracy_drop_percent * 1.5, 50))
ax2.text(0, accuracy_drop_percent + 2, f'-{accuracy_drop_percent:.1f}%',
         ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Accuracy comparison chart saved!")

# =============================================================================
# Part 12: Per-Class Precision Comparison
# =============================================================================
print("\n" + "=" * 60)
print("Part 12: Per-Class Precision Comparison")
print("=" * 60)

from sklearn.metrics import precision_recall_fscore_support

precision_orig, recall_orig, f1_orig, _ = precision_recall_fscore_support(
    y_test, y_pred_orig, labels=[0, 1, 2, 3, 4]
)
precision_no_kw, recall_no_kw, f1_no_kw, _ = precision_recall_fscore_support(
    y_test, y_pred_no_kw, labels=[0, 1, 2, 3, 4]
)

# Comparison visualization
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(5)
width = 0.35

bars1 = ax.bar(x - width/2, precision_orig * 100, width,
               label='Original (With Keywords)', color='steelblue')
bars2 = ax.bar(x + width/2, precision_no_kw * 100, width,
               label='No Keywords (Removed)', color='coral')

ax.set_xlabel('Cancer Type', fontsize=12)
ax.set_ylabel('Precision (%)', fontsize=12)
ax.set_title('Precision by Cancer Type: With vs Without Keywords', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([label_names[i] for i in range(5)], rotation=15)
ax.legend()
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('precision_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed comparison
print("\nPer-Class Precision Comparison:")
print(f"{'Cancer Type':<20} {'Original':<12} {'No Keywords':<12} {'Drop':<10}")
print("-" * 54)
for i in range(5):
    drop = (precision_orig[i] - precision_no_kw[i]) * 100
    print(f"{label_names[i]:<20} {precision_orig[i]*100:.1f}%      {precision_no_kw[i]*100:.1f}%      {drop:.1f}%")

# =============================================================================
# Part 13: Confusion Matrix Comparison
# =============================================================================
print("\n" + "=" * 60)
print("Part 13: Confusion Matrix Comparison")
print("=" * 60)

# Calculate confusion matrix for no-keyword model
cm_no_kw = confusion_matrix(y_test, y_pred_no_kw)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Original
ax1 = axes[0]
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
    xticklabels=[label_names[i] for i in range(5)],
    yticklabels=[label_names[i] for i in range(5)]
)
ax1.set_title(f'Confusion Matrix - Original\nAccuracy: {accuracy_original*100:.1f}%', fontsize=12)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')

# Right: No keywords
ax2 = axes[1]
sns.heatmap(
    cm_no_kw, annot=True, fmt='d', cmap='Oranges', ax=ax2,
    xticklabels=[label_names[i] for i in range(5)],
    yticklabels=[label_names[i] for i in range(5)]
)
ax2.set_title(f'Confusion Matrix - No Keywords\nAccuracy: {accuracy_no_keyword*100:.1f}%', fontsize=12)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Confusion matrix comparison saved!")

# =============================================================================
# Part 14: SHAP Comparison - Feature Importance
# =============================================================================
print("\n" + "=" * 60)
print("Part 14: SHAP Comparison (Optional)")
print("=" * 60)

try:
    # Extract TF-IDF vectorizer and model
    vectorizer_no_kw = pipeline_no_kw.named_steps["tfidf"]
    clf_no_kw = pipeline_no_kw.named_steps["clf"]

    # Transform test set
    X_test_tfidf_no_kw = vectorizer_no_kw.transform(X_test_no_kw)

    # Get feature names for no-keyword model
    feature_names_no_kw = vectorizer_no_kw.get_feature_names_out()

    # Create SHAP Explainer
    print("Calculating SHAP values for no-keyword model...")
    explainer_no_kw = shap.LinearExplainer(clf_no_kw, X_test_tfidf_no_kw)
    shap_values_no_kw = explainer_no_kw.shap_values(X_test_tfidf_no_kw)

    # Calculate feature importance for both
    mean_abs_shap_orig = np.abs(shap_values_orig).mean(axis=0)
    mean_abs_shap_orig = np.mean(mean_abs_shap_orig, axis=1)
    mean_abs_shap_no_kw = np.abs(shap_values_no_kw).mean(axis=0)
    mean_abs_shap_no_kw = np.mean(mean_abs_shap_no_kw, axis=1)

    # Get top 15 features (using original model's features)
    top_n = 15
    top_indices = np.argsort(mean_abs_shap_orig)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_values_orig = mean_abs_shap_orig[top_indices]

    # Get corresponding values from no-keyword model
    top_values_no_kw = []
    for f in top_features:
        if f in feature_names_no_kw:
            idx = np.where(feature_names_no_kw == f)[0][0]
            top_values_no_kw.append(mean_abs_shap_no_kw[idx])
        else:
            top_values_no_kw.append(0)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Original
    ax1 = axes[0]
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    ax1.barh(range(top_n), list(top_values_orig[::-1]), color=colors1)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_features[::-1])
    ax1.set_xlabel('Mean |SHAP Value|')
    ax1.set_title('SHAP Feature Importance - Original\n(With Keywords)', fontsize=12, pad=15)

    # Right: No Keywords
    ax2 = axes[1]
    colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, top_n))[::-1]
    ax2.barh(range(top_n), top_values_no_kw[::-1], color=colors2)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_features[::-1])
    ax2.set_xlabel('Mean |SHAP Value|')
    ax2.set_title('SHAP Feature Importance - No Keywords\n(Removed Disease Names)', fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('shap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nSHAP comparison completed!")

except Exception as e:
    print(f"SHAP comparison failed: {e}")
    print("Skipping this step...")

# =============================================================================
# Part 15: Bias and Variability Analysis
# =============================================================================
print("\n" + "=" * 60)
print("Part 15: Bias and Variability Analysis")
print("=" * 60)

# Perform 10-fold Cross Validation
print("Running 10-fold Cross Validation...")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_orig = cross_val_score(pipeline_orig, X_orig, y, cv=skf, scoring='accuracy')
cv_scores_no_kw = cross_val_score(pipeline_no_kw, X_no_kw, y, cv=skf, scoring='accuracy')

# Plot CV accuracy fluctuation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Original
ax1 = axes[0]
ax1.plot(range(1, 11), cv_scores_orig * 100, 'o-', color='steelblue', linewidth=2)
ax1.axhline(y=cv_scores_orig.mean() * 100, color='steelblue', linestyle='--',
            label=f'Mean: {cv_scores_orig.mean()*100:.1f}%')
ax1.fill_between(
    range(1, 11),
    (cv_scores_orig.mean() - cv_scores_orig.std()) * 100,
    (cv_scores_orig.mean() + cv_scores_orig.std()) * 100,
    alpha=0.2, color='steelblue'
)
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title(f'10-Fold CV - Original\nStd: {cv_scores_orig.std()*100:.2f}%', fontsize=12)
ax1.set_ylim(0, 100)
ax1.legend()

# Right: No keywords
ax2 = axes[1]
ax2.plot(range(1, 11), cv_scores_no_kw * 100, 'o-', color='coral', linewidth=2)
ax2.axhline(y=cv_scores_no_kw.mean() * 100, color='coral', linestyle='--',
            label=f'Mean: {cv_scores_no_kw.mean()*100:.1f}%')
ax2.fill_between(
    range(1, 11),
    (cv_scores_no_kw.mean() - cv_scores_no_kw.std()) * 100,
    (cv_scores_no_kw.mean() + cv_scores_no_kw.std()) * 100,
    alpha=0.2, color='coral'
)
ax2.set_xlabel('Fold', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title(f'10-Fold CV - No Keywords\nStd: {cv_scores_no_kw.std()*100:.2f}%', fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend()

plt.tight_layout()
plt.savefig('cv_variability.png', dpi=150, bbox_inches='tight')
plt.show()

# Print statistics
print("\n10-Fold CV Statistics:")
print(f"\nOriginal Data:")
print(f"   Mean Accuracy: {cv_scores_orig.mean()*100:.2f}%")
print(f"   Std Dev:       {cv_scores_orig.std()*100:.2f}%")
print(f"\nNo Keywords:")
print(f"   Mean Accuracy: {cv_scores_no_kw.mean()*100:.2f}%")
print(f"   Std Dev:       {cv_scores_no_kw.std()*100:.2f}%")

# =============================================================================
# Part 16: AUC and ROC Curves
# =============================================================================
print("\n" + "=" * 60)
print("Part 16: AUC and ROC Curves")
print("=" * 60)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Get decision function values
decision_orig = clf_orig.decision_function(X_test_tfidf_orig)
decision_no_kw = clf_no_kw.decision_function(X_test_tfidf_no_kw)

# Binarize labels
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

# Calculate ROC and AUC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Original
ax1 = axes[0]
colors = ['steelblue', 'coral', 'green', 'purple', 'gold']
for i, color in enumerate(colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], decision_orig[:, i])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=color, lw=2,
             label=f'{label_names[i]} (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=2)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve - Original (With Keywords)', fontsize=12)
ax1.legend(loc="lower right", fontsize=9)

# Right: No keywords
ax2 = axes[1]
for i, color in enumerate(colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], decision_no_kw[:, i])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color=color, lw=2,
             label=f'{label_names[i]} (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=2)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve - No Keywords (Removed)', fontsize=12)
ax2.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print AUC per class
print("\nAUC per Class:")
print(f"{'Cancer Type':<20} {'Original':<12} {'No Keywords':<12}")
print("-" * 44)
for i in range(5):
    fpr_orig, tpr_orig, _ = roc_curve(y_test_bin[:, i], decision_orig[:, i])
    auc_orig = auc(fpr_orig, tpr_orig)
    fpr_no_kw, tpr_no_kw, _ = roc_curve(y_test_bin[:, i], decision_no_kw[:, i])
    auc_no_kw = auc(fpr_no_kw, tpr_no_kw)
    print(f"{label_names[i]:<20} {auc_orig:.3f}         {auc_no_kw:.3f}")

# =============================================================================
# Part 17: Summary Report
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY REPORT")
print("=" * 60)

print("\n[Key Findings]")
print(f"""
1. Accuracy Comparison:
   - Original (with keywords):    {accuracy_original*100:.2f}%
   - After removing keywords:     {accuracy_no_keyword*100:.2f}%
   - Accuracy drop:               {accuracy_drop_percent:.1f}%

2. 10-Fold CV Variance:
   - Original Std:         {cv_scores_orig.std()*100:.2f}%
   - No keywords Std:      {cv_scores_no_kw.std()*100:.2f}%

3. Error Analysis:
   - Misclassified samples: {len(errors_X)} / {len(X_test)} ({len(errors_X)/len(X_test)*100:.1f}%)
""")

print("[Story Interpretation]")
print("If accuracy drops significantly after removing keywords:")
print("   - Model mainly relies on disease name keywords")
print("   - Model does not truly understand semantic content")
print("   - This is a typical 'data leakage' problem")
print("")
print("If some classes remain classifiable after removing keywords:")
print("   - These classes have unique terminology or research topics")
print("   - Model learned true semantic features")

print("\n[Generated Charts]")
print("   1. confusion_matrix_original.png")
print("   2. shap_summary_original.png")
print("   3. accuracy_comparison.png")
print("   4. precision_comparison.png")
print("   5. confusion_matrix_comparison.png")
print("   6. shap_comparison.png (optional)")
print("   7. cv_variability.png")
print("   8. roc_comparison.png")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
