# Cancer Classification Project

This project compares multiple NLP models to classify medical abstracts into 5 cancer categories (Colon, Liver, Lung, Stomach, Thyroid) and evaluates how model performance changes when key diagnostic terms are removed.

---

# Setup Environment

## Option 1: Google Colab (Recommended)

1. Open `model_comparison_merged.ipynb` in Google Colab
2. Upload the required CSV files when prompted:
   - `cancer_cleaned_trimmed.csv`
   - `cancer_cleaned_v2_trimmed.csv`
3. Run cells sequentially

## Option 2: Local Execution (VS Code / Jupyter)

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers xgboost shap
```

---

# Directory Structure

```
Classification-Assignment-/
├── model_comparison_merged.ipynb  # Main analysis notebook
├── cancer_cleaned_trimmed.csv      # Baseline dataset (with disease names)
├── cancer_cleaned_v2_trimmed.csv   # Dataset without disease keywords
├── README.md
└── data_preparation/
    ├── Colon_Cancer.csv
    ├── Liver_Cancer.csv
    ├── Lung_Cancer.csv
    ├── Stomach_Cancer.csv
    └── Thyroid_Cancer.csv
```

---

# Execution Steps

## Step 1: Prepare Data (Optional)

**Option A - Use Pre-processed Files (Recommended):**
- Upload `cancer_cleaned_trimmed.csv` and `cancer_cleaned_v2_trimmed.csv` to Colab or the project root

**Option B - Re-process Raw Data:**
- Run cells in `data_preparation/` folder to generate cleaned CSV files from raw data

## Step 2: Run the Notebook

**In Google Colab:**
1. Open `model_comparison_merged.ipynb` in Colab
2. Click "Runtime" → "Run all" OR run cells sequentially

**In VS Code:**
1. Open `model_comparison_merged.ipynb`
2. Install Jupyter extension
3. Click "Run All" or execute cells individually

# Key Outputs

## Model Performance
Champion Model: providing the best balance of speed and accuracy.

Classification Report: Detailed F1-scores for 5 cancer types (Colon, Liver, Lung, Stomach, Thyroid).

Ablation Study: Comparison showing the accuracy drop when disease-specific keywords are removed.

## Generated Visualizations

The following files are generated in the working directory:

| File | Description |
|------|-------------|
| `confusion_matrix_comparison.png` | Side-by-side confusion matrices (Original vs No Keywords) |
| `shap_feature_importance_comparison.png` | 3-panel feature importance comparison |
| `accuracy_comparison.png` | Accuracy drop annotation when keywords removed |
| `roc_curves.png` | ROC curves for all 5 cancer types |
| `cv_variability.png` | 10-fold CV stability comparison |
