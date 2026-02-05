# Cancer Classification Project

This project compares multiple NLP models to classify medical texts and evaluates how performance changes when key diagnostic terms are removed.

# Setup Environment

Install the required libraries:

Bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers xgboost

# Directory Structure

model_comparison_and_error_analysis.ipynb — Main analysis code.

cancer_cleaned_trimmed.csv — Baseline dataset.

cancer_cleaned_v2_trimmed.csv — Dataset without disease keywords.

data_preparation/

Colon_Cancer.csv
Liver_Cancer.csv
Lung_Cancer.csv
Stomach_Cancer.csv
Thyroid_Cancer.csv — The original, unprocessed data.

csv_cleaning.ipynb — The script used for cleaning and trimming the raw data.

# Execution Steps
   
re-process the raw data: run the script in the data_preparation folder.

Open the Notebook: Launch model_comparison_and_error_analysis.ipynb in Jupyter or VS Code.

Run Comparison: Execute the cells to train and compare models using TF-IDF.

BERT Features: Execute the BERT section to generate deep-learning embeddings.

Error Analysis: Run the final section to see how removing keywords affects accuracy.

# Key Outputs
   
Champion Model: providing the best balance of speed and accuracy.

Classification Report: Detailed F1-scores for 5 cancer types (Colon, Liver, Lung, Stomach, Thyroid).

Ablation Study: Comparison showing the accuracy drop when disease-specific keywords are removed.
