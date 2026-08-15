# OBDII-DS3 Predictive Maintenance — Notebook Build Prompts

الملف ده متقسم لبرومبتات صغيرة. ابعتهم للـAI **بالترتيب**، وبرومبت كل Step يكون بعد ما الـAI يخلص الـStep اللي قبله ويراجع الـoutput قبل الانتقال.

> **Important:** ممنوع اختراع columns أو labels. أي قرار متعلق بالـtarget لازم يكون مبني على الـdataset الفعلية وعلى البحث.

---

## STEP 0 — Research + Dataset Verification

```text
We are going to reproduce the Machine Learning pipeline from this paper:

"Enhancing predictive maintenance in automotive industry: addressing class imbalance using advanced machine learning techniques"
DOI: 10.1007/s42452-025-06827-3

Dataset:
OBDII-DS3
Kaggle:
https://www.kaggle.com/datasets/cephasax/obdii-ds3

Before writing any ML training code, verify the dataset and the paper.

Tasks:
1. Inspect the actual OBDII-DS3 dataset available in the environment.
2. Read the paper carefully, especially:
   - Dataset description
   - Original features
   - DTC_NUMBER / TROUBLE_CODES
   - Target definition
   - Class imbalance
   - Preprocessing
   - Feature selection
   - Models
   - SMOTE
   - Ensemble methods
   - Cost-sensitive learning
   - XGBoost
3. Verify whether the dataset in the environment is actually OBDII-DS3.
4. Read ALL CSV files in the dataset directory.
5. Print for every CSV:
   - filename
   - shape
   - columns
   - dtypes
   - first 5 rows
   - missing values
   - duplicate rows
6. Compare the actual columns with the paper.
7. Do NOT invent any feature, label, or target.
8. Determine exactly how DTC_NUMBER / TROUBLE_CODES are used to create the binary target.
9. If the paper or dataset leaves ambiguity about the target definition, STOP and explain the ambiguity before training.
10. If there is no ambiguity, explicitly document:
    Normal = 0
    Failure = 1

At the end, give me:
- Verified dataset name
- Files found
- Original feature count
- Actual column names
- Candidate target columns
- Exact target construction rule
- Class distribution
- Any discrepancy between paper and dataset

Do not train models yet.
```

---

## STEP 1 — Notebook Skeleton + Data Loading

```text
Now create the first executable part of the Jupyter Notebook.

Goal:
Build a clean, reproducible notebook for OBDII-DS3 predictive maintenance.

Use these libraries where available:
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
lime
scipy
joblib

Tasks:
1. Import libraries.
2. Set reproducible random seeds.
3. Define clean paths/configuration.
4. Read ALL CSV files belonging to the verified OBDII-DS3 dataset.
5. Inspect:
   - shape
   - columns
   - dtypes
   - first rows
   - missing values
   - duplicate rows
6. Construct the target ONLY using the verified rule from STEP 0.
7. Print:
   - final dataset shape
   - target distribution
   - target percentages
8. Keep the raw dataset untouched as raw_df.
9. Create a separate working dataframe.

Important:
- Do not invent columns.
- Do not train anything.
- Do not perform imputation, scaling, SMOTE, or feature selection yet.
- Add Markdown explaining every section.
- Add comments inside code.
- Add an output interpretation section after the code.
```

---

## STEP 2 — Target Validation

```text
Before any preprocessing or model training, validate the target construction.

Tasks:
1. Show examples of the original DTC_NUMBER / TROUBLE_CODES values.
2. Show how each relevant value maps to:
   Normal = 0
   Failure = 1
3. Verify that no unexpected target values exist.
4. Print:
   - total normal samples
   - total failure samples
   - percentages
5. Plot target distribution.
6. Check whether the target is heavily imbalanced.
7. Explicitly explain why Accuracy alone is insufficient.

If the target construction cannot be verified from the actual dataset and paper:
STOP.
Do not continue to training.

Otherwise save the verified target dataframe for the next step.
```

---

## STEP 3 — Exploratory Data Analysis

```text
Perform EDA on the verified OBDII-DS3 dataset.

Tasks:
1. Target distribution.
2. Missing-value analysis.
3. Numerical feature distributions.
4. Summary statistics.
5. Correlation matrix.
6. Boxplots for numerical features to identify possible outliers.
7. Class imbalance visualization.
8. Compare important numerical features between Normal and Failure classes where meaningful.

Do not modify the dataset during EDA.

Important:
- Use only columns that actually exist.
- Do not assume ENGINE_RPM, ENGINE_LOAD, ENGINE_COOLANT_TEMP, or THROTTLE_POS exist without checking.
- If one of these columns does not exist, explicitly say so and skip it.
- Do not create fake values.

For every major plot, provide a short interpretation.
```

---

## STEP 4 — Data Cleaning + Leakage-Safe Preprocessing

```text
Implement preprocessing for the verified dataset.

Requirements:
1. Remove duplicate rows where appropriate.
2. Handle missing values.
3. For ENGINE_RPM:
   - use interpolation only if the actual dataset structure makes interpolation logically valid.
4. For ENGINE_LOAD and ENGINE_COOLANT_TEMP:
   - use mean imputation only if appropriate and only if these columns actually exist.
5. Use KNN imputation for suitable numerical features when justified.
6. Use forward fill/backward fill only when justified by the actual row/time structure.
7. Detect outliers using IQR.
8. Handle outliers carefully and explain the chosen strategy.

CRITICAL DATA LEAKAGE RULE:
All learned preprocessing must be fitted ONLY on training data.

Therefore:
- First establish train/test split where appropriate.
- Build preprocessing using sklearn Pipeline / ColumnTransformer where possible.
- Never calculate imputation statistics using the full dataset before splitting.
- Never fit scaler on test data.
- Never apply SMOTE to test data.

Do not use the test set for any preprocessing decision.

Explain every preprocessing decision and why it is valid for this dataset.
```

---

## STEP 5 — Train/Test Split

```text
Create the final train/test split.

Requirements:
- Use train_test_split.
- Use stratify=y.
- Use a fixed random_state.
- Clearly print:
  X shape
  y shape
  train size
  test size
  train class distribution
  test class distribution
- State the exact train/test percentage.

Important:
The test set must remain untouched until final evaluation.

Do not apply:
- SMOTE
- imputation fitting
- scaling fitting
- feature selection using the full dataset

in a way that leaks information from test data.
```

---

## STEP 6 — Feature Selection

```text
Start from the actual original features available in OBDII-DS3.

Perform feature selection using evidence, not guesswork.

Consider:
1. Domain relevance.
2. Missingness.
3. Correlation / redundancy.
4. Model-based feature importance.

Show:
- Original features
- Features removed
- Selected features
- Reason for each removal

Important:
- Do not remove a feature simply because its name looks unimportant.
- Do not invent features.
- Do not use test labels to select features.
- Feature selection must be fitted only using training data.
- If model-based selection is used, perform it inside the training workflow where possible.

At the end print the final selected feature list.
```

---

## STEP 7 — Baseline Models

```text
Train baseline classification models on the training data.

Models:
1. Logistic Regression
2. SVM
3. Decision Tree
4. Random Forest
5. XGBoost
6. LightGBM

Requirements:
- Use reasonable initial hyperparameters.
- Handle preprocessing correctly per model.
- Do not use test data during training.
- Do not apply SMOTE yet.
- Keep the same train/test split for all models.

Evaluate on the untouched test set.

For every model calculate:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

Important:
Because of class imbalance, prioritize:
F1
Recall
Precision
ROC-AUC

Create a comparison dataframe:
Model | Accuracy | Precision | Recall | F1 | ROC-AUC
```

---

## STEP 8 — Baseline Hyperparameter Tuning

```text
Now tune the baseline models.

Use GridSearchCV or RandomizedSearchCV.

Models to tune:
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

Requirements:
1. Use cross-validation.
2. Use an imbalance-aware scoring strategy such as F1 or ROC-AUC where appropriate.
3. Do not use the test set during tuning.
4. Put preprocessing inside the CV pipeline whenever necessary.
5. Print best parameters.
6. Evaluate each tuned model ONCE on the untouched test set.
7. Update the comparison table.

Do not choose the final best model based on Accuracy alone.
```

---

## STEP 9 — SMOTE Experiment

```text
Run a separate SMOTE experiment.

CRITICAL:
SMOTE must be applied ONLY to the training data.

Never:
- SMOTE before train/test split
- SMOTE the test set
- fit preprocessing using test data

Tasks:
1. Print training class distribution before SMOTE.
2. Apply SMOTE to training data only.
3. Print distribution after SMOTE.
4. Train the same baseline model families.
5. Evaluate on the original untouched test set.

Metrics:
Accuracy
Precision
Recall
F1
ROC-AUC

Create:
Model | Accuracy | Precision | Recall | F1 | ROC-AUC

Compare SMOTE results against baseline results.
Explain which models benefited from SMOTE and why.
```

---

## STEP 10 — Ensemble Methods

```text
Implement ensemble methods suitable for imbalanced binary classification.

At minimum:
1. Balanced Random Forest
2. Easy Ensemble
3. RUSBoost
4. Balanced Bagging

Use the appropriate classes from imbalanced-learn.

Requirements:
- Training data only.
- Test set untouched.
- Use reasonable hyperparameters first.
- Tune important parameters if computationally practical.
- Evaluate using:
  Accuracy
  Precision
  Recall
  F1
  ROC-AUC
  Confusion Matrix

Focus especially on:
Recall
Precision
F1
ROC-AUC

Add all results to the master comparison dataframe.
```

---

## STEP 11 — Cost-Sensitive Learning

```text
Implement cost-sensitive versions of suitable models.

At minimum investigate:
- XGBoost
- LightGBM
- AdaBoost
- Random Forest if supported appropriately

Use:
- class_weight where supported
- scale_pos_weight for XGBoost where appropriate
- sample weights where appropriate

Do NOT manually guess class weights.

Calculate class weights from the training data only.

For XGBoost:
scale_pos_weight should be based on the training class distribution, e.g.:
number_of_negative_training_samples / number_of_positive_training_samples

Explain exactly how each weighting strategy was calculated.

Evaluate:
Accuracy
Precision
Recall
F1
ROC-AUC

Add results to the master comparison table.
```

---

## STEP 12 — Final XGBoost Model

```text
Build a strong final XGBoost classifier.

Requirements:
1. Hyperparameter tuning.
2. Cross-validation.
3. Class imbalance handling.
4. Appropriate evaluation metric.
5. No test-set leakage.
6. Use training data only during CV and tuning.

Tune reasonable parameters such as:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight
- gamma
- reg_alpha
- reg_lambda

Do not blindly use all parameters if the dataset is small.

Print:
- best parameters
- best CV score
- final test Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- confusion matrix
- classification report
```

---

## STEP 13 — Final Model Comparison

```text
Create one final master comparison table containing every trained approach.

Columns:
Model
Approach
Accuracy
Precision
Recall
F1
ROC-AUC

Approaches should distinguish:
- Baseline
- Tuned Baseline
- SMOTE
- Ensemble
- Cost-Sensitive

Then identify programmatically:
1. Best model according to F1
2. Best model according to Recall
3. Best model according to ROC-AUC

Do not hard-code the answers.

Explain the trade-off between:
- False Negatives
- False Positives
- Precision
- Recall
- F1
- ROC-AUC

For predictive maintenance, explicitly discuss why False Negatives are especially important.
```

---

## STEP 14 — Confusion Matrices

```text
Create confusion matrices for the best models.

For every selected best model show:
- TN
- FP
- FN
- TP

Also calculate:
- False Negative Rate
- False Positive Rate

Explain:
- What a False Negative means in this automotive predictive-maintenance problem.
- Why missing a Failure can be more dangerous than generating an unnecessary maintenance alert.

Do not make unsupported claims about specific components.
```

---

## STEP 15 — ROC Curves

```text
Create ROC curves for the model groups.

Show:
- Baseline models
- SMOTE models
- Ensemble models
- Cost-sensitive models

Display AUC in the legend.

Use the same untouched test set for comparison.

Explain which model has the strongest discrimination according to ROC-AUC.
```

---

## STEP 16 — Feature Importance

```text
For the best tree-based model, calculate feature importance.

Tasks:
1. Extract actual feature names after preprocessing.
2. Calculate model-based feature importance.
3. Sort descending.
4. Plot a clean horizontal bar chart.
5. Print the top features.

Important:
- Use only actual dataset features.
- Do not invent feature names.
- If preprocessing changes feature representation, map the importance back to the correct feature names.
```

---

## STEP 17 — SHAP Explainability

```text
Use SHAP to explain the best tree-based model.

Tasks:
1. Create an appropriate SHAP explainer for the actual selected model.
2. Generate:
   - SHAP summary plot
   - SHAP bar plot
3. Select one actual test sample predicted as Failure.
4. Select one actual test sample predicted as Normal.
5. Generate local explanations for both.
6. Explain which features pushed the prediction toward Failure or Normal.

Important:
- Use actual model inputs after preprocessing.
- Correctly map transformed features back to meaningful feature names.
- Do not fabricate explanations.
- If a SHAP plot requires a compatible explainer/model format, adapt the implementation correctly.
```

---

## STEP 18 — LIME

```text
Use LIME for local explanation.

Tasks:
1. Create a LIME Tabular Explainer compatible with the final model.
2. Select one actual test sample predicted as Failure.
3. Explain why the model predicted Failure.
4. Display the most influential feature conditions.
5. Compare the LIME explanation with SHAP where possible.

Do not create synthetic claims about vehicle components.
Only explain what the trained model learned from the available features.
```

---

## STEP 19 — Partial Dependence Plots

```text
Create Partial Dependence Plots for important features.

Priority:
- ENGINE_RPM
- ENGINE_COOLANT_TEMP
- ENGINE_LOAD
- THROTTLE_POS

BUT:
Only plot a feature if it actually exists in the verified dataset and is part of the final model input.

Also include additional top features based on actual feature importance.

Requirements:
- Use sklearn PartialDependenceDisplay where appropriate.
- Explain the relationship between each feature and predicted Failure probability.
- Do not interpret the relationship as causal.
```

---

## STEP 20 — Save Complete Model Pipeline

```text
Prepare the model for real deployment.

Save ONE complete deployment artifact using joblib:
best_model.pkl

The saved object must contain everything required for inference, including:
- preprocessing
- selected feature list
- imputer(s)
- scaler if used
- feature selection if used
- final trained model

Prefer saving a single sklearn Pipeline when possible so inference is reproducible.

Also save a metadata object if useful containing:
- feature names
- target definition
- model name
- random_state
- training date
- best hyperparameters
- metrics

Verify that the saved model can be loaded successfully in a fresh Python object.
```

---

## STEP 21 — Real-Time Inference Function

```text
Create a deployment-ready function:

predict_vehicle_status(input_data)

Requirements:
Input:
OBD-II sensor values corresponding ONLY to the features used by the final model.

Output:
{
    "prediction": "Normal" or "Failure",
    "failure_probability": XX,
    "risk_level": "Low" / "Medium" / "High"
}

Risk levels must be defined explicitly and programmatically.

Do not invent scientifically validated thresholds.

If Low/Medium/High thresholds are not supported by the paper, clearly label them as application-level thresholds and make them configurable.

The function must:
1. Validate required input features.
2. Reject missing required fields clearly.
3. Apply the saved preprocessing/model pipeline.
4. Return predicted class.
5. Return Failure probability.
6. Return risk level.

Make the function suitable for later integration into a Flask/FastAPI backend.
```

---

## STEP 22 — New OBD-II Data Test

```text
Test the deployment function using a real row from the held-out test set.

Requirements:
1. Take one actual test sample.
2. Convert it into the exact input format expected by predict_vehicle_status().
3. Run inference.
4. Print:
   - actual label
   - predicted label
   - failure probability
   - risk level

This is only a software inference demonstration.

Do not claim that one sample proves real-world performance.
```

---

## STEP 23 — Paper vs Reproduction Comparison

```text
Now compare our actual results with the results reported in:

"Enhancing predictive maintenance in automotive industry: addressing class imbalance using advanced machine learning techniques"
DOI: 10.1007/s42452-025-06827-3

Create a table:

Metric / Model | Paper Result | Our Result | Difference | Explanation

Use only numbers actually reported in the paper.

Do NOT fabricate paper values.

Explain possible reasons for differences:
- preprocessing differences
- train/test split
- random seed
- feature selection
- class distribution
- implementation differences
- library versions
- hyperparameters
- dataset version
- leakage prevention
- computational constraints

Explicitly state:
Our results are a reproduction/implementation attempt and are not expected to exactly match the paper.
```

---

## STEP 24 — Automotive Super App Integration

```text
Create the final Markdown section:

"How this model fits our Automotive Super App"

Explain this pipeline:

OBD-II Bluetooth Scanner
        ↓
Live Vehicle Sensor Data
        ↓
Data Validation
        ↓
Preprocessing
        ↓
ML Predictive Maintenance Model
        ↓
Failure Probability
        ↓
Risk Level
        ↓
AI Car Assistant
        ↓
Maintenance Recommendation

Explain:
1. How OBD-II data enters the system.
2. How the backend sends sensor values to the model.
3. How the model returns Failure probability.
4. How the application can trigger a maintenance recommendation.
5. How the AI assistant can explain the result to the user.
6. How the provider/garage system can receive a maintenance request.

CRITICAL:
Do NOT claim the model specifically predicts:
- Battery failure
- Fuel pump failure
- Brake pad failure
- Any other specific component failure

unless the actual dataset contains labels for those specific failures.

The model's claims must remain consistent with the actual OBDII-DS3 target and available features.
```

---

## STEP 25 — Final Notebook Audit

```text
Perform a final audit of the entire notebook.

Check for:

DATASET
- Is this actually OBDII-DS3?
- Are all CSV files handled?
- Are actual columns used?
- No invented labels/features?

TARGET
- Is DTC_NUMBER / TROUBLE_CODES handled according to the verified rule?
- Normal = 0
- Failure = 1
- Any ambiguity documented?

DATA LEAKAGE
- Was train/test split done before learned preprocessing?
- Was test data excluded from SMOTE?
- Was test data excluded from imputation fitting?
- Was test data excluded from feature selection?
- Was test data excluded from hyperparameter tuning?

MODELS
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- SMOTE experiments
- Balanced Random Forest
- Easy Ensemble
- RUSBoost
- Balanced Bagging
- Cost-sensitive learning

EVALUATION
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix
- Classification Report
- ROC curves

EXPLAINABILITY
- Feature importance
- SHAP summary
- SHAP bar
- SHAP local Failure
- SHAP local Normal
- LIME Failure
- PDP

DEPLOYMENT
- best_model.pkl
- preprocessing included
- feature list included
- inference function works
- test sample inference works

REPRODUCIBILITY
Print a final summary containing:
- dataset shape
- target distribution
- original feature count
- selected feature list
- train/test sizes
- random_state
- best hyperparameters
- final metrics
- best F1 model
- best Recall model
- best ROC-AUC model

Then identify any remaining weaknesses or limitations.
Do not hide failures or missing components.
```

---

## STEP 26 — Final Clean Notebook

```text
Now refactor the entire notebook into a clean final version.

Structure it exactly as:

1. Project Overview
2. Research Reference
3. Dataset Verification
4. Imports and Configuration
5. Load Dataset
6. Target Construction
7. Exploratory Data Analysis
8. Train/Test Split
9. Leakage-Safe Preprocessing
10. Feature Selection
11. Baseline Models
12. Hyperparameter Tuning
13. SMOTE Experiment
14. Ensemble Methods
15. Cost-Sensitive Learning
16. Final XGBoost
17. Model Comparison
18. Confusion Matrices
19. ROC Curves
20. Feature Importance
21. SHAP Explainability
22. LIME Explainability
23. Partial Dependence Plots
24. Model Saving
25. Real-Time Inference
26. Paper vs Reproduction
27. Automotive Super App Integration
28. Limitations
29. Final Summary

For every section:
- Markdown explanation
- executable code
- output interpretation

Quality requirements:
- No hard-coded metrics.
- No fabricated results.
- No invented columns.
- No data leakage.
- Reproducible random seeds.
- Clear variable names.
- Functions for repeated operations.
- Tables for comparisons.
- Professional plots.
- Robust error handling.
- Comments in important code.
- The notebook must run from top to bottom after the dataset path is configured.

At the very end print a concise final report generated FROM THE ACTUAL COMPUTED RESULTS.
```

---

# Execution Rule

نفّذ الـsteps **واحد واحد** بالترتيب.

بعد كل Step:
1. شغّل الكود.
2. راجع الـoutput.
3. لو فيه discrepancy أو error، أصلحه قبل الانتقال.
4. لا تفترض أي حاجة من الورقة لو الـdataset الفعلية بتقول حاجة مختلفة.
5. لا تنتقل للـStep التالي لو target definition لسه غير مؤكدة.
