# Project Title: High-Accuracy Prediction of Contact Lens Lot Output via Batch-Level Modeling

## 1. Introduction / Overview

This project focuses on accurately predicting the final output count (number of passed lenses) for a production lot in contact lens manufacturing.
It demonstrates the superiority of using granular batch-level data compared to aggregated lot-level features and utilizes a tuned Random Forest model to achieve high prediction accuracy (R² > 0.9).
The goal is to provide a tool that enables better inventory withdrawal planning based on predicted lot yields.
## 2. Problem Statement

Accurately forecasting the output yield of a manufacturing lot *before* processing is crucial for efficient inventory management and order fulfillment planning.
Predicting lot output based solely on aggregated summary statistics of its constituent batches can lead to information loss and suboptimal accuracy.
There is a need for a reliable predictive model to estimate lot output based on detailed batch history, particularly considering factors like variable storage times.
## 3. Goal / Objective

Primary Goal: Develop a machine learning regression model to accurately predict Total_Pass_Lens_Lot (count) for a 6-batch production lot.
Methodology Goal: Compare the predictive performance of modeling using aggregated lot-level features versus modeling using granular batch-level features (with aggregated predictions).
Performance Goal: Achieve a high R-squared (>0.85, achieved 0.925) and low MAE/RMSE on the test set for lot output prediction using the best identified approach.
Identify key batch-level features driving prediction accuracy.
## 4. Data

Utilized **simulated data** (Python, Pandas, NumPy) reflecting multi-stage batch production, based on realistic process flows and domain expertise.
Generated features for **individual batches** including: Power, Mould_Line, Storage_Duration_Days, upstream QC results (Mould_Dimension_QC_Result, Filling_QC_Result), Monomer_Lot_Number, and Mould_In_Count.
Simulated Lens_Pass_Count per batch was influenced by factors like storage duration (>180 days), low power range (-2.50D to 0.00D), and upstream QC results, targeting realistic batch yields (typically 70-95%).
Batches were programmatically grouped into 'Lots' of 6 batches based on sharedPower to prepare data for Lot ID-based splitting and final evaluation.
The **batch-level Lens_Pass_Count** served as the target variable for the primary modeling approach (Approach 2).
The **Lot-level Total_Pass_Lens_Lot** (sum of batch pass counts) served as the target for the alternative modeling approach (Approach 1) and the final evaluation metric.
## 5. Methodology / Workflow

**Data Simulation:** Generated batch-level data with realistic dependencies.
**Lot Grouping:** Assigned unique Lot IDs to groups of 6 same-power batches.
**Approach Comparison & Model Exploration:**
*Approach 1 (Lot-Level Features):* Calculated aggregated features per Lot. Tested models (MLP/DNN via TF/Keras, Random Forest) predicting total lot output directly. Found performance limited (best RF R²≈0.53), likely due to information loss.
*Approach 2 (Batch-Level Features):* Used granular batch-level features to predict individual batch pass counts (Lens_Pass_Count). This approach was explored further.
**Feature Engineering (Batch Level):** Prepared batch features (incl. storage duration, input count, QC status, power), applying One-Hot Encoding for categorical variables using Pandas.
**Model Comparison (Batch Level):** Trained and evaluated several models on the preprocessed batch-level data, including RandomForestRegressor (Scikit-learn), XGBoost, and an MLP Neural Network (TensorFlow/Keras).
**Model Selection:** Selected Random Forest as the most promising model based on initial batch-level evaluations (considering aggregated Lot-level metrics).
**Hyperparameter Tuning:** Optimized the selected batch-level Random Forest model using RandomizedSearchCV(Scikit-learn) to maximize R-squared.
**Prediction Aggregation:** Used the **tuned** Random Forest model to predict Lens_Pass_Countfor all batches in the test set. Summed these predictions by Lot_ID to get the final predicted Total_Pass_Lens_Lot.
**Final Evaluation:** Calculated final Lot-level MAE, RMSE, and R² on the test set by comparing the aggregated predictions against the actual summed pass counts.
**Residual Analysis:** Examined the residuals of the tuned batch-level model to validate model fit and assumptions.
[Analysis Notebook](Project_3_Lot_Output_Prediction.ipynb)


## 6. Key Findings & Results

**Batch-Level Modeling Superiority:** Predicting at the individual batch level and aggregating results (Approach 2) significantly outperformed modeling directly on aggregated lot-level features (Approach 1).
**Tuned Random Forest Performance:** The optimized Random Forest Regressor trained on batch-level data achieved high accuracy for predicting the final aggregated lot output:
**R-squared (Test Set): 0.925** (explaining ~93% of variance)
**MAE (Test Set): ≈ 102 lenses** (average error per lot)
**RMSE (Test Set): ≈ 126 lenses**
**Key Batch-Level Drivers:** Feature importance analysis of the best RF model highlighted:
**Mould_In_Count**: Most dominant factor.
**Storage_Duration_Days**: Second most critical predictor.
**Is_Low_Powe_Lot**: Flag for low power range (-2.5D to 0.0D) ranked third.
**Upstream QC Results** (Mould_Dimension_QC_Result_Pass, Filling_QC_Result_Pass): Also contributed.
**Residual Analysis:** Indicated the final model was generally well-behaved with no major systematic errors, supporting the validity of the results.
## 7. Conclusions & Business Impact

Modeling at the granular batch level preserves critical information lost during feature aggregation, leading to substantially more accurate Lot output predictions for this process.
The final tuned Random Forest model (R²≈0.93) provides a highly accurate method for forecasting the expected number of usable lenses from a production lot *before* committing resources to delensing.
This accurate prediction capability directly enables **more precise inventory withdrawal planning** from the warehouse to meet specific customer order quantities, potentially reducing delays caused by under-prediction or wasted effort from over-prediction.
Insights from batch-level feature importance reinforce the need to manage storage duration and maintain high upstream quality control.
## 8. Technologies Used

Python, Pandas, NumPy, Scikit-learn (RandomForestRegressor, RandomizedSearchCV, train_test_split, metrics, OneHotEncoder implicitly via get_dummies), Matplotlib, Seaborn, Statsmodels (for EDA), TensorFlow/Keras & XGBoost (explored), Google Colab, Git/GitHub.
## 9. Limitations

Analysis based on **simulated data**; requires validation on real-world data.
Simulation captured key factors but omits full real-world complexity and noise.
Hyperparameter tuning (RandomizedSearchCV) explored a limited parameter space.
Assumed batch independence within a lot after accounting for features.
## 10. Future Work

Validate the batch-level modeling approach using historical production data.
Deploy the trained RF model (or a retrained version on real data) as part of an inventory planning tool.
Explore more advanced feature engineering at the batch level.
Conduct more extensive hyperparameter tuning or compare with tuned Gradient Boosting models.
