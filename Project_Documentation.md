# Project Documentation: Car Evaluation System

## 1. Abstract
Purchasing a vehicle requires evaluating multiple trade-offs across acquisition cost, maintenance expenses, passenger capacity, cargo utility, and safety ratings. The **Car Evaluation System** leverages Data Science and Machine Learning (Random Forest Classification) to automate vehicle assessment and provide data-driven buying recommendations alongside a depreciation-based resale valuation calculator.

## 2. System Methodology
1. **Data Acquisition & Generation:** 1,728 structural combinations across 6 core attributes (`buying_price`, `maintenance_cost`, `doors`, `persons`, `luggage_boot`, `safety`).
2. **Data Cleaning Pipeline:** Automated detection and removal of duplicate entries and mode-imputation for missing categorical records.
3. **Exploratory Data Analysis (EDA):** Statistical visualization of target class distribution and inter-feature correlations.
4. **Predictive Modeling:** Supervised multi-class classification using a 150-tree Random Forest Classifier with an 80/20 stratified train-test split.

## 3. Key Findings & Visual Insights
- **Safety & Seating Dominance:** Feature importance analysis reveals that `safety` and `persons` (seating capacity) are the two strongest determinants of whether a vehicle is classified as acceptable or unacceptable.
- **Cost Sensitivity:** High buying prices paired with high maintenance costs consistently result in an `Unacceptable` rating regardless of door count or boot size.
- **Model Performance:** The Random Forest model achieves >95% test accuracy across all four evaluation classes.
-