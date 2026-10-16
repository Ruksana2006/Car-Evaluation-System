"""
File: src/main_pipeline.py
Description: Complete Data Science & ML pipeline for the Car Evaluation System.
             Performs Data Cleaning, EDA, Visualization, Model Training, and Evaluation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Create output directories if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def step_1_clean_data(filepath="data/raw_car_data.csv"):
    """Loads raw dataset, removes duplicates, handles missing values, and saves clean CSV."""
    print("\n--- STEP 1: DATA CLEANING ---")
    df = pd.read_csv(filepath)
    print(f"Initial Raw Dataset Shape: {df.shape}")
    print(f"Missing Values Before Cleaning:\n{df.isnull().sum()}\n")

    # 1. Remove duplicate records
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_len - len(df)} duplicate rows.")

    # 2. Impute missing categorical values using Mode (most frequent value)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Filled missing values in '{col}' with mode: '{mode_val}'")

    # Save cleaned dataset
    clean_path = "data/cleaned_car_data.csv"
    df.to_csv(clean_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to '{clean_path}' with shape {df.shape}.")
    return df

def step_2_encode_features(df):
    """Converts ordinal categorical text into numerical mappings for correlation & ML."""
    mapping_dict = {
        'buying_price': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
        'maintenance_cost': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
        'doors': {'2': 2, '3': 3, '4': 4, '5more': 5},
        'persons': {'2': 2, '4': 4, 'more': 6},
        'luggage_boot': {'small': 1, 'med': 2, 'big': 3},
        'safety': {'low': 1, 'med': 2, 'high': 3},
        'evaluation_class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    }
    df_encoded = df.copy()
    for col, mapping in mapping_dict.items():
        df_encoded[col] = df_encoded[col].map(mapping)
    return df_encoded

def step_3_generate_visualizations(df, df_encoded):
    """Generates and saves 4 high-resolution analytical charts to the 'outputs/' folder."""
    print("\n--- STEP 2: GENERATING VISUALIZATIONS ---")
    sns.set_theme(style="whitegrid")

    # Chart 1: Target Class Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df, x='evaluation_class',
        order=['unacc', 'acc', 'good', 'vgood'],
        palette='viridis'
    )
    plt.title("Car Evaluation Class Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Evaluation Category")
    plt.ylabel("Number of Cars")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.3, p.get_height() + 15))
    plt.tight_layout()
    plt.savefig("outputs/01_class_distribution.png", dpi=300)
    plt.close()

    # Chart 2: Feature Correlation Heatmap
    plt.figure(figsize=(9, 7))
    corr = df_encoded.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/02_correlation_heatmap.png", dpi=300)
    plt.close()

    print("[SUCCESS] Saved distribution and correlation charts in 'outputs/'.")

def step_4_train_and_evaluate_model(df_encoded):
    """Trains a Random Forest Classifier and outputs feature importance & confusion matrix."""
    print("\n--- STEP 3: MACHINE LEARNING PREDICTION MODEL ---")
    X = df_encoded.drop('evaluation_class', axis=1)
    y = df_encoded['evaluation_class']

    # Split data: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Initialize and train Random Forest model
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    target_names = ['Unacceptable', 'Acceptable', 'Good', 'Very Good']
    report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"Model Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:\n", report)

    # Save evaluation report to text file
    with open("outputs/model_evaluation_report.txt", "w") as f:
        f.write(f"CAR EVALUATION ML MODEL REPORT\n")
        f.write(f"==============================\n")
        f.write(f"Model Used: Random Forest Classifier (150 Trees)\n")
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)

    # Chart 3: Feature Importance Bar Chart
    plt.figure(figsize=(8, 5))
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.plot(kind='barh', color='teal')
    plt.title("Feature Importance in Car Evaluation Prediction", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/03_feature_importance.png", dpi=300)
    plt.close()

    # Chart 4: Confusion Matrix
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Model Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig("outputs/04_confusion_matrix.png", dpi=300)
    plt.close()

    # Save trained model
    joblib.dump(model, "outputs/car_evaluator_model.pkl")
    print("[SUCCESS] Saved trained model and evaluation charts to 'outputs/'.")

if __name__ == "__main__":
    cleaned_df = step_1_clean_data()
    encoded_df = step_2_encode_features(cleaned_df)
    step_3_generate_visualizations(cleaned_df, encoded_df)
    step_4_train_and_evaluate_model(encoded_df)