import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Importing our professional pipeline blocks
from data_pipeline import DataIngestionPipeline, ClinicalFeatureEngineer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def execute_production_training():
    print("🚀 Starting HemaPredict AI High-Accuracy Training Pipeline...")
    
    # 1. Load the actual clinical dataset
    try:
        df = pd.read_csv("dataset.csv")
        print("✅ Successfully loaded 'dataset.csv'")
    except FileNotFoundError:
        print("❌ Error: 'dataset.csv' not found. Please place your real dataset in the project directory.")
        return

    # Extract features (X) and labels (y)
    if 'Target' not in df.columns:
        print("❌ Error: Dataset must contain a 'Target' column for classifications (0, 1, 2).")
        return
        
    X_raw = df.drop(columns=['Target'])
    y = df['Target']

    # 2. Initialize and Fit Data Ingestion (Imputation & Scaling)
    ingestion_pipeline = DataIngestionPipeline()
    X_processed = ingestion_pipeline.process_data(X_raw, is_training=True)
    
    # 3. Execute Mathematical Feature Engineering (NLR, PLR, SII, HRR)
    feature_engineer = ClinicalFeatureEngineer()
    X_engineered = feature_engineer.generate_features(X_processed)
    
    # 4. Stratified Train-Test Split (80% Train, 20% Independent Validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_split=0.2, stratify=y, random_state=42
    )
    
    print(f"📊 Training Matrix Shape: {X_train.shape} | Validation Matrix Shape: {X_test.shape}")

    # 5. Train the High-Accuracy Core Models
    print("🧠 Training Optimized XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    print("🧠 Training Optimized LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)

    print("🧠 Training Optimized CatBoost Classifier...")
    cat_model = CatBoostClassifier(verbose=0, random_state=42)
    cat_model.fit(X_train, y_train)

    # 6. Evaluate Ensemble Accuracy on Independent Testing Cohort
    xgb_preds = xgb_model.predict_proba(X_test)
    lgb_preds = lgb_model.predict_proba(X_test)
    cat_preds = cat_model.predict_proba(X_test)
    
    consensus_preds_proba = (xgb_preds + lgb_preds + cat_preds) / 3.0
    final_predictions = np.argmax(consensus_preds_proba, axis=1)
    
    # Calculate and display true production metrics
    production_accuracy = accuracy_score(y_test, final_predictions)
    print(f"\n🎯 True Production Validation Accuracy: {production_accuracy * 100:.2f}%")
    print("\n📋 Detailed Empirical Classification Report:\n")
    print(classification_report(y_test, final_predictions, target_names=['Healthy', 'Hematological', 'Solid Tumor']))

    # 7. Serialize and Save the entire high-accuracy artifact box
    artifacts = {
        'ingestion_pipeline': ingestion_pipeline,
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'cat_model': cat_model,
        'feature_names': X_engineered.columns.tolist()
    }
    
    joblib.dump(artifacts, "hemapredict_model.joblib")
    print("💾 High-accuracy model artifacts successfully exported to 'hemapredict_model.joblib'")

if __name__ == "__main__":
    execute_production_training()