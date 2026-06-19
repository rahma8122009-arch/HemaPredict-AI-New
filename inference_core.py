import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap

class EnsembleInferenceCore:
    """
    Orchestrates tree-based ensemble models (XGBoost, LightGBM, CatBoost) to process clinical features
    and output weighted multi-class probabilities alongside game-theoretic SHAP explanations.
    """
    def __init__(self):
        # Initializing the ensemble with standard hyper-parameters
        self.xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        self.lgb_model = lgb.LGBMClassifier(random_state=42)
        self.cat_model = CatBoostClassifier(verbose=0, random_state=42)
        self.explainer = None
        self.is_trained = False

    def train_mock_ensemble(self):
        """
        Generates synthetic clinical data to pre-train the ensemble for live deployment demonstration.
        In production, this is replaced by loading your curated global dataset.
        """
        np.random.seed(42)
        # Creating a baseline structural training matrix matching the 9 engineered features
        X_dummy = pd.DataFrame(np.random.rand(100, 9), columns=[
            'Neutrophils', 'Lymphocytes', 'Platelets', 'Hemoglobin', 'RBC', 'NLR', 'PLR', 'SII', 'HRR'
        ])
        # Multi-class target: 0=Healthy, 1=Hematological Malignancy, 2=Solid Tissue Neoplasm
        y_dummy = pd.Series(np.random.choice([0, 1, 2], size=100))
        
        self.xgb_model.fit(X_dummy, y_dummy)
        self.lgb_model.fit(X_dummy, y_dummy)
        self.cat_model.fit(X_dummy, y_dummy)
        
        # Binding the SHAP explainer to the structural XGBoost model core
        self.explainer = shap.TreeExplainer(self.xgb_model)
        self.is_trained = True

    def predict_risk(self, X_patient: pd.DataFrame) -> dict:
        """
        Computes consensus probability distributions across the gradient boosting architectures.
        """
        if not self.is_trained:
            self.train_mock_ensemble()
            
        xgb_proba = self.xgb_model.predict_proba(X_patient)[0]
        lgb_proba = self.lgb_model.predict_proba(X_patient)[0]
        cat_proba = self.cat_model.predict_proba(X_patient)[0]
        
        # Calculating the multi-class consensus weighted probabilities
        consensus_proba = (xgb_proba + lgb_proba + cat_proba) / 3.0
        
        classes = ['Healthy Cohort Profile', 'Hematological Malignancy', 'Solid Tissue Neoplasm']
        pred_class_idx = np.argmax(consensus_proba)
        
        return {
            'predicted_class': classes[pred_class_idx],
            'confidence': consensus_proba[pred_class_idx],
            'probabilities': dict(zip(classes, consensus_proba)),
            'class_index': pred_class_idx
        }

    def get_shap_values(self, X_patient: pd.DataFrame):
        """
        Extracts game-theoretic feature attributions for post-hoc diagnostic transparency.
        """
        if self.explainer is None:
            self.train_mock_ensemble()
        return self.explainer(X_patient)