import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

class DataIngestionPipeline:
    """
    Handles data loading, schema validation, and missing value micro-imputation 
    using K-Nearest Neighbors as specified in the HemaPredict AI framework.
    """
    def __init__(self, n_neighbors=5):
        # Micro-imputation via KNN using 5 closest clinical profiles
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def process_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Validates data schema, applies KNN micro-imputation, and performs MinMax rescaling.
        """
        # Ensure all required features exist (handling basic required columns)
        required_columns = ['Neutrophils', 'Lymphocytes', 'Platelets', 'Hemoglobin', 'RBC']
        for col in required_columns:
            if col not in df.columns:
                # Initialize missing columns with NaN if they don't exist
                df[col] = np.nan
                
        # Reorder to maintain strict schema consistency
        df = df[required_columns]
        
        # Execute Imputation
        if is_training:
            imputed_data = self.imputer.fit_transform(df)
            scaled_data = self.scaler.fit_transform(imputed_data)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                # Fallback if predicting single instances without prior training
                imputed_data = self.imputer.fit_transform(df)
                scaled_data = self.scaler.fit_transform(imputed_data)
            else:
                imputed_data = self.imputer.transform(df)
                scaled_data = self.scaler.transform(imputed_data)
                
        return pd.DataFrame(scaled_data, columns=required_columns)


class ClinicalFeatureEngineer:
    """
    Executes domain-knowledge driven feature engineering to calculate novel mathematical indices
    representing systemic immune responses and host-tumor interactions.
    """
    def __init__(self):
        self.epsilon = 1e-9  # Prevents critical division by zero errors

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates NLR, PLR, SII, and HRR mathematical indices from normalized CBC metrics.
        """
        df_engineered = df.copy()
        
        # 1. Neutrophil-to-Lymphocyte Ratio (NLR)
        df_engineered['NLR'] = df_engineered['Neutrophils'] / (df_engineered['Lymphocytes'] + self.epsilon)
        
        # 2. Platelet-to-Lymphocyte Ratio (PLR)
        df_engineered['PLR'] = df_engineered['Platelets'] / (df_engineered['Lymphocytes'] + self.epsilon)
        
        # 3. Systemic Immune-Inflammation Index (SII)
        df_engineered['SII'] = (df_engineered['Platelets'] * df_engineered['Neutrophils']) / (df_engineered['Lymphocytes'] + self.epsilon)
        
        # 4. Hemoglobin-to-Red-Cell Ratio (HRR)
        df_engineered['HRR'] = df_engineered['Hemoglobin'] / (df_engineered['RBC'] + self.epsilon)
        
        return df_engineered

    def select_important_features(self, X: pd.DataFrame, y: pd.Series, top_k: int = 5) -> list:
        """
        Screens engineered features using Mutual Information (MI) scores to remove noisy metrics.
        """
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_scores_series = pd.Series(mi_scores, index=X.columns)
        selected_features = mi_scores_series.nlargest(top_k).index.tolist()
        return selected_features