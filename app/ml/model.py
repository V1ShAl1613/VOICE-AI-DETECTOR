import joblib
import numpy as np
import os
from app.core.config import settings

class ModelLoader:
    _instance = None
    model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance
    
    def load_model(self):
        try:
            if os.path.exists(settings.MODEL_PATH):
                self.model = joblib.load(settings.MODEL_PATH)
                print(f"Model loaded from {settings.MODEL_PATH}")
            else:
                print(f"WARNING: Model not found at {settings.MODEL_PATH}. Inference will fail.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
    def predict(self, features: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        
        # Reshape for single sample
        features_reshaped = features.reshape(1, -1)
        
        # Get probability
        # XGBoost/Sklearn classes_: [0, 1] where 1 is AI
        try:
            # Check if model supports predict_proba
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features_reshaped)
                # Assuming index 1 is positive class (AI)
                score = probs[0][1]
            else:
                # Fallback for models without probability (shouldn't happen with XGB/Logistic)
                score = float(self.model.predict(features_reshaped)[0])
                
            return score
        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")

# Global instance
model_loader = ModelLoader()
