import os
import glob
import numpy as np
import joblib
import sys
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Add parent dir to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio.features import extract_features

# Configuration
DATASET_ROOT = "dataset"
MODEL_OUTPUT_PATH = "app/ml/voice_auth_model.pkl"

def load_dataset(root_path):
    features_list = []
    labels_list = []
    
    # Structure: dataset/human/lang/*.mp3, dataset/ai_generated/lang/*.mp3
    # Mappings: human -> 0, ai_generated -> 1
    
    classes = {"human": 0, "ai_generated": 1}
    languages = ["tamil", "english", "hindi", "malayalam", "telugu"]
    
    total_files = 0
    
    if not os.path.exists(root_path):
        print(f"Error: Dataset root {root_path} does not exist.")
        return np.array([]), np.array([])
    
    for cls_name, label in classes.items():
        cls_path = os.path.join(root_path, cls_name)
        if not os.path.exists(cls_path):
            print(f"Warning: Class directory {cls_path} not found.")
            continue
            
        print(f"Scanning {cls_name} directory...")
        
        # We can look for files directly or inside language subfolders
        # The prompt specifies dataset/human/tamil/*.mp3 etc.
        # So we walk recursively
        
        audio_files = []
        for ext in ['*.mp3', '*.wav']:
            # Search strictly within language subfolders as per structure, or just recursive
            files = glob.glob(os.path.join(cls_path, '**', ext), recursive=True)
            audio_files.extend(files)
            
        print(f"Found {len(audio_files)} files for class {cls_name}")
        
        for file_path in audio_files:
            try:
                # Extract Features
                feat = extract_features(file_path)
                features_list.append(feat)
                labels_list.append(label)
                total_files += 1
                
                # Progress logging
                if total_files % 10 == 0:
                    print(f"Processed {total_files} files...", end='\r')
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                
    print(f"\nTotal files processed: {total_files}")
    return np.array(features_list), np.array(labels_list)

def check_imbalance(y):
    if len(y) == 0:
        return
    
    counts = Counter(y)
    total = len(y)
    
    # Count 0 (Human) and 1 (AI)
    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)
    
    ratio_0 = count_0 / total
    ratio_1 = count_1 / total
    
    print(f"Dataset Distribution: Human: {count_0} ({ratio_0:.1%}), AI: {count_1} ({ratio_1:.1%})")
    
    if ratio_0 < 0.4 or ratio_0 > 0.6:
        print("WARNING: Class imbalance detected! Split is outside 40-60% range.")
        print("Recommendation: Add more samples to the minority class for better performance.")

def train():
    print("Initializing training pipeline...")
    X, y = load_dataset(DATASET_ROOT)
    
    if len(X) == 0:
        print("ERROR: No training data found in 'dataset/' directory.")
        print("Please populate 'dataset/human' and 'dataset/ai_generated' with MP3/WAV files.")
        print("Exiting without training.")
        sys.exit(1)
        
    check_imbalance(y)
    
    # Train/Val Split
    print("Splitting data...")
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print("Error during split: Likely not enough samples per class. Need at least 2 per class.")
        sys.exit(1)
    
    # Define Model Pipeline
    clf = None
    try:
        from xgboost import XGBClassifier
        print("Using XGBoost Classifier")
        clf = XGBClassifier(
            n_estimators=200, 
            max_depth=6, 
            learning_rate=0.05, 
            subsample=0.8,
            eval_metric='logloss',
        )
    except ImportError:
        print("XGBoost not found, falling back to Logistic Regression")
        clf = LogisticRegression(class_weight='balanced')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    if len(X_val) > 0:
        y_pred = pipeline.predict(X_val)
        try:
            # Check if model has predict_proba
            if hasattr(clf, "predict_proba"):
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0
            else:
                auc = 0
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            auc = 0
            
        print("--- Validation Metrics ---")
        print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
        print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
        print(f"ROC AUC:   {auc:.4f}")
    
    # Save
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train()

