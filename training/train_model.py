import os
import glob
import numpy as np
import joblib
import sys
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from scipy.stats import randint, uniform

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
        
        audio_files = []
        for ext in ['*.mp3', '*.wav']:
            files = glob.glob(os.path.join(cls_path, '**', ext), recursive=True)
            audio_files.extend(files)
            
        print(f"Found {len(audio_files)} files for class {cls_name}")
        
        for file_path in audio_files:
            try:
                # Extract Features (using the enhanced 92-feature extractor)
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
    
    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)
    
    ratio_0 = count_0 / total
    ratio_1 = count_1 / total
    
    print(f"Dataset Distribution: Human: {count_0} ({ratio_0:.1%}), AI: {count_1} ({ratio_1:.1%})")
    
    if ratio_0 < 0.4 or ratio_0 > 0.6:
        print("WARNING: Class imbalance detected! Split is outside 40-60% range.")
        print("Recommendation: Add more samples to the minority class for better performance.")

def train():
    print("=" * 60)
    print("VOICE AI DETECTOR - ENHANCED TRAINING PIPELINE")
    print("=" * 60)
    
    print("\nStep 1: Loading dataset...")
    X, y = load_dataset(DATASET_ROOT)
    
    if len(X) == 0:
        print("ERROR: No training data found in 'dataset/' directory.")
        print("Please populate 'dataset/human' and 'dataset/ai_generated' with MP3/WAV files.")
        sys.exit(1)
        
    check_imbalance(y)
    print(f"Feature vector size: {X.shape[1]}")
    
    # Train/Val Split
    print("\nStep 2: Splitting data (80/20)...")
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("Error: Not enough samples per class. Need at least 2 per class.")
        sys.exit(1)
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Scale features first for all models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # ===================== MODEL TRAINING =====================
    
    best_model = None
    best_score = 0
    best_name = ""
    
    # --- Option 1: XGBoost with Hyperparameter Tuning ---
    try:
        from xgboost import XGBClassifier
        print("\nStep 3a: Training XGBoost with RandomizedSearchCV...")
        
        # Calculate scale_pos_weight for imbalance handling
        counts = Counter(y_train)
        scale_pos_weight = counts.get(0, 1) / max(counts.get(1, 1), 1)
        
        xgb_base = XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 10),
            'scale_pos_weight': [1.0, scale_pos_weight],
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0.5, 2),
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            xgb_base, param_dist,
            n_iter=50,
            cv=cv, scoring='accuracy',
            random_state=42, n_jobs=-1,
            verbose=0
        )
        search.fit(X_train_scaled, y_train)
        
        xgb_model = search.best_estimator_
        xgb_pred = xgb_model.predict(X_val_scaled)
        xgb_acc = accuracy_score(y_val, xgb_pred)
        
        print(f"  XGBoost Best CV Score: {search.best_score_:.4f}")
        print(f"  XGBoost Validation Accuracy: {xgb_acc:.4f}")
        print(f"  Best Params: {search.best_params_}")
        
        if xgb_acc > best_score:
            best_model = xgb_model
            best_score = xgb_acc
            best_name = "XGBoost (Tuned)"
            
    except ImportError:
        print("XGBoost not available, skipping...")
    
    # --- Option 2: Random Forest ---
    print("\nStep 3b: Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_val_scaled)
    rf_acc = accuracy_score(y_val, rf_pred)
    print(f"  Random Forest Validation Accuracy: {rf_acc:.4f}")
    
    if rf_acc > best_score:
        best_model = rf_model
        best_score = rf_acc
        best_name = "Random Forest"
    
    # --- Option 3: Gradient Boosting ---
    print("\nStep 3c: Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_val_scaled)
    gb_acc = accuracy_score(y_val, gb_pred)
    print(f"  Gradient Boosting Validation Accuracy: {gb_acc:.4f}")
    
    if gb_acc > best_score:
        best_model = gb_model
        best_score = gb_acc
        best_name = "Gradient Boosting"
    
    # --- Option 4: Voting Ensemble of all three ---
    print("\nStep 3d: Training Voting Ensemble...")
    try:
        estimators = [('rf', rf_model), ('gb', gb_model)]
        if 'xgb_model' in dir():
            estimators.append(('xgb', xgb_model))
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)
        ens_pred = ensemble.predict(X_val_scaled)
        ens_acc = accuracy_score(y_val, ens_pred)
        print(f"  Ensemble Validation Accuracy: {ens_acc:.4f}")
        
        if ens_acc > best_score:
            best_model = ensemble
            best_score = ens_acc
            best_name = "Voting Ensemble"
    except Exception as e:
        print(f"  Ensemble failed: {e}")
    
    # ===================== EVALUATION =====================
    
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name} (Accuracy: {best_score:.4f})")
    print(f"{'=' * 60}")
    
    # Full evaluation of best model
    y_pred = best_model.predict(X_val_scaled)
    
    try:
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_val_scaled)[:, 1]
            auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0
        else:
            auc = 0
    except Exception:
        auc = 0
        
    print("\n--- Final Validation Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    
    # ===================== SAVE =====================
    
    # Wrap in a pipeline with scaler so inference just works
    final_pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', best_model)
    ])
    
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(final_pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
    print(f"Model type: {best_name}")
    print("Training complete! 🚀")

if __name__ == "__main__":
    train()
