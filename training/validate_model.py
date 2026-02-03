import os
import sys
import numpy as np
import joblib
import glob
from app.core.config import settings

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio.features import extract_features

def validate_model():
    print("--- STEP 3: MODEL SANITY CHECK (OFFLINE) ---")
    
    # 1. Load Model
    if not os.path.exists(settings.MODEL_PATH):
        print(f"❌ CRITICAL FAILURE: Model artifact not found at {settings.MODEL_PATH}")
        print("   Run 'python training/train_model.py' first.")
        sys.exit(1)
        
    try:
        model = joblib.load(settings.MODEL_PATH)
        print(f"✅ Model loaded successfully from {settings.MODEL_PATH}")
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Could not load model: {e}")
        sys.exit(1)

    # 2. Find Test Samples
    # We need at least one human and one AI sample
    # We'll look in the dataset directory itself if available
    human_samples = glob.glob("dataset/human/**/*.mp3", recursive=True) + glob.glob("dataset/human/**/*.wav", recursive=True)
    ai_samples = glob.glob("dataset/ai_generated/**/*.mp3", recursive=True) + glob.glob("dataset/ai_generated/**/*.wav", recursive=True)
    
    if not human_samples or not ai_samples:
        print("⚠️ WARNING: Could not find samples in 'dataset/' for sanity check.")
        print("   Skipping prediction validation. Ensure you verify manually.")
        return

    human_sample = human_samples[0]
    ai_sample = ai_samples[0]
    
    print(f"   Testing Human Sample: {human_sample}")
    print(f"   Testing AI Sample:    {ai_sample}")
    
    # 3. Run Predictions
    threshold = settings.AI_PROBABILITY_THRESHOLD
    
    # Test Human
    try:
        h_feat = extract_features(human_sample).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            h_score = model.predict_proba(h_feat)[0][1]
        else:
            h_score = float(model.predict(h_feat)[0])
            
        print(f"   Human Score: {h_score:.4f} (Threshold: {threshold})")
        
        if h_score >= threshold:
            print("❌ SANITY CHECK FAILED: Human sample classified as AI!")
        else:
            print("✅ Human classified correctly.")
            
    except Exception as e:
        print(f"❌ Error processing human sample: {e}")

    # Test AI
    try:
        a_feat = extract_features(ai_sample).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            a_score = model.predict_proba(a_feat)[0][1]
        else:
            a_score = float(model.predict(a_feat)[0])
            
        print(f"   AI Score:    {a_score:.4f} (Threshold: {threshold})")
        
        if a_score < threshold:
            print("❌ SANITY CHECK FAILED: AI sample classified as Human!")
        else:
            print("✅ AI classified correctly.")
            
    except Exception as e:
        print(f"❌ Error processing AI sample: {e}")

if __name__ == "__main__":
    validate_model()
