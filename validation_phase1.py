import joblib
import sys
import os

print("=" * 60)
print("PHASE 1: MODEL VERIFICATION")
print("=" * 60)

model_path = "app/ml/voice_auth_model.pkl"

# Check if file exists
if not os.path.exists(model_path):
    print(f"❌ FAIL: Model file not found at {model_path}")
    print("ERROR: MODEL NOT TRAINED – model file missing")
    sys.exit(1)

print(f"✅ Model file exists: {model_path}")

# Get file stats
file_size = os.path.getsize(model_path)
print(f"   Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

# Try to load the model
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ FAIL: Could not load model")
    print(f"ERROR: MODEL FILE CORRUPTED OR INVALID")
    print(f"Exception: {e}")
    sys.exit(1)

# Check model type
print(f"   Model type: {type(model)}")

# If it's a pipeline, check the classifier
if hasattr(model, 'named_steps'):
    print("   Pipeline detected:")
    for step_name, step in model.named_steps.items():
        print(f"     - {step_name}: {type(step).__name__}")
    
    if 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
        classifier_type = type(classifier).__name__
        print(f"   Classifier: {classifier_type}")
        
        # Check if it's XGBoost or LogisticRegression
        if 'XGB' in classifier_type or 'LogisticRegression' in classifier_type:
            print(f"✅ Valid classifier type: {classifier_type}")
        else:
            print(f"⚠️  Unexpected classifier type: {classifier_type}")
else:
    print(f"   Direct model (not a pipeline): {type(model).__name__}")

print("\n✅ PHASE 1 PASSED: Model artifact is valid and loadable")
