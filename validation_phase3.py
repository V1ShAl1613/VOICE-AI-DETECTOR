import os
import sys
import glob
import random
import joblib
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.audio.features import extract_features

print("=" * 60)
print("PHASE 3: OFFLINE PREDICTION CONSISTENCY")
print("=" * 60)

# Load model
model_path = "app/ml/voice_auth_model.pkl"
model = joblib.load(model_path)
print(f"✅ Model loaded from {model_path}")

# Set random seed for reproducibility
random.seed(42)

# Select test files
human_files = []
ai_files = []

for lang in ["tamil", "english", "hindi", "malayalam", "telugu"]:
    human_lang = glob.glob(f"dataset/human/{lang}/*.wav") + glob.glob(f"dataset/human/{lang}/*.mp3")
    ai_lang = glob.glob(f"dataset/ai_generated/{lang}/*.wav") + glob.glob(f"dataset/ai_generated/{lang}/*.mp3")
    
    human_files.extend(human_lang)
    ai_files.extend(ai_lang)

# Sample 5 from each
if len(human_files) < 5:
    print(f"⚠️  Only {len(human_files)} human files available (need 5)")
    human_sample = human_files
else:
    human_sample = random.sample(human_files, 5)

if len(ai_files) < 5:
    print(f"⚠️  Only {len(ai_files)} AI files available (need 5)")
    ai_sample = ai_files
else:
    ai_sample = random.sample(ai_files, 5)

print(f"\nTesting with {len(human_sample)} human + {len(ai_sample)} AI samples")

# Default threshold
THRESHOLD = 0.5

# Test predictions
results = []

print("\n" + "-" * 60)
print("HUMAN SAMPLES")
print("-" * 60)

for i, file_path in enumerate(human_sample, 1):
    try:
        features = extract_features(file_path)
        features_array = np.array(features).reshape(1, -1)
        
        # Get probability of being AI (class 1)
        proba = model.predict_proba(features_array)[0]
        ai_score = proba[1]
        
        prediction = model.predict(features_array)[0]
        correct = (prediction == 0)  # 0 = human
        
        filename = os.path.basename(file_path)
        print(f"{i}. {filename[:40]:40s} | AI Score: {ai_score:.4f} | {'✅ CORRECT' if correct else '❌ WRONG'}")
        
        results.append({
            'file': filename,
            'true_label': 'human',
            'ai_score': ai_score,
            'predicted': 'ai' if prediction == 1 else 'human',
            'correct': correct
        })
    except Exception as e:
        print(f"{i}. {os.path.basename(file_path):40s} | ❌ ERROR: {e}")

print("\n" + "-" * 60)
print("AI-GENERATED SAMPLES")
print("-" * 60)

for i, file_path in enumerate(ai_sample, 1):
    try:
        features = extract_features(file_path)
        features_array = np.array(features).reshape(1, -1)
        
        # Get probability of being AI (class 1)
        proba = model.predict_proba(features_array)[0]
        ai_score = proba[1]
        
        prediction = model.predict(features_array)[0]
        correct = (prediction == 1)  # 1 = AI
        
        filename = os.path.basename(file_path)
        print(f"{i}. {filename[:40]:40s} | AI Score: {ai_score:.4f} | {'✅ CORRECT' if correct else '❌ WRONG'}")
        
        results.append({
            'file': filename,
            'true_label': 'ai',
            'ai_score': ai_score,
            'predicted': 'ai' if prediction == 1 else 'human',
            'correct': correct
        })
    except Exception as e:
        print(f"{i}. {os.path.basename(file_path):40s} | ❌ ERROR: {e}")

# Calculate metrics
total = len(results)
correct = sum(1 for r in results if r['correct'])
accuracy = (correct / total * 100) if total > 0 else 0

human_scores = [r['ai_score'] for r in results if r['true_label'] == 'human']
ai_scores = [r['ai_score'] for r in results if r['true_label'] == 'ai']

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
print(f"Human AI scores (should be < 0.5): {[f'{s:.3f}' for s in human_scores]}")
print(f"AI AI scores (should be > 0.5):    {[f'{s:.3f}' for s in ai_scores]}")

# Check if predictions are random
all_scores = [r['ai_score'] for r in results]
avg_score = np.mean(all_scores)
std_score = np.std(all_scores)

print(f"\nScore statistics:")
print(f"  Mean: {avg_score:.4f}")
print(f"  Std Dev: {std_score:.4f}")

# Validation checks
issues = []

if accuracy < 80:
    issues.append(f"❌ Accuracy too low: {accuracy:.1f}% < 80%")

if std_score < 0.1:
    issues.append(f"⚠️  Low variance in scores ({std_score:.4f}) - may indicate random predictions")

if 0.45 <= avg_score <= 0.55 and std_score < 0.15:
    issues.append("⚠️  Scores clustered around 0.5 - model may not be trained")

if len(human_scores) > 0:
    avg_human = np.mean(human_scores)
    if avg_human > THRESHOLD:
        issues.append(f"⚠️  Human samples have high average AI score: {avg_human:.3f}")

if len(ai_scores) > 0:
    avg_ai = np.mean(ai_scores)
    if avg_ai < THRESHOLD:
        issues.append(f"⚠️  AI samples have low average AI score: {avg_ai:.3f}")

if issues:
    print("\n❌ ISSUES DETECTED:")
    for issue in issues:
        print(f"  {issue}")
    
    if accuracy < 80:
        print("\n❌ PHASE 3 FAILED: MODEL BEHAVIOR INVALID – LIKELY NOT TRAINED")
        sys.exit(1)
    else:
        print("\n⚠️  PHASE 3 PASSED WITH WARNINGS")
else:
    print("\n✅ PHASE 3 PASSED: Model predictions are consistent and accurate")
