import requests
import base64
import json
import sys
import glob
import random
import numpy as np

print("=" * 60)
print("PHASE 6: BEHAVIORAL ACCURACY CHECK")
print("=" * 60)

API_URL = "http://localhost:8001/api/voice-detection"
API_KEY = "secret123"

# Get test files (WAV only to avoid Windows file cleanup issues)
human_files = []
ai_files = []

for lang in ["tamil", "english", "hindi", "malayalam", "telugu"]:
    human_lang = glob.glob(f"dataset/human/{lang}/*.wav")
    ai_lang = glob.glob(f"dataset/ai_generated/{lang}/*.wav")  # WAV files only
    
    human_files.extend(human_lang)
    ai_files.extend(ai_lang)

random.seed(42)

# Sample 5 from each
human_sample = random.sample(human_files, min(5, len(human_files)))
ai_sample = random.sample(ai_files, min(5, len(ai_files)))

print(f"Testing {len(human_sample)} human + {len(ai_sample)} AI samples")

def encode_audio_file(file_path):
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_audio(file_path, expected_class, test_num):
    audio_base64 = encode_audio_file(file_path)
    
    payload = {
        "language": "Tamil",  # Language is language-agnostic
        "audioFormat": "wav",
        "audioBase64": audio_base64
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        is_correct = (data['classification'] == expected_class)
        
        filename = file_path.split('\\')[-1][:35]
        status = "✅" if is_correct else "❌"
        
        print(f"{test_num:2d}. {filename:35s} | {data['classification']:12s} | "
              f"Conf: {data['confidenceScore']:.4f} | {status}")
        
        return {
            'file': filename,
            'expected': expected_class,
            'predicted': data['classification'],
            'confidence': data['confidenceScore'],
            'explanation': data['explanation'],
            'correct': is_correct
        }
    except Exception as e:
        print(f"{test_num:2d}. ERROR: {str(e)[:50]}")
        return None

# Run tests
print("\n" + "-" * 80)
print("HUMAN SAMPLES (Expected: HUMAN)")
print("-" * 80)

results = []
test_num = 1

for file_path in human_sample:
    result = test_audio(file_path, 'HUMAN', test_num)
    if result:
        results.append(result)
    test_num += 1

print("\n" + "-" * 80)
print("AI-GENERATED SAMPLES (Expected: AI_GENERATED)")
print("-" * 80)

for file_path in ai_sample:
    result = test_audio(file_path, 'AI_GENERATED', test_num)
    if result:
        results.append(result)
    test_num += 1

# Analysis
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

if len(results) == 0:
    print("❌ No results to analyze")
    sys.exit(1)

correct = sum(1 for r in results if r['correct'])
total = len(results)
accuracy = (correct / total * 100) if total > 0 else 0

print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")

# Check explanation variance
explanations = [r['explanation'] for r in results]
unique_explanations = len(set(explanations))

print(f"Unique explanations: {unique_explanations}/{total}")

if unique_explanations == 1:
    print("⚠️  All explanations are identical - may indicate lack of feature diversity")
elif unique_explanations < total * 0.3:
    print("⚠️  Low explanation variance")
else:
    print("✅ Good explanation variance")

# Check confidence correlation
human_confidences = [r['confidence'] for r in results if r['expected'] == 'HUMAN']
ai_confidences = [r['confidence'] for r in results if r['expected'] == 'AI_GENERATED']

if human_confidences:
    print(f"Human confidence range: {min(human_confidences):.4f} - {max(human_confidences):.4f}")
if ai_confidences:
    print(f"AI confidence range: {min(ai_confidences):.4f} - {max(ai_confidences):.4f}")

# Check confidence is in valid range [0.05, 0.95]
out_of_range = [r for r in results if not (0.05 <= r['confidence'] <= 0.95)]
if out_of_range:
    print(f"⚠️  {len(out_of_range)} confidence scores out of range [0.05, 0.95]")

# Final verdict
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

if accuracy >= 80:
    if total >= 8:
        print(f"✅ PHASE 6 PASSED: {correct}/{total} correct ({accuracy:.1f}% >= 80%)")
    else:
        print(f"⚠️  PHASE 6 PASSED but tested fewer than 10 samples ({total}/10)")
else:
    print(f"❌ PHASE 6 FAILED: {correct}/{total} correct ({accuracy:.1f}% < 80%)")
    sys.exit(1)
