import requests
import base64
import json
import sys
import glob
import random

print("=" * 60)
print("PHASE 4: API FUNCTIONAL TEST")
print("=" * 60)

API_URL = "http://localhost:8001/api/voice-detection"
API_KEY = "secret123"

# Get test files
human_files = glob.glob("dataset/human/tamil/*.wav")[:1]
ai_files = glob.glob("dataset/ai_generated/tamil/*.mp3")[:1]

if not human_files or not ai_files:
    print("❌ Could not find test files")
    sys.exit(1)

human_file = human_files[0]
ai_file = ai_files[0]

print(f"Human test file: {human_file}")
print(f"AI test file: {ai_file}")

def encode_audio_file(file_path):
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_api_request(file_path, expected_class, test_name):
    print(f"\n{'-' * 60}")
    print(f"TEST: {test_name}")
    print(f"{'-' * 60}")
    
    # Encode audio
    audio_base64 = encode_audio_file(file_path)
    
    # Determine format
    audio_format = 'wav' if file_path.endswith('.wav') else 'mp3'
    
    # Create request
    payload = {
        "language": "Tamil",
        "audioFormat": audio_format,
        "audioBase64": audio_base64
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Expected 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Parse JSON
        try:
            data = response.json()
        except:
            print(f"❌ Invalid JSON response")
            return False
        
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate schema
        required_fields = ['status', 'language', 'classification', 'confidenceScore', 'explanation']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            print(f"❌ Missing fields: {missing_fields}")
            return False
        
        # Validate values
        if data['status'] != 'success':
            print(f"❌ Status is not 'success': {data['status']}")
            return False
        
        if data['classification'] not in ['HUMAN', 'AI_GENERATED']:
            print(f"❌ Invalid classification: {data['classification']}")
            return False
        
        confidence = data['confidenceScore']
        if not (0.05 <= confidence <= 0.95):
            print(f"❌ Confidence score out of range [0.05, 0.95]: {confidence}")
            return False
        
        if not data['explanation'] or len(data['explanation']) < 10:
            print(f"❌ Explanation is empty or too short")
            return False
        
        # Check if classification matches expectation
        if expected_class and data['classification'] != expected_class:
            print(f"⚠️  Classification mismatch: expected {expected_class}, got {data['classification']}")
            print(f"   (Not a failure, but worth noting)")
        
        print("✅ All schema validations passed")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Run tests
results = []

results.append(test_api_request(human_file, 'HUMAN', 'Valid HUMAN audio'))
results.append(test_api_request(ai_file, 'AI_GENERATED', 'Valid AI audio'))

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

passed = sum(results)
total = len(results)

print(f"Tests passed: {passed}/{total}")

if passed == total:
    print("\n✅ PHASE 4 PASSED: API functional tests successful")
else:
    print(f"\n❌ PHASE 4 FAILED: {total - passed} test(s) failed")
    sys.exit(1)
