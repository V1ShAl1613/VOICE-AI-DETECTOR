import requests
import base64
import glob

print("=" * 60)
print("MP3 FIX VERIFICATION TEST")
print("=" * 60)

API_URL = "http://localhost:8001/api/voice-detection"
API_KEY = "secret123"

# Get an MP3 file from AI-generated samples
mp3_files = glob.glob("dataset/ai_generated/*/*.mp3")[:1]

if not mp3_files:
    print("❌ No MP3 files found for testing")
    exit(1)

mp3_file = mp3_files[0]
print(f"\nTesting with: {mp3_file}")

# Encode audio
with open(mp3_file, 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# Create request
payload = {
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": audio_base64
}

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

try:
    print("\nSending MP3 request to API...")
    response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ SUCCESS - MP3 file processed")
        print(f"Classification: {data['classification']}")
        print(f"Confidence: {data['confidenceScore']}")
        print(f"Explanation: {data['explanation'][:80]}...")
    else:
        print(f"\n❌ FAILED - Status {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
