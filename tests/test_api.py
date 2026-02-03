import requests
import base64
import os

API_URL = "http://localhost:8001/api/voice-detection"
API_KEY = "secret123"

def test_api():
    print(f"Testing API at {API_URL}...")
    
    # Create a dummy MP3 file (1 frame of silence) or just random bytes with mp3 header if possible
    # For simplicity, we'll assume the decoder can handle a small valid MP3 or we use a proper base64 string
    # This is a minimal valid MP3 frame (MPEG 1 Layer 3, 128kbps, 44.1kHz)
    dummy_mp3_hex = "FFFB906400000000000000000000000000000000" 
    # That hex is likely too short/invalid, but let's try a very basic string
    # Note: The decoder in app/audio/decoder.py just writes bytes. librosa.load might fail if invalid.
    # Ideally we need a real file.
    
    # Using a placeholder valid base64 for a silent mp3 (extremely short)
    # Read a real file from the dataset
    try:
        # Try to find a real file
        import glob
        files = glob.glob("dataset/human/**/*.wav", recursive=True)
        if not files:
            files = glob.glob("dataset/human/**/*.mp3", recursive=True)
            
        if files:
            file_path = files[0]
            print(f"Using audio file: {file_path}")
            with open(file_path, "rb") as f:
                audio_data = f.read()
                valid_mp3_b64 = base64.b64encode(audio_data).decode("utf-8")
        else:
             print("No dataset files found, falling back to dummy (might fail)")
             valid_mp3_b64 = "/+MYxAAAAANIAAAAAExBTUUzLjEwMKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqg="
    except Exception as e:
        print(f"Error reading file: {e}")
        valid_mp3_b64 = ""

    payload = {
        "language": "Tamil",
        "audioFormat": "mp3", # The API might ignore this if decoder handles headers, but we send what we have
        "audioBase64": valid_mp3_b64
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Request Successful")
            print(f"   Classification: {data['classification']}")
            print(f"   Confidence: {data['confidenceScore']}")
            print(f"   Explanation: {data['explanation']}")
        else:
            print("❌ API Request Failed")
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Ensure the server is running: uvicorn app.main:app --reload")

if __name__ == "__main__":
    test_api()
