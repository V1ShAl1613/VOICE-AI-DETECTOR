import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.audio.decoder import decode_audio, cleanup_temp_dir
from app.audio.features import extract_features
import glob

print("=" * 60)
print("DETAILED MP3 DEBUG TEST")
print("=" * 60)

# Get an MP3 file
mp3_files = glob.glob("dataset/ai_generated/*/*.mp3")[:1]

if not mp3_files:
    print("❌ No MP3 files found")
    exit(1)

mp3_file = mp3_files[0]
print(f"\nTesting: {mp3_file}")

# Read and encode
with open(mp3_file, 'rb') as f:
    import base64
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

print(f"Encoded size: {len(audio_base64)} chars")

# Step 1: Decode
print("\n1. Decoding audio...")
try:
    temp_path, temp_dir = decode_audio(audio_base64)
    print(f"   ✅ Decoded to: {temp_path}")
    print(f"   Temp dir: {temp_dir}")
    print(f"   File exists: {os.path.exists(temp_path)}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Extract features
print("\n2. Extracting features...")
try:
    features = extract_features(temp_path)
    print(f"   ✅ Extracted {len(features)} features")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    cleanup_temp_dir(temp_dir)
    exit(1)

# Step 3: Cleanup
print("\n3. Cleaning up...")
try:
    cleanup_temp_dir(temp_dir)
    print(f"   ✅ Cleaned up")
    print(f"   Temp dir exists: {os.path.exists(temp_dir)}")
except Exception as e:
    print(f"   ⚠️  Cleanup issue: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ ALL STEPS COMPLETED")
