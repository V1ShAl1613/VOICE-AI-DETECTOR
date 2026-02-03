import os
import glob
import sys

print("=" * 60)
print("PHASE 2: DATASET SANITY CHECK")
print("=" * 60)

dataset_root = "dataset"

if not os.path.exists(dataset_root):
    print(f"❌ FAIL: Dataset directory '{dataset_root}' not found")
    sys.exit(1)

print(f"✅ Dataset root exists: {dataset_root}")

# Expected structure
classes = ["human", "ai_generated"]
languages = ["tamil", "english", "hindi", "malayalam", "telugu"]

total_files = 0
class_counts = {}
issues = []

for cls in classes:
    cls_path = os.path.join(dataset_root, cls)
    
    if not os.path.exists(cls_path):
        issues.append(f"❌ Missing class directory: {cls}")
        continue
    
    print(f"\n📁 Class: {cls}")
    class_total = 0
    
    for lang in languages:
        lang_path = os.path.join(cls_path, lang)
        
        if not os.path.exists(lang_path):
            issues.append(f"❌ Missing language directory: {cls}/{lang}")
            print(f"   {lang}: ❌ MISSING")
            continue
        
        # Count audio files
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.MP3', '*.WAV']:
            audio_files.extend(glob.glob(os.path.join(lang_path, ext)))
        
        file_count = len(audio_files)
        class_total += file_count
        
        if file_count == 0:
            issues.append(f"⚠️  Empty directory: {cls}/{lang}")
            print(f"   {lang}: ⚠️  EMPTY (0 files)")
        else:
            print(f"   {lang}: ✅ {file_count} files")
    
    class_counts[cls] = class_total
    total_files += class_total
    print(f"   Total for {cls}: {class_total} files")

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Human samples: {class_counts.get('human', 0)}")
print(f"AI-generated samples: {class_counts.get('ai_generated', 0)}")
print(f"Total samples: {total_files}")

# Calculate distribution
if total_files > 0:
    human_pct = (class_counts.get('human', 0) / total_files) * 100
    ai_pct = (class_counts.get('ai_generated', 0) / total_files) * 100
    print(f"\nDistribution:")
    print(f"  Human: {human_pct:.1f}%")
    print(f"  AI-generated: {ai_pct:.1f}%")

# Check for critical issues
critical_fail = False

if class_counts.get('human', 0) == 0:
    print(f"\n❌ CRITICAL: No human samples found!")
    critical_fail = True

if class_counts.get('ai_generated', 0) == 0:
    print(f"\n❌ CRITICAL: No AI-generated samples found!")
    critical_fail = True

if total_files < 10:
    print(f"\n❌ CRITICAL: Insufficient samples ({total_files} < 10)")
    critical_fail = True

# Display issues
if issues:
    print(f"\n⚠️  Issues found ({len(issues)}):")
    for issue in issues:
        print(f"  {issue}")

if critical_fail:
    print("\n❌ PHASE 2 FAILED: Critical dataset issues detected")
    sys.exit(1)
else:
    print("\n✅ PHASE 2 PASSED: Dataset structure is valid")
