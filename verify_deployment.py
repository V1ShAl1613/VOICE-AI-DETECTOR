#!/usr/bin/env python3
"""
Deployment Verification Script
Run this to confirm your local code is correct before debugging Render.
"""

import os
import sys

def check_file_exists(path, should_exist=True):
    """Check if file exists and matches expectation"""
    exists = os.path.exists(path)
    status = "✅" if exists == should_exist else "❌"
    expectation = "should exist" if should_exist else "should NOT exist"
    print(f"{status} {path} - {expectation}: {'PASS' if exists == should_exist else 'FAIL'}")
    return exists == should_exist

def check_file_content(path, search_string, should_contain=True):
    """Check if file contains specific string"""
    try:
        with open(path, 'r') as f:
            content = f.read()
            contains = search_string in content
            status = "✅" if contains == should_contain else "❌"
            expectation = "should contain" if should_contain else "should NOT contain"
            print(f"{status} {path} {expectation} '{search_string}': {'PASS' if contains == should_contain else 'FAIL'}")
            return contains == should_contain
    except FileNotFoundError:
        print(f"❌ {path} - FILE NOT FOUND")
        return False

def main():
    print("=" * 60)
    print("DEPLOYMENT VERIFICATION - LOCAL CODE CHECK")
    print("=" * 60)
    print()
    
    all_passed = True
    
    print("1. Checking critical files...")
    all_passed &= check_file_exists("app/audio/core_features.py", should_exist=True)
    all_passed &= check_file_exists("app/audio/features.py", should_exist=False)
    all_passed &= check_file_exists("app/ml/voice_auth_model.pkl", should_exist=True)
    all_passed &= check_file_exists(".dockerignore", should_exist=True)
    print()
    
    print("2. Checking imports...")
    all_passed &= check_file_content("app/api/routes.py", "from app.audio.core_features import extract_features", should_contain=True)
    all_passed &= check_file_content("app/api/routes.py", "from app.audio.features import", should_contain=False)
    print()
    
    print("3. Checking dependencies...")
    all_passed &= check_file_content("requirements.txt", "pydub", should_contain=False)
    all_passed &= check_file_content("requirements.txt", "librosa", should_contain=True)
    print()
    
    print("4. Checking .dockerignore...")
    # Check for wildcard *.pkl pattern (not in comments)
    try:
        with open(".dockerignore", 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            has_wildcard_pkl = any(line == "*.pkl" or line == "!app/ml/voice_auth_model.pkl" for line in lines)
            status = "✅" if not has_wildcard_pkl else "❌"
            print(f"{status} .dockerignore should NOT have '*.pkl' wildcard: {'PASS' if not has_wildcard_pkl else 'FAIL'}")
            all_passed &= not has_wildcard_pkl
    except FileNotFoundError:
        print(f"❌ .dockerignore - FILE NOT FOUND")
        all_passed = False
    all_passed &= check_file_content(".dockerignore", "dataset/*.pkl", should_contain=True)
    print()
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Local code is CORRECT")
        print("The issue is with Render's deployment configuration.")
        print("Follow RENDER_CONFIG_FIX.md to fix Render settings.")
    else:
        print("❌ SOME CHECKS FAILED - Local code has issues")
        print("Fix local code first before debugging Render.")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
