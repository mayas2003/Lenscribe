#!/usr/bin/env python3
"""
Installation script for voice processing dependencies.

This script helps install and verify the required dependencies for voice synthesis
with emotion control and voice cloning capabilities.
"""

import subprocess
import sys
import os
import importlib


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Installing {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to install {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"[OK] {package_name} is available")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} is not available")
        return False


def install_voice_dependencies():
    """Install voice processing dependencies."""
    print("Voice Processing Dependencies Installation")
    print("=" * 50)
    
    # Core dependencies
    dependencies = [
        ("TTS", "TTS", "pip install TTS"),
        ("torchaudio", "torchaudio", "pip install torchaudio"),
        ("librosa", "librosa", "pip install librosa"),
        ("soundfile", "soundfile", "pip install soundfile"),
        ("speechrecognition", "speech_recognition", "pip install SpeechRecognition"),
        ("pydub", "pydub", "pip install pydub"),
    ]
    
    # Optional dependencies
    optional_dependencies = [
        ("pyaudio", "pyaudio", "pip install pyaudio"),
        ("webrtcvad", "webrtcvad", "pip install webrtcvad"),
    ]
    
    print("\nInstalling core dependencies...")
    core_success = True
    for package, import_name, install_cmd in dependencies:
        if not check_package(package, import_name):
            if not run_command(install_cmd, package):
                core_success = False
    
    print("\nInstalling optional dependencies...")
    optional_success = True
    for package, import_name, install_cmd in optional_dependencies:
        if not check_package(package, import_name):
            if not run_command(install_cmd, package):
                optional_success = False
    
    # Test voice processor
    print("\nTesting voice processor...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from models.voice_processor import VoiceProcessor
        print("[OK] VoiceProcessor can be imported")
        
        # Test initialization (this might fail if TTS is not properly installed)
        try:
            vp = VoiceProcessor()
            print("[OK] VoiceProcessor can be initialized")
        except Exception as e:
            print(f"[WARN] VoiceProcessor initialization failed: {e}")
            print("  This is expected if TTS is not properly installed")
        
    except ImportError as e:
        print(f"[FAIL] VoiceProcessor import failed: {e}")
        core_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Installation Summary:")
    print(f"Core dependencies: {'[OK] Success' if core_success else '[FAIL] Failed'}")
    print(f"Optional dependencies: {'[OK] Success' if optional_success else '[WARN] Some failed'}")
    
    if core_success:
        print("\n[SUCCESS] Voice processing is ready to use!")
        print("\nNext steps:")
        print("1. Run: python examples/voice_example.py")
        print("2. Check: docs/voice_integration_guide.md")
        print("3. Test: python tests/test_voice_processor.py")
    else:
        print("\n[ERROR] Some core dependencies failed to install.")
        print("Please check the error messages above and try again.")
        print("\nManual installation:")
        print("pip install TTS torchaudio librosa soundfile")
    
    return core_success


def main():
    """Main installation function."""
    print("Lenscribe Voice Processing Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    success = install_voice_dependencies()
    
    if success:
        print("\n[SUCCESS] Voice processing setup completed successfully!")
        return 0
    else:
        print("\n[ERROR] Voice processing setup failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
