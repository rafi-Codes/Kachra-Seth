"""Environment setup and verification script."""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print("✅ Python version is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def verify_directories():
    """Verify that all required directories exist."""
    print("\nVerifying directory structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/annotations",
        "data/synthetic",
        "models/architectures",
        "models/checkpoints",
        "models/onnx",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "configs",
        "notebooks",
        "tests"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def verify_config():
    """Verify that configuration file exists."""
    print("\nVerifying configuration...")
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        print("✅ Configuration file exists")
        return True
    else:
        print("❌ Configuration file missing")
        return False


def main():
    """Main setup function."""
    print("=" * 50)
    print("Stack Count Prediction - Environment Setup")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        verify_directories(),
        verify_config()
    ]
    
    if all(checks):
        print("\n✅ Basic structure verified!")
        
        response = input("\nDo you want to install dependencies now? (y/n): ")
        if response.lower() == 'y':
            if install_dependencies():
                print("\n" + "=" * 50)
                print("🎉 Setup complete! You're ready to start.")
                print("=" * 50)
            else:
                print("\n❌ Setup failed. Please install dependencies manually.")
        else:
            print("\nSkipping dependency installation.")
            print("Run 'pip install -r requirements.txt' when ready.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
