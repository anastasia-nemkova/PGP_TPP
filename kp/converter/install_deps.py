import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Installing required Python packages...")
    
    required = [
        'numpy',
        'imageio',
        'imageio[ffmpeg]',
        'scikit-image',
        'opencv-python',
    ]
    
    for package in required:
        try:
            __import__(package.replace('-', '_').split('[')[0])
            print(f"{package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            install(package)

if __name__ == "__main__":
    main()