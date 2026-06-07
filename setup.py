from setuptools import setup, find_packages

setup(
    name="stack-count-prediction",
    version="1.0.0",
    description="Deep learning model for counting items in stacked objects",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "albumentations>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "onnxruntime>=1.16.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "demo": ["gradio>=4.0.0", "streamlit>=1.28.0"],
    },
)
