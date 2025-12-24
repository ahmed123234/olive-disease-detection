"""Setup configuration for olive-disease-detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="olive-disease-detection",
    version="1.0.0",
    author="Ahmed Ghannam",
    author_email="your.email@example.com",
    description="Deep learning system for automated olive disease detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/olive-disease-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "Flask>=3.0.0",
        "FastAPI>=0.103.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.3.0",
        "PyYAML>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
        "gpu": [
            "pytorch-cuda>=11.8",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "google-cloud-storage>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "olive-train=src.models.train:main",
            "olive-predict=src.models.predict:main",
            "olive-evaluate=src.models.evaluate:main",
        ],
    },
)
