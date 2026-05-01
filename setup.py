from setuptools import setup, find_packages

setup(
    name="image-forgery-detection",
    version="0.1.0",
    description="Deep learning-based image forgery detection and localization",
    author="Md Yeasin Arafat",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "albumentations>=1.3.0",
        "timm>=0.9.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "tensorboard>=2.14.0", "seaborn>=0.12.0"],
    },
)
