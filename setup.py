"""Setup script for segshort package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies needed for the package
# Users should install from requirements.txt for full functionality
core_requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "Pillow>=8.0.0",
    "tqdm>=4.60.0",
    "pycocotools>=2.0.0",
    "segmentation-models-pytorch>=0.3.0",
]

REPOSITORY_URL = "https://github.com/acharaakshit/label-flips"

setup(
    name="segshort",
    version="0.1.0",
    author="Akshit Achara, Yovin Yahathugoda",
    description="Research codebase for studying spurious correlations in semantic segmentation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=REPOSITORY_URL,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        # "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "segshort-generate-waterbirds=segshort.dataset_generation.generate_waterbirds:main",
            "segshort-generate-coco-cd=segshort.dataset_generation.generate_coco_cd:main",
            "segshort-train=segshort.training.train_cv:main",
            "segshort-eval=segshort.training.test_models:main",
        ],
    },
)
