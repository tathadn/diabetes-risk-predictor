"""
Setup script for the diabetes_prediction package.

Install in development mode with:
    pip install -e .
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="diabetes_prediction",
    version="1.0.0",
    author="Federated ML Project",
    author_email="",
    description=(
        "Diabetes prediction using the Diabetes Health Indicators Dataset "
        "(CDC BRFSS). Covers EDA, preprocessing, multi-model training, "
        "evaluation, and final predictions."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "pylint>=2.10.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={},
)
