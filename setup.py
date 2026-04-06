"""
Argentine Financial Distress Prediction Model

A production-ready machine learning pipeline for predicting financial distress
in Argentine publicly traded companies using CNV regulatory data.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="argentine_distress_prediction",
    version="1.0.0",
    author="Pablo Kaegi",
    author_email="pablokaegi@email.com",
    description="Financial distress prediction for Argentine publicly traded companies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pablokaegi/risk_credit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cnv-scraper=src.data_acquisition.cnv_scraper:main",
            "ratio-calculator=src.features.ratio_calculator:main",
            "train-model=src.model.classifier:main",
        ],
    },
)