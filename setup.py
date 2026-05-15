from setuptools import setup, find_packages

setup(
    name="sentiment-analysis-tool",
    version="0.1.0",
    description="Compare WordNet, SenticNet, and RoBERTa sentiment analysis methods",
    author="Dem00star",
    url="https://github.com/Dem00star/Sentimental-analysis-tool",
    packages=find_packages(exclude=["tests*", "notebooks*", "data*"]),
    python_requires=">=3.8",
    install_requires=[
        "nltk>=3.8",
        "transformers>=4.40",
        "torch>=2.0",
        "senticnet>=1.6",
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
    ],
    entry_points={
        "console_scripts": [
            "sentiment=main:cli",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
