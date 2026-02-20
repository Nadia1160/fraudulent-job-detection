from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="fake-jobs-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Machine Learning Approach to Detecting Fraudulent Job Types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nadia1160/fraudulent-job-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)