"""Setup configuration for cuttle package."""

from setuptools import setup, find_packages

setup(
    name="cuttle",
    version="0.1.0",
    description="Cuttle card game environment with DQN training",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "gymnasium",
        "torch",
    ],
)

