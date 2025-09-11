from setuptools import setup, find_packages

setup(
    name="protein-folding-vqe",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "cirq~=1.0",
        "cirq-google~=1.0.dev",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)