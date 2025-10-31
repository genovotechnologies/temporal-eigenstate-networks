from setuptools import setup, find_packages
import os

# Read version from src/__init__.py
def get_version():
    init_file = os.path.join(os.path.dirname(__file__), 'src', '__init__.py')
    with open(init_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="temporal-eigenstate-networks",
    version=get_version(),
    author="Oluwatosin Afolabi",
    author_email="afolabi@genovotech.com",
    description="Temporal Eigenstate Networks: Linear-Complexity Sequence Modeling via Spectral Decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/genovotechnologies/temporal-eigenstate-networks",
    project_urls={
        "Bug Tracker": "https://github.com/genovotechnologies/temporal-eigenstate-networks/issues",
        "Documentation": "https://github.com/genovotechnologies/temporal-eigenstate-networks#readme",
        "Source Code": "https://github.com/genovotechnologies/temporal-eigenstate-networks",
    },
    packages=find_packages(exclude=["tests*", "examples*", "scripts*", "paper*"]),
    package_data={
        "src": ["py.typed"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Private :: Do Not Upload",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "full": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "scipy>=1.10.0",
            "tqdm>=4.65.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
    },
    license="Proprietary",
    keywords="deep-learning transformers attention sequence-modeling eigenstate neural-networks spectral-decomposition",
    zip_safe=False,
)
