from setuptools import setup, find_packages

setup(
    name="ten",
    version="0.1.0",
    description="Temporal Eigenstate Networks: Linear-Complexity Sequence Modeling via Spectral Decomposition",
    author="Oluwatosin Afolabi",
    author_email="afolabi@genovotech.com",
    url="https://github.com/genovotechnologies/temporal-eigenstate-networks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "einops",
    ],
    extras_require={
        "train": ["datasets", "transformers", "wandb"],
        "dev": ["pytest", "black", "ruff"],
    },
    license="MIT",
)
