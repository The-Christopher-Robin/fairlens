from setuptools import setup, find_packages

setup(
    name="fairlens",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "fairlens=scripts.run_eval:main",
        ],
    },
)
