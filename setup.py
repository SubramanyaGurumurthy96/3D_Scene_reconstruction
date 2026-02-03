from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

requirements = (ROOT / "requirements.txt").read_text().strip().splitlines()
requirements = [r for r in requirements if r and not r.startswith("#")]

long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="3d-scene-reconstruction",
    version="0.1.0",
    description="MiniLyra 3D scene reconstruction package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src", exclude=("assets", "trial")),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "train-minilyra=scene_reconstruction.train_minilyra:main",
        ],
    },
)
