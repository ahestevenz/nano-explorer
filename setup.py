from setuptools import find_packages, setup

# name/version duplicated here for setuptools < 61 (Python 3.6 / Jetson Nano).
# Canonical metadata lives in pyproject.toml [project].
setup(
    name="nano-explorer",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    scripts=["bin/nano-explorer"],
)
