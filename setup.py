from setuptools import find_packages, setup

setup(
    name="nano-explorer",
    version="0.1.0",
    description="Modular CLI toolkit for the Waveshare JetBot / Jetson Nano B01",
    author="Your Name",
    # Note: this package is designed and tested for Python 3.6 (JetPack 4.6.1 / Jetson Nano B01).
    # The constraint is relaxed here to allow CI to run on Python 3.10+ since GitHub Actions
    # no longer provides Python 3.6 runners. Do not use this package outside a 3.6 environment
    # in production — behaviour on newer Python versions is not guaranteed.
    python_requires=">=3.6",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "PyYAML>=5.4.1",
        "numpy>=1.19.5",
        "Pillow>=8.4.0",
        "loguru>=0.5.3",
        "colorama>=0.4.4",
        "pynput>=1.7.6",
        "aiohttp>=3.7.4",
        "tqdm>=4.64.1",
        "pupil-apriltags>=1.0.4",
    ],
    scripts=["bin/nano-explorer"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
