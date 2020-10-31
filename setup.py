from BAARD import __version__

from setuptools import setup, find_packages
from os import path

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="BAARD",
    version=__version__,
    description="A simple package for BAARD.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dlon450/BAARD_package",
    packages=find_packages(),
    install_requires=["numpy"]
)