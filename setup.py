import setuptools
from setuptools import Command, find_packages, setup
import os

with open("README.md", "r") as fh:
    long_description = fh.read()


# Get the requirements.txt
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + "/requirements.txt"
install_requires = []  # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()
        print(install_requires)


setuptools.setup(
    name="deeper",
    version="0.0.1",
    author="David Diaz",
    author_email="suvalakisoftware",
    description="A collection of deep learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suvalaki/Deeper",
    packages=find_packages(include=["deeper", "deeper.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    setup_requires=install_requires,
    install_requires=install_requires,
)
