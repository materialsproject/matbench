import os

from setuptools import find_packages, setup

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# source of version is in the constants file
VERSION_FILE = os.path.join(MODULE_DIR, "matbench/constants.py")
token = "VERSION = "
with open(VERSION_FILE) as f:
    for line in f.readlines():
        if token in line:
            version = line.replace(token, "").strip()
# Double quotes are contained in the read line, remove them
version = version.replace('"', "")


setup(
    name="matbench",
    version=version,
    description="a machine learning benchmark for materials science",
    long_description="A machine learning benchmark for materials science. "
    "https://github.com/materialsproject/matbench",
    url="https://github.com/materialsproject/matbench",
    author="Alex Dunn, Anubhav Jain",
    author_email="ardunn@lbl.gov",
    license="modified BSD",
    packages=find_packages(include="matbench/*"),
    package_data={"matbench": ["*.json"]},
    install_requires=[
        "matminer>=0.7.4",
        "scipy>=1.9.0",
        "monty>=2022.4.26",
        "scikit-learn>=1.0.1",
    ],
)
