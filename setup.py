import os
from setuptools import setup, find_packages

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(MODULE_DIR, "requirements.txt"), "r") as f:
    requirements = f.read().replace(" ", "").split("\n")

# source of version is in the constants file
VERSION_FILE = os.path.join(MODULE_DIR, "matbench/constants.py")
token = "VERSION = "
with open(VERSION_FILE, "r") as f:
    version = None
    for line in f.readlines():
        if token in line:
            version = line.replace(token, "").strip()
# Double quotes are contained in the read line, remove them
version = version.replace("\"", "")


if __name__ == "__main__":
    setup(
        name='matbench',
        version=version,
        description='a machine learning benchmark for materials science',
        long_description="A machine learning benchmark for materials science. "
                         "https://github.com/materialsproject/matbench",
        url='https://github.com/materialsproject/matbench',
        author=['Alex Dunn', 'Anubhav Jain'],
        author_email='ardunn@lbl.gov',
        license='modified BSD',
        packages=find_packages(where="."),
        package_data={
            "matbench": ["*.json"],
            "matbench.tests": ["*.json"]
        },
        zip_safe=False,
        install_requires=requirements,
        extras_require={},
        test_suite='matbench',
        tests_require='tests',
        include_package_data=True
    )
