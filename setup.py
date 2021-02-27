import os
from setuptools import setup, find_packages


version = "0.1"
module_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(module_dir, "requirements.txt"), "r") as f:
    requirements = f.read().replace(" ", "").split("\n")

if __name__ == "__main__":
    setup(
        name='matbench',
        version=version,
        description='a machine learning benchmark for materials science',
        long_description="A machine learning benchmark for materials science. "
                         "https://github.com/hackingmaterials/matbench",
        url='https://github.com/hackingmaterials/automatminer',
        author=['Alex Dunn', 'Anubhav Jain'],
        author_email='ardunn@lbl.gov',
        license='modified BSD',
        packages=find_packages(where="."),
        package_data={
            "matbench": ["*.json"]
        },
        zip_safe=False,
        install_requires=requirements,
        extras_require={},
        classifiers=['Programming Language :: Python :: 3.6',
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: System Administrators',
                     'Intended Audience :: Information Technology',
                     'Operating System :: OS Independent',
                     'Topic :: Other/Nonlisted Topic',
                     'Topic :: Scientific/Engineering'],
        test_suite='matbench',
        tests_require='tests',
        include_package_data=True
    )