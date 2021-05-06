#!/bin/bash

# Intended to be run from the root directory, ../


rm -rf dist build
python3 setup.py sdist bdist_wheel
twine upload dist/* --verbose

