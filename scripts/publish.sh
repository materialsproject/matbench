#!/bin/bash


rm -rf dist build
python3 setup.py sdist bdist_wheel
twine upload dist/* --verbose

