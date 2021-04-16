#!/bin/bash

# Check formatting and linting

flake8_errors=$(flake8 matbench)
isort_errors=$(isort --check-only matbench)

if [ -z "${flake8_errors// }" ] && [ -z "${isort_errors// }" ]
  then
    echo "Code is well-formatted."
    exit 0
  else
    >&2 echo "Code misformatted!"
    >&2 echo "$flake8_errors"
    >&2 echo "$isort_errors"
    exit 1
fi
