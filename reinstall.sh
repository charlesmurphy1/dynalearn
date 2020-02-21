#!/bin/bash

rm -r -f dist/ build/
python setup.py bdist_wheel
pip install --upgrade --force-reinstall dist/dynalearn-0.1-py3-none-any.whl
