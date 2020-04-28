#!/bin/bash

rm -r -f dist/ build/
python setup.py bdist_wheel
pip install --upgrade --force-reinstall dist/dynalearn-0.2-py3-none-any.whl
