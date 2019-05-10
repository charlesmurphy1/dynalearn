#!/bin/bash

for d in `find . -maxdepth 1 -type d`; do 
    bash ${d}/launch.sh
done