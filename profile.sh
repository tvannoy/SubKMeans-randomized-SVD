#!/bin/sh

mkdir profile_results

./profiletest.py plane > profile_results/plane.txt
./profiletest.py oliveoil > profile_results/oliveoil.txt
./profiletest.py symbols > profile_results/symbols.txt
./profiletest.py starlightcurves > profile_results/starlightcurves.txt
./profiletest.py drivface > profile_results/drivface.txt
./profiletest.py rnaseq > profile_results/rnaseq.txt
