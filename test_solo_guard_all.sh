#!/bin/bash

dir=$(pwd)
guardpath=$dir/guards
hybridpath=$dir/hybrid

for d in "$guardpath"/*; do
    echo ${d##*/}
    uv run test_solo_guard.py -s ${d##*/}
done

for d in "$hybridpath"/*; do
    echo ${d##*/}
    uv run test_solo_guard.py -s --hybrid ${d##*/}
done