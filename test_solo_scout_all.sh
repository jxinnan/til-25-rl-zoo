#!/bin/bash

dir=$(pwd)
scoutpath=$dir/scouts
hybridpath=$dir/hybrid

for d in "$scoutpath"/*; do
    echo ${d##*/}
    uv run test_solo_scout.py -s -g ${d##*/}
done

for d in "$hybridpath"/*; do
    echo ${d##*/}
    uv run test_solo_scout.py -s -g --hybrid ${d##*/}
done