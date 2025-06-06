#!/bin/bash

until [ -f logs/test_solo_scout/calgary_004.npz ]
do
    sleep 60
done
uv run test_solo_guard.py -s banqiao_4M
    