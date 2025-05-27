for d in scouts; do
    uv run test_solo_scout.py -g $d
done

for d in hybrid; do
    uv run test_solo_scout.py -g --hybrid $d
done