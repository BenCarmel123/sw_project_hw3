#!/bin/bash
# Usage: ./sym_comp.sh <goal> <test_number> [--full]
# Example: ./sym_comp.sh sym 6
# Example (full output): ./sym_comp.sh ddg 7 --full

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <goal> <test_number> [--full]"
    exit 1
fi

GOAL=$1
TEST_NUM=$2
MODE=$3

# Build input path from test number
INPUT="/Users/bcarmel/Downloads/project-tests-v8-27fca/tests/input_${TEST_NUM}.txt"

# Decide how many lines to show
if [ "$MODE" == "--full" ]; then
    HEAD_CMD="cat"
else
    HEAD_CMD="head -n 10"
fi

echo "=== C output ($GOAL, input_${TEST_NUM}.txt) ==="
C_OUTPUT=$(./symnmf $GOAL "$INPUT" | $HEAD_CMD)
echo "$C_OUTPUT"

echo
echo "=== Python output ($GOAL, input_${TEST_NUM}.txt) ==="
PY_OUTPUT=$(python3 symnmf.py 2 $GOAL "$INPUT" | $HEAD_CMD)
echo "$PY_OUTPUT"

echo
echo "=== Differences (diff) ==="
diff <(echo "$C_OUTPUT") <(echo "$PY_OUTPUT") || echo "✅ No differences!"
