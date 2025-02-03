#!/bin/bash

# Stop the script on the first error
set -e

echo "Discovering test files..."
TEST_FILES=$(find tests -name "test_*.py" | sort)

if [ -z "$TEST_FILES" ]; then
    echo "No test files found!"
    exit 1
fi

echo "Running tests sequentially..."
for test_file in $TEST_FILES; do
    # Convert file path to Python module format
    test_module=$(echo "$test_file" | sed 's|/|.|g' | sed 's|.py$||')
    
    echo "Running: $test_module"
    python -m unittest "$test_module"
done

echo "All tests completed successfully!"
