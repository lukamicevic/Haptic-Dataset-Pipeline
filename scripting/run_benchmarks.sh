#!/bin/bash
# Benchmark runner for Rust vs Python signal processing

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
INPUT_DIR="$PROJECT_DIR/test-signals/inputSignals"
TMP_OUTPUT="/tmp/bench_output.wav"

# Check for hyperfine
if ! command -v hyperfine > /dev/null 2>&1; then
    echo "Error: hyperfine not found. Install with: brew install hyperfine"
    exit 1
fi

# Check for rust_signals module
if ! python -c "import rust_signals" 2>/dev/null; then
    echo "Error: rust_signals module not found."
    echo "Build with: cd $PROJECT_DIR/rust-signals && maturin develop --release"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# Define test files for each size
INPUT1_SMALL="$INPUT_DIR/1-9887-A-49.wav"
INPUT2_SMALL="$INPUT_DIR/1-19840-A-36.wav"
INPUT1_MEDIUM="$INPUT_DIR/bench_medium_1.wav"
INPUT2_MEDIUM="$INPUT_DIR/bench_medium_2.wav"
INPUT1_LARGE="$INPUT_DIR/bench_large_1.wav"
INPUT2_LARGE="$INPUT_DIR/bench_large_2.wav"

POSITION=50000
BALANCE=0.5

echo "=== Rust vs Python Signal Processing Benchmark ==="
echo ""

run_benchmark() {
    local size=$1
    local input1=$2
    local input2=$3

    if [ ! -f "$input1" ] || [ ! -f "$input2" ]; then
        echo "Skipping $size: files not found"
        return
    fi

    echo "--- Benchmarking $size files ---"
    echo "  Input 1: $input1"
    echo "  Input 2: $input2"
    echo ""

    for op in mix insert; do
        echo "Running $op benchmark ($size)..."

        RESULT_FILE="$RESULTS_DIR/${op}_${size}.md"

        hyperfine \
            --warmup 3 \
            --runs 10 \
            --export-markdown "$RESULT_FILE" \
            --command-name "Python ($op)" \
            "python $SCRIPT_DIR/bench_python.py --op $op --pos $POSITION --balance $BALANCE --input1 $input1 --input2 $input2 --output $TMP_OUTPUT" \
            --command-name "Rust ($op)" \
            "python $SCRIPT_DIR/bench_rust.py --op $op --pos $POSITION --balance $BALANCE --input1 $input1 --input2 $input2 --output $TMP_OUTPUT"

        echo "Results saved: $RESULT_FILE"
        echo ""
    done
}

# Run benchmarks for each size
run_benchmark "small" "$INPUT1_SMALL" "$INPUT2_SMALL"
run_benchmark "medium" "$INPUT1_MEDIUM" "$INPUT2_MEDIUM"
run_benchmark "large" "$INPUT1_LARGE" "$INPUT2_LARGE"

# Cleanup
rm -f "$TMP_OUTPUT"

echo "=== Benchmark Complete ==="
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Summary of result files:"
ls -la "$RESULTS_DIR"
