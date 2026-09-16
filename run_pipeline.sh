#!/usr/bin/env bash
set -e

# Detect Python binary
if [ -x "/usr/local/bin/python3" ]; then
  PYTHON_CMD="/usr/local/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  PYTHON_CMD="python"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "    Hiver SDE Take-Home: @Uber_Support Production AI Pipeline    "
echo "=================================================================="
echo "Working Directory: $SCRIPT_DIR"
echo "Using Python: $PYTHON_CMD"
echo ""

# Stage 1: Data Extraction & Golden Set
echo ">>> STAGE 1: Dataset Extraction & Golden Set Construction (@Uber_Support)..."
if [ ! -f "data/processed/golden_set.jsonl" ]; then
  $PYTHON_CMD src/data_processor.py
else
  echo "Golden set already exists at data/processed/golden_set.jsonl (200 real examples)."
fi
echo ""

# Stage 2: Baselines
echo ">>> STAGE 2: Running Trivial & Simple Baselines..."
$PYTHON_CMD src/baselines.py
echo ""

# Stage 3: Agent Inference Pipeline
echo ">>> STAGE 3: Executing Proposed Agent Inference Pipeline..."
$PYTHON_CMD src/pipeline.py
echo ""

# Stage 4: Evaluation & LLM Judge + 95% Bootstrap CIs
echo ">>> STAGE 4: Running Evaluation Harness & LLM-as-a-Judge..."
$PYTHON_CMD src/evaluate.py
echo ""

# Stage 5: Human-Judge Agreement Verification
echo ">>> STAGE 5: Running Human-LLM Judge Correlation Analysis..."
$PYTHON_CMD src/human_agreement.py
echo ""

# Stage 6: Multi-Class Confusion Matrix Generation
echo ">>> STAGE 6: Generating Multi-Class Confusion Matrix..."
$PYTHON_CMD src/confusion_matrix.py
echo ""

# Stage 7: Operational Business Cost Optimization
echo ">>> STAGE 7: Running Operational Business Cost Analysis..."
$PYTHON_CMD src/cost_analysis.py
echo ""

echo "=================================================================="
echo " Pipeline Finished Successfully! All Results Available in results/"
echo "=================================================================="
