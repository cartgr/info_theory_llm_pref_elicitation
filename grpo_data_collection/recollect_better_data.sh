#!/bin/bash

# Re-collect GRPO data with better prompts and larger model

echo "============================================================"
echo "Re-collecting GRPO data with improved prompts"
echo "============================================================"
echo "Using: Qwen2.5-3B-Instruct (better quality)"
echo "New prompt format: Direct question-asking format"
echo "============================================================"
echo ""

python grpo_data_collection/run_grpo_collection.py \
    --num-generations 8 \
    --max-depth 3 \
    --branch-factor 2 \
    --seed 42

echo ""
echo "============================================================"
echo "Data collection complete!"
echo "Check grpo_data_collection/outputs/ for results"
echo "============================================================"
