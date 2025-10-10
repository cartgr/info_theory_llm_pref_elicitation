#!/bin/bash

# Optimized Overnight GRPO Run v3
# - Reuses existing collected data (don't waste it!)
# - Optionally adds new prompts
# - Memory-efficient training

set -e

echo "============================================================"
echo "OPTIMIZED OVERNIGHT GRPO RUN v3"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Key improvement: REUSE EXISTING DATA!"
echo "  ✓ Extract ~120 prompts from previous collection"
echo "  ✓ Optionally add more for diversity"
echo "  ✓ Train on combined dataset"
echo ""
echo "Plan:"
echo "  1. Extract existing prompts (~1 min)"
echo "     - Reuse overnight_run_20251009_221738 data"
echo "     - ~120 quality prompts already collected"
echo ""
echo "  2. [Optional] Add new prompts (~15 min)"
echo "     - Can add more for diversity"
echo "     - Or skip to save time"
echo ""
echo "  3. Train with online rewards (~6-8 hours)"
echo "     - Model: Qwen2.5-1.5B-Instruct"
echo "     - 2 epochs, 3 generations/prompt"
echo ""
echo "Estimated total time: 6-9 hours"
echo "============================================================"
echo ""

export PYTORCH_ENABLE_MPS_FALLBACK=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="grpo_data_collection/outputs/overnight_v3_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}/run.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "============================================================"
log "PHASE 1: EXTRACT EXISTING PROMPTS"
log "============================================================"

log "Extracting prompts from all previous collections..."

python grpo_data_collection/extract_and_combine_prompts.py \
    --base-dir grpo_data_collection/outputs \
    --output "$OUTPUT_DIR/combined_prompts" \
    --deduplicate \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    log "ERROR: Prompt extraction failed!"
    exit 1
fi

log "Prompt extraction complete!"
log ""

# Find the combined prompts file (search in timestamped subdirectory)
PROMPTS_FILE=$(find "$OUTPUT_DIR" -name "train.json" -path "*/trl_format/train.json" | head -1)
log "Using prompts from: $PROMPTS_FILE"

# Validate prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    log "ERROR: Could not find train.json file"
    exit 1
fi

# Count prompts
PROMPT_COUNT=$(python -c "import json; print(len(json.load(open('$PROMPTS_FILE'))['prompt']))")
log "Total prompts ready for training: $PROMPT_COUNT"

# Optional: Add more prompts (uncomment to enable)
# log ""
# log "============================================================"
# log "PHASE 1b: ADD NEW PROMPTS (OPTIONAL)"
# log "============================================================"
# log "Adding 40 more prompts for diversity..."
#
# python grpo_data_collection/fast_collect_prompts.py \
#     --domain cars \
#     --max-depth 3 \
#     --prompts-per-depth 10 \
#     --questions-per-prompt 8 \
#     --output-dir "$OUTPUT_DIR/new_prompts" \
#     2>&1 | tee -a "$LOG_FILE"
#
# # Combine old and new (would need to implement merge logic)

log ""
log "============================================================"
log "PHASE 2: TRAINING WITH ONLINE REWARDS"
log "============================================================"

log "Starting GRPO training..."
log "Dataset: $PROMPT_COUNT prompts (reused from previous collections)"
log "Model: Qwen/Qwen2.5-1.5B-Instruct"
log "Epochs: 2"
log "Generations per prompt: 3"
log ""

python grpo_data_collection/train_with_online_rewards.py \
    --prompts "$PROMPTS_FILE" \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output "$OUTPUT_DIR/trained_model" \
    --num-generations 3 \
    --epochs 2 \
    --batch-size 1 \
    --lr 5e-6 \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    log "ERROR: Training failed (check log for details)"
    log "Partial results may be in: $OUTPUT_DIR/trained_model"
    exit 1
fi

log "Training complete!"
log ""

log "============================================================"
log "RUN COMPLETE!"
log "============================================================"
log "End time: $(date)"
log ""
log "Results:"
log "  Prompts extracted: $OUTPUT_DIR/combined_prompts/"
log "  Trained model: $OUTPUT_DIR/trained_model/final_model"
log "  Full log: $LOG_FILE"
log ""
log "Statistics:"
log "  Total prompts: $PROMPT_COUNT (reused!)"
log "  Training steps: $(($PROMPT_COUNT * 2))"
log "  Estimated API calls: $(($PROMPT_COUNT * 3 * 20 * 2))"
log ""
log "To chat with model:"
log "  python grpo_data_collection/chat_with_model.py \\"
log "    --model-path $OUTPUT_DIR/trained_model/final_model"
log ""
log "============================================================"

# Create summary
cat > "$OUTPUT_DIR/README.md" << EOF
# Overnight Run v3 - Data Reuse

**Completed:** $(date)

## Key Insight: Reused Existing Data!
Instead of re-collecting, we extracted $PROMPT_COUNT prompts from previous runs.

## Configuration
- Prompts: $PROMPT_COUNT (extracted from previous collections)
- Training: Qwen2.5-1.5B-Instruct, 2 epochs, 3 gens/prompt
- Learning rate: 5e-6

## Data Sources
See \`combined_prompts/metadata.json\` for list of source datasets.

## Time Saved
- Previous: 2-3 hours collection + 6-8 hours training = 8-11 hours
- Now: 1 min extraction + 6-8 hours training = 6-8 hours
- **Saved: 2-3 hours!**

## Results
- \`combined_prompts/\`: Extracted and combined prompts
- \`trained_model/final_model/\`: Trained model
- \`run.log\`: Complete log

## Usage
\`\`\`bash
# Chat with trained model
python grpo_data_collection/chat_with_model.py \\
  --model-path $OUTPUT_DIR/trained_model/final_model
\`\`\`

## Growing the Dataset
To add more prompts later:
\`\`\`bash
# Collect new prompts
python grpo_data_collection/fast_collect_prompts.py \\
  --prompts-per-depth 20

# Re-extract to combine old + new
python grpo_data_collection/extract_and_combine_prompts.py

# Retrain on larger dataset
\`\`\`
EOF

log "Summary saved to $OUTPUT_DIR/README.md"
log "Done! 🎉"
