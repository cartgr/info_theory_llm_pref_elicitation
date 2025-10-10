#!/bin/bash

# Optimized Overnight GRPO Run
# - Fast data collection (no reward computation)
# - Memory-efficient training

set -e

echo "============================================================"
echo "OPTIMIZED OVERNIGHT GRPO RUN v2"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Improvements:"
echo "  ✓ Fast collection: 15 min (was 2-3 hours)"
echo "  ✓ No wasted reward computation during collection"
echo "  ✓ Fixed model saving/loading"
echo "  ✓ All rewards computed online during training"
echo ""
echo "Plan:"
echo "  1. Fast prompt collection (~15 min)"
echo "     - 40 prompts per depth (0-3)"
echo "     - 8 questions per prompt"
echo "     - Total: ~160 training prompts"
echo ""
echo "  2. Train with online rewards (~6-8 hours)"
echo "     - Model: Qwen2.5-1.5B-Instruct (memory-safe)"
echo "     - 2 epochs"
echo "     - 3 generations per prompt"
echo ""
echo "Estimated total time: 6-9 hours"
echo "============================================================"
echo ""

export PYTORCH_ENABLE_MPS_FALLBACK=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="grpo_data_collection/outputs/overnight_v2_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}/run.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "============================================================"
log "PHASE 1: FAST PROMPT COLLECTION"
log "============================================================"

log "Collecting prompts (NO reward computation)..."

python grpo_data_collection/fast_collect_prompts.py \
    --domain cars \
    --max-depth 3 \
    --prompts-per-depth 40 \
    --questions-per-prompt 8 \
    --output-dir "$OUTPUT_DIR/collected" \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    log "ERROR: Prompt collection failed!"
    exit 1
fi

log "Prompt collection complete!"
log ""

# Find the collected data
PROMPTS_FILE=$(find "$OUTPUT_DIR/collected" -name "train.json" | head -1)
log "Using prompts from: $PROMPTS_FILE"

log "============================================================"
log "PHASE 2: TRAINING WITH ONLINE REWARDS"
log "============================================================"

log "Starting GRPO training..."
log "Model: Qwen/Qwen2.5-1.5B-Instruct (memory-safe)"
log "Epochs: 2"
log "Generations: 3"
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
    log "ERROR: Training failed (see log for details)"
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
log "  Collected prompts: $OUTPUT_DIR/collected/"
log "  Trained model: $OUTPUT_DIR/trained_model/final_model"
log "  Full log: $LOG_FILE"
log ""
log "To chat with model:"
log "  python grpo_data_collection/chat_with_model.py \\"
log "    --model-path $OUTPUT_DIR/trained_model/final_model"
log ""
log "============================================================"

# Create summary
cat > "$OUTPUT_DIR/README.md" << EOF
# Optimized Overnight Run v2

**Completed:** $(date)

## Configuration
- Fast collection: 40 prompts/depth × 4 depths = 160 prompts
- Training: Qwen2.5-1.5B-Instruct, 2 epochs, 3 gens/prompt
- Learning rate: 5e-6

## Improvements over v1
- ✓ 10x faster collection (15 min vs 2-3 hours)
- ✓ No wasted reward computation
- ✓ Memory-safe model size
- ✓ Fixed model saving/loading

## Results
- \`collected/\`: Training prompts
- \`trained_model/final_model/\`: Trained model
- \`run.log\`: Complete log

## Usage
\`\`\`bash
# Chat with trained model
python grpo_data_collection/chat_with_model.py \\
  --model-path $OUTPUT_DIR/trained_model/final_model
\`\`\`
EOF

log "Summary saved to $OUTPUT_DIR/README.md"
log "Done! 🎉"
