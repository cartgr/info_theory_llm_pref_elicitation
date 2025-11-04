#!/bin/bash

# Comprehensive GRPO Training with Diverse Data
# Collects data from 20 personas and trains on it

set -e

export PYTORCH_ENABLE_MPS_FALLBACK=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="grpo_data_collection/outputs/diverse_run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}/run.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "============================================================"
log "COMPREHENSIVE GRPO TRAINING WITH DIVERSE DATA"
log "============================================================"
log "Start time: $(date)"
log ""

# Configuration
PROMPTS_PER_PERSONA=10  # 10 prompts per persona at each depth
MAX_DEPTH=3             # Depths 0, 1, 2, 3
QUESTIONS_PER_PROMPT=8  # 8 candidate questions per prompt
NUM_PERSONAS=20         # 20 different personas

TOTAL_PROMPTS=$((NUM_PERSONAS * PROMPTS_PER_PERSONA * (MAX_DEPTH + 1)))

log "Configuration:"
log "  Personas: $NUM_PERSONAS"
log "  Prompts per persona per depth: $PROMPTS_PER_PERSONA"
log "  Depths: 0-$MAX_DEPTH ($(($MAX_DEPTH + 1)) levels)"
log "  Questions per prompt: $QUESTIONS_PER_PROMPT"
log "  Total training prompts: $TOTAL_PROMPTS"
log ""
log "Expected time:"
log "  Data collection: ~30-45 minutes"
log "  Training (2 epochs): ~8-12 hours"
log "  Total: ~9-13 hours"
log ""

log "============================================================"
log "PHASE 1: DIVERSE DATA COLLECTION"
log "============================================================"
log ""

python grpo_data_collection/collect_diverse_data.py \
    --domain cars \
    --prompts-per-persona $PROMPTS_PER_PERSONA \
    --max-depth $MAX_DEPTH \
    --questions-per-prompt $QUESTIONS_PER_PROMPT \
    --output-dir "$OUTPUT_DIR/collected" \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    log "ERROR: Data collection failed!"
    exit 1
fi

# Find the collected data
PROMPTS_FILE=$(find "$OUTPUT_DIR/collected" -name "train.json" | head -1)
log ""
log "Using training data: $PROMPTS_FILE"

# Count prompts
ACTUAL_PROMPTS=$(python -c "import json; print(len(json.load(open('$PROMPTS_FILE'))['prompt']))")
log "Collected prompts: $ACTUAL_PROMPTS"
log ""

log "============================================================"
log "PHASE 2: GRPO TRAINING WITH ONLINE REWARDS"
log "============================================================"
log ""
log "Training configuration:"
log "  Model: Qwen/Qwen2.5-1.5B-Instruct"
log "  Epochs: 2"
log "  Generations per prompt: 3"
log "  Learning rate: 5e-6"
log "  Batch size: 1"
log ""
log "Starting training (this will take 8-12 hours)..."
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

log ""
log "============================================================"
log "RUN COMPLETE!"
log "============================================================"
log "End time: $(date)"
log ""
log "Results:"
log "  Collected data: $OUTPUT_DIR/collected/"
log "  Trained model: $OUTPUT_DIR/trained_model/final_model"
log "  Full log: $LOG_FILE"
log ""
log "Statistics:"
log "  Total prompts: $ACTUAL_PROMPTS"
log "  Personas covered: $NUM_PERSONAS"
log "  Training steps: $((ACTUAL_PROMPTS * 2))"
log "  Estimated API calls: $((ACTUAL_PROMPTS * 3 * 20 * 2))"
log ""
log "To chat with trained model:"
log "  python grpo_data_collection/chat_with_model.py \\"
log "    --model-path $OUTPUT_DIR/trained_model/final_model"
log ""
log "To test model:"
log "  python grpo_data_collection/test_saved_model.py \\"
log "    --model-path $OUTPUT_DIR/trained_model/final_model"
log ""
log "============================================================"

# Create summary README
cat > "$OUTPUT_DIR/README.md" << EOF
# Diverse Persona GRPO Training Run

**Completed:** $(date)

## Dataset

This run used diverse training data covering multiple personas and scenarios:

- **Personas:** $NUM_PERSONAS different user profiles
- **Scenarios:** Various conversation goals and contexts
- **Prompts:** $ACTUAL_PROMPTS total training prompts
- **Depth range:** 0-$MAX_DEPTH (initial questions + multi-turn conversations)

### Personas Include:
- Budget-conscious performance seekers
- Safety-focused families
- Eco-friendly buyers
- Luxury vehicle shoppers
- Practical commuters
- Automotive enthusiasts
- Work truck needs
- City drivers
- And more...

## Training Configuration

- **Model:** Qwen/Qwen2.5-1.5B-Instruct
- **Training:** 2 epochs, 3 generations/prompt
- **Learning rate:** 5e-6
- **Reward function:** Online information-theoretic rewards

## Expected Improvements

With this diverse dataset, the model should:
- Ask relevant questions across different user types
- Handle various conversation contexts
- Generate appropriate follow-ups based on user answers
- Balance between broad discovery and specific details

## Results

- \`collected/\`: Training prompts with metadata
- \`trained_model/final_model/\`: Trained model
- \`run.log\`: Complete training log

## Usage

### Interactive Chat
\`\`\`bash
python grpo_data_collection/chat_with_model.py \\
  --model-path $OUTPUT_DIR/trained_model/final_model
\`\`\`

### Test Generation
\`\`\`bash
python grpo_data_collection/test_saved_model.py \\
  $OUTPUT_DIR/trained_model/final_model
\`\`\`

## Data Statistics

See \`collected/*/collection_stats.json\` for detailed statistics.
EOF

log "Summary saved to $OUTPUT_DIR/README.md"
log "Done! 🎉"
