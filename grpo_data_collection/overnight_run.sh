#!/bin/bash

# Overnight GRPO data collection and training
# Estimated time: ~8 hours
# - Data collection: ~2-3 hours
# - Training: ~5-6 hours

set -e  # Exit on error

echo "============================================================"
echo "OVERNIGHT GRPO RUN"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Plan:"
echo "  1. Collect large dataset with better prompts"
echo "     - 16 generations per turn (more diversity)"
echo "     - Depth 4 (deeper conversations)"
echo "     - Branch factor 3 (more paths)"
echo "     - Expected: ~120-150 data points"
echo ""
echo "  2. Train with online rewards"
echo "     - Model: Qwen2.5-3B-Instruct (better quality)"
echo "     - 3 epochs (see learning progression)"
echo "     - 4 generations per prompt during training"
echo ""
echo "Estimated total time: 7-9 hours"
echo "============================================================"
echo ""

# Set environment for MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="grpo_data_collection/outputs/overnight_run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="${OUTPUT_DIR}/run.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "============================================================"
log "PHASE 1: DATA COLLECTION"
log "============================================================"

log "Collecting GRPO training data..."

python grpo_data_collection/run_grpo_collection.py \
    --num-generations 16 \
    --max-depth 4 \
    --branch-factor 3 \
    --seed 42 \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    log "ERROR: Data collection failed!"
    exit 1
fi

log "Data collection complete!"
log ""

# Find the most recent data collection output
LATEST_DATA=$(ls -td grpo_data_collection/outputs/run_* | head -1)
log "Using data from: $LATEST_DATA"

# Copy data to our output directory for reference
cp -r "$LATEST_DATA" "${OUTPUT_DIR}/collected_data"

log "============================================================"
log "PHASE 2: TRAINING WITH ONLINE REWARDS"
log "============================================================"

log "Starting GRPO training..."
log "Model: Qwen/Qwen2.5-3B-Instruct"
log "Epochs: 3"
log "Generations per prompt: 4"
log ""

python grpo_data_collection/train_with_online_rewards.py \
    --prompts "${LATEST_DATA}/trl_format/train.json" \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output "${OUTPUT_DIR}/trained_model" \
    --num-generations 4 \
    --epochs 3 \
    --batch-size 1 \
    --lr 5e-6 \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    log "ERROR: Training failed!"
    exit 1
fi

log "Training complete!"
log ""

log "============================================================"
log "PHASE 3: EVALUATION"
log "============================================================"

log "Testing trained model..."

# Test the model with a few prompts
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '${OUTPUT_DIR}/trained_model/final_model'
print(f'Loading model from {model_path}...')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to('mps')

test_prompts = [
    'You are interviewing someone about their cars preferences. Ask one specific question to learn what they\'re looking for.\n\nQuestion:',
    'You are interviewing someone about their cars preferences. Here\'s the conversation so far:\n\nQ: What\'s your budget?\nA: Around 25k\n\nAsk one follow-up question to learn more.\n\nQuestion:'
]

print('\n' + '='*60)
print('SAMPLE GENERATIONS FROM TRAINED MODEL')
print('='*60)

for i, prompt in enumerate(test_prompts, 1):
    print(f'\nTest {i}:')
    print(f'Prompt: {prompt[:100]}...')

    inputs = tokenizer(prompt, return_tensors='pt').to('mps')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    question = generated[len(prompt):].strip()
    print(f'Generated: {question}')

print('\n' + '='*60)
" 2>&1 | tee -a "$LOG_FILE"

log "============================================================"
log "RUN COMPLETE!"
log "============================================================"
log "End time: $(date)"
log ""
log "Results saved to: ${OUTPUT_DIR}"
log "  - collected_data/: GRPO training data"
log "  - trained_model/: Trained model weights"
log "  - run.log: Complete log of the run"
log ""
log "To analyze results:"
log "  python grpo_data_collection/analyze_dataset.py ${OUTPUT_DIR}/collected_data/dataset.json --plot"
log ""
log "To chat with trained model:"
log "  python grpo_data_collection/chat_with_model.py --model-path ${OUTPUT_DIR}/trained_model/final_model"
log ""
log "============================================================"

# Save a summary file
cat > "${OUTPUT_DIR}/README.md" << EOF
# Overnight GRPO Run Results

**Start time:** $(date)
**Configuration:**
- Data collection: 16 generations, depth 4, branch factor 3
- Training model: Qwen2.5-3B-Instruct
- Training: 3 epochs, 4 generations per prompt
- Learning rate: 5e-6

**Outputs:**
- \`collected_data/\`: GRPO training dataset
- \`trained_model/\`: Trained question-asking model
- \`run.log\`: Complete execution log

**Next steps:**
\`\`\`bash
# Analyze collected data
python grpo_data_collection/analyze_dataset.py ${OUTPUT_DIR}/collected_data/dataset.json --plot

# Chat with trained model
python grpo_data_collection/chat_with_model.py --model-path ${OUTPUT_DIR}/trained_model/final_model
\`\`\`
EOF

log "Summary written to ${OUTPUT_DIR}/README.md"
log "Sweet dreams! 🌙"
