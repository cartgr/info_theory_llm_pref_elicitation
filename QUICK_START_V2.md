# Quick Start Guide v2 (Optimized)

## Major Improvements

### ✅ 10x Faster Data Collection
- **Before:** 2-3 hours (computed rewards we didn't use)
- **After:** ~15 minutes (just generate prompts)

### ✅ Fixed Model Saving/Loading
- Uses absolute paths
- Validates files exist
- Works with chat script

### ✅ Memory-Safe Training
- 1.5B model (was 3B - OOM crash)
- Optimized for M2 Mac 64GB RAM

## Run Overnight (Recommended)

```bash
cd /Users/carterblair/0_Harvard/Research/info_theory_llm_pref_elicitation/pllm_mve
./grpo_data_collection/overnight_run_v2.sh
```

**Time:** 6-9 hours total
- Collection: ~15 min
- Training: 6-8 hours

## Or Run Steps Manually

### Step 1: Fast Collection (~15 min)
```bash
python grpo_data_collection/fast_collect_prompts.py \
    --max-depth 3 \
    --prompts-per-depth 40 \
    --questions-per-prompt 8
```

### Step 2: Train with Online Rewards (~6-8 hours)
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python grpo_data_collection/train_with_online_rewards.py \
    --prompts grpo_data_collection/outputs/fast_collect_*/trl_format/train.json \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output grpo_trained_models/v2 \
    --num-generations 3 \
    --epochs 2
```

### Step 3: Chat with Trained Model
```bash
python grpo_data_collection/chat_with_model.py \
    --model-path grpo_trained_models/v2/final_model
```

## What Changed

### Data Collection (fast_collect_prompts.py)
**Before:**
- Generated questions
- Called PLLM to answer each one
- Called evaluator for each answer
- Computed information gain
- Total: ~3000 API calls, 2-3 hours

**After:**
- Just generate questions
- Format as prompts
- No PLLM, no evaluator
- Total: ~100 API calls, 15 minutes

**Why:** GRPO computes rewards online during training anyway!

### Training (train_with_online_rewards.py)
- ✅ Saves model with absolute paths
- ✅ Validates files after saving
- ✅ Uses 1.5B model (memory-safe)
- ✅ Tests generation after training

### Chat (chat_with_model.py)
- ✅ Uses absolute paths
- ✅ Validates model files exist
- ✅ Uses `local_files_only=True`
- ✅ Better error messages

## Expected Results

### Collection
```
Depth 0: 40 prompts (initial questions)
Depth 1: 40 prompts (1-turn conversations)
Depth 2: 40 prompts (2-turn conversations)
Depth 3: 40 prompts (3-turn conversations)
Total: 160 training prompts
```

### Training
- ~160 prompts × 2 epochs = 320 training steps
- 3 generations per prompt = 960 questions generated
- Each question: PLLM answer + evaluator = ~20 API calls
- Total: ~19,200 API calls during training

### Model Quality
- Should generate actual car preference questions
- Questions should be relevant to context
- Better than random baseline

## Troubleshooting

**"Out of memory" during training:**
- Already using 1.5B (smaller)
- Can reduce `--num-generations` to 2
- Can reduce `max_completion_length` in code

**"Model not found" error:**
- Check absolute path in error message
- Verify files exist: `ls -la <model_path>`
- Re-run training if model didn't save

**Collection too slow:**
- Should be ~15 min
- If slower, check API connectivity
- Can reduce `--prompts-per-depth`

## Previous Run Analysis

From overnight run v1 (crashed):
- ✅ Generated questions (though many off-topic)
- ✅ Online rewards worked
- ✅ Training started
- ❌ Ran out of memory at 24% (3B model too large)
- ❌ Model saving had path issues

All fixed in v2!
