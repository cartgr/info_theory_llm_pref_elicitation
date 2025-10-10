# Overnight GRPO Training Run

## Quick Start

```bash
cd /Users/carterblair/0_Harvard/Research/info_theory_llm_pref_elicitation/pllm_mve
./grpo_data_collection/overnight_run.sh
```

## What This Does

### Phase 1: Data Collection (~2-3 hours)
- Generates **~120-150 training examples**
- 16 candidate questions per turn (high diversity)
- Tree depth 4 (longer conversations)
- Branch factor 3 (explores multiple paths)
- Uses improved prompt format (direct question-asking)

### Phase 2: Training (~5-6 hours)
- Model: **Qwen2.5-3B-Instruct** (better quality than 0.5B)
- **3 epochs** (will see learning progression)
- 4 generations per prompt during training
- **Online reward computation** (real information-theoretic rewards)
- Learning rate: 5e-6 (conservative for stability)

### Phase 3: Evaluation (~5 minutes)
- Tests trained model on sample prompts
- Saves example generations

## Total Time: ~7-9 hours

## What You'll Get

**Tomorrow morning you'll have:**

1. **Large dataset** (`collected_data/`)
   - Diverse question-answer pairs
   - Tree-structured dialogues
   - Pre-computed rewards and advantages

2. **Trained model** (`trained_model/`)
   - Fine-tuned on question-asking task
   - Optimized with information-theoretic rewards
   - Ready to use for generating questions

3. **Complete logs** (`run.log`)
   - Every API call
   - Reward values
   - Training metrics

4. **Analysis-ready data** (`analysis.csv`, plots)

## After It Completes

### Check the results:
```bash
# Find the output directory
ls -lt grpo_data_collection/outputs/overnight_run_*

# View the log
tail -100 grpo_data_collection/outputs/overnight_run_*/run.log

# See sample generations
cat grpo_data_collection/outputs/overnight_run_*/README.md
```

### Analyze the data:
```bash
python grpo_data_collection/analyze_dataset.py \
    grpo_data_collection/outputs/overnight_run_*/collected_data/dataset.json \
    --plot
```

### Chat with your model:
```bash
python grpo_data_collection/chat_with_model.py \
    --model-path grpo_data_collection/outputs/overnight_run_*/trained_model/final_model
```

## Expected Results

### Data Collection
- **Total data points:** 120-150
- **Total questions generated:** ~2000
- **API calls:** ~4000-5000
- **Reward range:** -1.0 to +0.5 (information gain)

### Training
- **Training steps:** ~120-150 per epoch, 360-450 total
- **API calls during training:** ~15,000-20,000 (online rewards)
- **Expected improvement:** Rewards should increase over epochs
- **Loss:** Should decrease (model learning to maximize rewards)

### Quality Indicators

**Good signs:**
- ✅ Questions are about cars (not meta-instructions)
- ✅ Questions are diverse and specific
- ✅ Some rewards are positive (information gain)
- ✅ Later epochs have higher average rewards
- ✅ Generated questions make sense

**Bad signs (if you see these, let me know):**
- ❌ Questions still off-topic
- ❌ All rewards very negative
- ❌ Generated garbage text
- ❌ Training loss increases

## Monitoring During the Night

If you wake up and want to check progress:

```bash
# Check if still running
ps aux | grep overnight_run

# See latest log output
tail -f grpo_data_collection/outputs/overnight_run_*/run.log

# Check how many steps completed
grep "epoch" grpo_data_collection/outputs/overnight_run_*/run.log | tail -5
```

## If Something Goes Wrong

The script will stop and save logs if it hits an error. Common issues:

1. **API rate limits:** Script will show errors in log
2. **Memory issues:** M2 Mac should handle 3B model fine with 64GB
3. **MPS errors:** Already using `PYTORCH_ENABLE_MPS_FALLBACK=1`

## Sleep tight! 🌙

The script will run all night and have results ready in the morning.
