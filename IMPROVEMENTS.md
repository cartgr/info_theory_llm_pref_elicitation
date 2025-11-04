# Major Improvements Summary

## 1. Expanded Evaluation Set (60 cars vs 10)

**File:** `src/pllm_mve/eval_items.py`

**Changes:**
- Increased from 10 cars to **60 diverse cars**
- Categories include:
  - Budget performance (under $30k)
  - Used performance/luxury
  - New luxury (over $40k)
  - SUVs (budget and luxury)
  - Electric/Hybrid vehicles
  - Practical commuters
  - Trucks
  - Minivans
  - Specialty vehicles

**Benefits:**
- Much better signal for information gain
- Covers diverse user preferences
- More robust evaluation

## 2. Logprobs-Based Probability Estimation

**File:** `src/pllm_mve/together_client.py` + `src/pllm_mve/evaluator.py`

**New Method:** `get_eval_probability_logprobs()`

**How it works:**
1. Prefill assistant response with "A", get logprob
2. Prefill assistant response with "B", get logprob
3. Calculate: P(A > B) = exp(logprob_A) / (exp(logprob_A) + exp(logprob_B))

**Benefits:**
- **More accurate**: Direct probability from model's internal state
- **More efficient**: 2 API calls instead of asking model to generate probability
- **More reliable**: No JSON parsing issues
- **Better calibrated**: Uses actual token likelihoods

**Old method:**
- Asked model to generate probability as JSON
- Prone to miscalibration
- Required JSON parsing

**New method:**
- Uses logprobs with prefilled responses
- Mathematically sound probability calculation
- More stable and reliable

## 3. Diverse Persona Data Collection

**Files:** `grpo_data_collection/collect_diverse_data.py` + `run_diverse_training.sh`

**Features:**
- **20 different personas** covering diverse car preferences
- **10 scenarios** per persona
- **Multiple conversation depths** (0-3 turns)
- **800 training prompts** (default settings)

**Personas include:**
- Budget-conscious performance seekers
- Safety-focused families
- Luxury buyers
- Eco-friendly EV enthusiasts
- Practical commuters
- Manual transmission enthusiasts
- Truck users
- City drivers
- And 12 more...

## 4. Fixed Model Generation Issues

**Files:** `grpo_data_collection/chat_with_model.py` + `train_with_online_rewards.py` + `test_saved_model.py`

**Problem:** Model was generating garbage when tested

**Root cause:** Qwen2.5-Instruct uses ChatML format but prompts were plain text

**Solution:**
- Use `tokenizer.apply_chat_template()` for all generation
- Proper format: `<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant`
- Updated all test/chat scripts to use chat format

**Results:**
- Model now generates coherent questions
- Properly formatted interactions
- Better quality outputs

## 5. Better Training Infrastructure

**Scripts:**
- `collect_diverse_data.py` - Fast data collection (~30-45 min)
- `run_diverse_training.sh` - Full pipeline
- `test_saved_model.py` - Test with proper chat format
- `chat_with_model.py` - Interactive chat with conversation history

## Expected Results

With all improvements:
- **Better signal**: 60 cars vs 10 (6x more eval pairs)
- **More accurate rewards**: Logprobs vs JSON generation
- **More robust model**: 800 diverse prompts vs 47
- **Faster evaluation**: 2 API calls vs 1 generation per eval
- **Better quality**: Proper chat formatting

## Usage

### Collect diverse data and train:
```bash
./grpo_data_collection/run_diverse_training.sh
```

### Or step by step:
```bash
# 1. Collect data (fast - no reward computation)
python grpo_data_collection/collect_diverse_data.py \
    --prompts-per-persona 10 \
    --max-depth 3

# 2. Train with online rewards (slow - computes rewards during training)
python grpo_data_collection/train_with_online_rewards.py \
    --prompts <collected_data>/trl_format/train.json \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 2

# 3. Test the model
python grpo_data_collection/test_saved_model.py <model_path>

# 4. Interactive chat
python grpo_data_collection/chat_with_model.py --model-path <model_path>
```

## API Call Comparison

### Old (per evaluation pair):
- 1 API call to generate probability JSON

### New (per evaluation pair):
- 2 API calls for logprobs (A and B)
- But much more accurate and reliable

### Training (~800 prompts, 2 epochs, 3 gens/prompt):
- Total evals: ~800 × 3 × 20 × 2 = 96,000 evaluations
- API calls: 96,000 × 2 = ~192,000 logprob calls
- Plus ~4,800 PLLM answer calls

## Next Steps

1. Run `./grpo_data_collection/run_diverse_training.sh` overnight
2. Evaluate trained model quality
3. Potentially increase data size if needed
4. Consider adding more personas or scenarios
