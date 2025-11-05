# Best-of-N Question Selection for Preference Elicitation

## Quick Start

### Prerequisites
```bash
conda activate llm-finetune
cd pllm_mve
```

### Run Experiment
```bash
# Single seed (fast, ~5-10 minutes)
python run_bestofn_experiment.py \
    --config ../experiments/mve_one_persona.yml \
    --num-candidates 5 \
    --num-samples 3 \
    --debug

# Multiple seeds for statistical significance (~1-2 hours)
python run_bestofn_experiment.py \
    --config ../experiments/mve_one_persona.yml \
    --num-candidates 5 \
    --num-samples 3 \
    --num-seeds 10
```

### Results
Results are saved to `experiments/outputs/runs/bestofn_mve_one_persona_<timestamp>/`

---

## Overview

### What is Preference Elicitation?

Preference elicitation is the problem of learning someone's preferences through strategic questioning. The goal is to ask questions that **maximize information gain** about their underlying preference structure.

### The Challenge

Given a set of items (e.g., cars), we want to predict pairwise preferences (e.g., "Does the user prefer car A over car B?"). We can ask questions and get answers, but questions have different informativeness. How do we choose which questions to ask?

### The Best-of-N Solution

**Key Idea:** Instead of generating a single question, generate **k candidate questions**, evaluate each by sampling **t hypothetical responses**, and select the question with the **highest expected information gain**.

This improves over direct generation by:
1. **Exploration**: Generates multiple diverse candidates
2. **Evaluation**: Simulates multiple possible responses per candidate
3. **Selection**: Picks the question that performs best in expectation

---

## The Best-of-N Method

### Algorithm

For each turn (typically 3 turns per episode):

```
1. Generate k=5 candidate questions using LLM
   ├─ "What feature would you prioritize?"
   ├─ "How important is fuel efficiency?"
   ├─ "What's your typical budget?"
   └─ ... (5 total)

2. For each candidate question:
   ├─ Sample t=3 hypothetical answers from PLLM
   ├─ For each answer:
   │  ├─ Compute evaluator score on fixed eval set
   │  └─ Calculate information gain vs baseline
   └─ Average the t gains → expected gain

3. Select question with highest expected gain

4. Actually ask the PLLM the selected question

5. Update beliefs based on actual answer
```

### Visual Example

```
Turn 0: Initial beliefs (no info)
  Evaluator accuracy: 50% (random)
  Score: -1.25

Generate 5 candidates:
  Q1: "What's most important?" → [gain: -1.29]  ❌ (high variance)
  Q2: "What feature to prioritize?" → [gain: +0.52] ✅ (best!)
  Q3: "How important is handling?" → [gain: -0.05]
  Q4: "Infotainment or simple?" → [gain: +0.26]
  Q5: "Relaxed or sporty?" → [gain: -0.81]

Selected: Q2
Actual gain: +0.54 (close to expected!)

Turn 1: Updated beliefs
  Evaluator accuracy: 50% (still learning)
  Score: -0.71 (improved!)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` (num_candidates) | 5 | Number of candidate questions to generate |
| `t` (num_samples) | 3 | Number of hypothetical responses per candidate |
| Aggregation | mean | How to combine t samples (mean = expected value) |
| Turns | 3 | Number of questions to ask total |

### Why This Works

**Variance Reduction**: Averaging over t samples gives a more accurate estimate of expected gain than a single sample.

**Better Exploration**: Generating k candidates explores the space of possible questions.

**Explicit Evaluation**: Unlike direct generation (which relies on the LLM's implicit notion of "informative"), we explicitly measure information gain.

---

## Direct Baseline

For comparison, we also run a **direct baseline** that:
1. Prompts the LLM: "Generate 1 question to ask next"
2. No candidate generation
3. No evaluation or selection
4. Directly asks the PLLM

### Fair Comparison

Both methods use:
- **Same prompts**: "You are an expert interviewer... Generate informative questions..."
- **Same temperature**: 0.8 (for diversity)
- **Same model**: Llama-3.3-70B-Instruct-Turbo
- **Same evaluation**: Log score on fixed 20 pairs

**The ONLY difference**: Best-of-N generates and evaluates multiple candidates.

---

## Experimental Setup

### Configuration File

`experiments/mve_one_persona.yml`:
```yaml
episode:
  persona: |
    You are a 28-year-old software engineer in Austin, Texas.
    Budget: Strict maximum of $35,000
    Must-haves:
    - Fun to drive with responsive handling
    - Manual transmission STRONGLY preferred
    - Good reliability record (10+ years)
    - Decent fuel economy (25+ mpg)
    Dealbreakers:
    - Luxury brands (expensive maintenance)
    - SUVs or trucks
    - Poor reliability ratings
  domain: cars
  num_items: 10
  num_pairs: 20
  num_turns: 3
  model: "meta-llama/Llama-3.3-70B-Instruct-Turbo"
```

### Key Components

**PLLM (Participant LLM)**: Simulates a user with the specified persona. Provides:
- Pairwise preference labels (20 pairs, fixed at episode start)
- Answers to questions during dialogue

**QLLM (Question LLM)**: Generates candidate questions based on:
- Conversation history
- List of items being considered
- Domain context

**Evaluator D**: Predicts P(A > B | transcript) for each pair using:
- Only the answers (NOT the questions!)
- Logprobs from LLM for calibrated probabilities
- Returns prob in [0.001, 0.999] (clipped to avoid log(0))

### Scoring Rule

We use the **logarithmic scoring rule**:

```
F(S) = (1/|E|) * Σ log p(y_ij | S)

where:
- E = evaluation set (20 pairs)
- y_ij = true label (1 if A>B, 0 if B>A)
- p(y_ij | S) = evaluator's probability given transcript S

Reward: r = F(S_new) - F(S_old)
```

**Properties:**
- **Strictly proper**: Maximized when predictions = true probabilities
- **Information-theoretic**: Directly measures information content
- **Penalizes overconfidence**: log(0.01) is much worse than log(0.49)

---

## Output & Interpretation

### File Structure

```
experiments/outputs/runs/bestofn_mve_one_persona_<timestamp>/
├── config.yml                    # Copy of configuration
├── seed_0/
│   ├── episode_data.json         # Items, pairs, labels
│   ├── episode_logs_bestofn.json # Best-of-N trajectory
│   ├── episode_logs_direct.json  # Direct baseline trajectory
│   └── comparison.json           # Head-to-head results
├── seed_1/
│   └── ...
└── aggregate_results.json        # Statistics across seeds
```

### Reading Results

**episode_logs_bestofn.json**:
```json
[
  {
    "turn": 0,
    "question": "What feature would you prioritize?",
    "answer": "Track mode for handling and acceleration",
    "score": -0.7090,
    "reward": 0.5403,
    "expected_gain": 0.5238,
    "policy": "bestofn"
  },
  ...
]
```

**comparison.json**:
```json
{
  "bestofn": {
    "total_reward": 0.5973,
    "final_score": -0.6520
  },
  "direct": {
    "total_reward": 0.1805,
    "final_score": -1.0683
  },
  "improvement": {
    "total_reward": 0.4168  // Best-of-N is 2.3x better!
  }
}
```

### Metrics Explained

**Accuracy @ 0.5**: Binary classification accuracy (threshold predictions at 0.5)
- Can be misleading! A model can have 70% accuracy but terrible log score if overconfident.

**Log Score**: Average log-probability of correct predictions
- **Lower is worse** (more negative)
- Range: -∞ to 0 (0 = perfect, -∞ = completely wrong)
- Example: -0.65 is better than -1.07

**Reward**: Change in log score after a turn
- **Positive**: Question improved predictions ✅
- **Negative**: Question hurt predictions ❌
- Can be negative even when accuracy improves (due to overconfidence)

**Expected Gain vs Actual Reward**:
- Expected gain = prediction before asking
- Actual reward = result after asking
- Close values = good estimation
- Large difference = high variance in PLLM responses

### Understanding Overconfidence

**Example from output:**
```
Turn 0: Accuracy 50%, Score -0.71  ✅ Getting direction right
Turn 1: Accuracy 70%, Score -1.42  ❌ Overconfident on wrong predictions!
```

What happened:
- Model predicted 0.999 for several pairs
- Got some wrong → log(0.001) = -6.91 penalty EACH
- Log score tanks even though binary accuracy improved

**This is good behavior!** Log score correctly penalizes unjustified confidence.

---

## Implementation Details

### Key Files

```
pllm_mve/
├── run_bestofn_experiment.py          # Main experiment runner
└── src/pllm_mve/
    ├── qllm_policy.py                 # Question generation & selection
    │   ├── generate_candidate_questions()  # LLM generates k questions
    │   ├── select_bestofn_question()       # Evaluate & pick best
    │   └── select_direct_question()        # Baseline (no selection)
    ├── together_client.py             # Together API wrapper
    │   ├── get_ab_preference()            # PLLM labels pairs
    │   ├── get_eval_probability_logprobs() # Evaluator predictions
    │   └── answer_question()              # PLLM answers
    ├── evaluator.py                   # Evaluator D
    ├── pllm.py                        # Participant LLM
    ├── scoring.py                     # Log score computation
    └── types.py                       # Data structures
```

### Logprobs for Evaluation

Instead of asking the LLM "What's the probability?", we use logprobs:

```python
# Prompt: "Which option does the participant prefer? Answer 'A' or 'B'."
response = model.generate(..., logprobs=5)

# Extract P(A) and P(B) from top 5 tokens
top_logprobs = {'A': -0.5, 'B': -1.2, 'Since': -3.1, ...}

# Calculate P(A > B)
p_a = exp(-0.5) = 0.607
p_b = exp(-1.2) = 0.301
P(A > B) = p_a / (p_a + p_b) = 0.668
```

**Benefits:**
- More accurate than JSON generation
- Properly normalized probabilities
- Captures model uncertainty

**Handling missing tokens**: If 'B' isn't in top 5, assume P(A>B) ≈ 1.0 (model is very confident).

### PLLM Simulation

The PLLM maintains **consistency** via:
1. **Fixed persona**: Same persona text for all comparisons
2. **Cached labels**: Pairwise preferences computed once at episode start
3. **Temperature 0.7**: Slightly stochastic answers for realism

---

## Customization

### Change the Persona

Edit `experiments/mve_one_persona.yml`:
```yaml
persona: |
  You are a 35-year-old parent with two kids.

  Budget: $40,000 max

  Must-haves:
  - High safety ratings (5-star crash test)
  - Third row seating or very spacious back seats
  - Good reliability (want to keep 8+ years)
  - Modern safety features (lane assist, blind spot)

  Dealbreakers:
  - Poor safety ratings
  - Cramped interior
  - Sports cars (not practical)
```

**More detailed personas = better learning!**

### Adjust k and t

```bash
# More candidates, fewer samples (fast exploration)
python run_bestofn_experiment.py --num-candidates 10 --num-samples 1

# Fewer candidates, more samples (accurate estimation)
python run_bestofn_experiment.py --num-candidates 3 --num-samples 5

# Default (balanced)
python run_bestofn_experiment.py --num-candidates 5 --num-samples 3
```

**Trade-off**: k×t = total API calls per turn
- Higher k×t = more accurate but slower & more expensive
- Recommended: k×t ≤ 15 for reasonable runtime

### Run Multiple Seeds

```bash
# Statistical significance with 10 seeds
python run_bestofn_experiment.py --num-seeds 10
```

Outputs `aggregate_results.json` with:
- Mean reward across seeds
- Standard deviation
- Success rate (% of seeds where best-of-N wins)

### Change Domains

Modify `eval_items.py` to add new domains:
```python
def generate_items_for_domain(domain, persona, num_items):
    if domain == "restaurants":
        # Generate restaurant options
    elif domain == "movies":
        # Generate movie options
    elif domain == "vacation":
        # Your new domain!
```

Then update config: `domain: vacation`

---

## Troubleshooting

### "Could not find logprobs for both A and B"

**Cause**: Model is very confident, 'B' not in top 5 tokens.

**Solution**: Already handled! Code assigns P(A>B) = 0.999 when this occurs.

**To verify**: Check debug output for `INFO: 'B' not in top 5`

### Empty Response / JSON Parse Errors

**Cause**: Model doesn't support JSON mode properly (e.g., `gpt-oss-20b`).

**Solution**: Use Llama models which have good JSON mode support:
- `meta-llama/Llama-3.2-3B-Instruct-Turbo` (fast, less accurate)
- `meta-llama/Llama-3.3-70B-Instruct-Turbo` (slower, more accurate)

### API Rate Limits

**Symptoms**: Lots of "API overloaded" messages, retries.

**Solutions**:
1. Reduce seeds: `--num-seeds 3` instead of 10
2. Reduce k×t: `--num-candidates 3 --num-samples 2`
3. Add delays in code (not implemented yet)

### Negative Rewards

**Is this bad?**: Not necessarily! Negative rewards can occur when:
1. PLLM gives unexpected answer
2. Evaluator becomes overconfident (correctly penalized by log score)
3. Question was genuinely uninformative

**Check**:
- Is accuracy improving? If yes, might be calibration issue
- Look at predictions: Are they extreme (0.001, 0.999)? Overconfidence!
- Compare to expected gain: Large difference = high variance

### High Variance in Expected Gains

**Example**: `Gains: [0.534, -2.205, -2.205] → Mean: -1.292`

**Cause**: With t=3 samples, one lucky answer gives +0.53, two others catastrophically fail.

**Solutions**:
1. Increase t: `--num-samples 5` (more stable estimates)
2. Use different aggregation (not implemented): median instead of mean
3. Accept it: Real-world variance! Selection still helps.

---

## Future Work

### Human-in-the-Loop

Current: PLLM simulates user
Future: `run_human_experiment.py`
- Human provides pairwise labels
- System asks questions interactively
- Real preference elicitation!

### Alternative Scoring Rules

Current: Log score
Future options:
- Brier score (quadratic, less sensitive to overconfidence)
- Ranked accuracy (ignore probability calibration)
- Custom domain-specific metrics

### Better Baselines

Current: Direct generation
Future:
- **Greedy single-sample**: Original MVE (k candidates, t=1 sample each)
- **Random questions**: No LLM generation
- **Template questions**: Fixed question bank
- **Human-written**: Expert-designed questions

### Adaptive Sampling

Current: Fixed t=3 for all candidates
Future: Adaptive t
- Sample more for promising candidates
- Early stopping for clearly bad candidates
- Thompson sampling for exploration

---

## Citation

If you use this code, please cite:

```bibtex
@misc{pllm_bestofn_2024,
  title={Best-of-N Question Selection for LLM-Based Preference Elicitation},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]}
}
```

---

## Contact

Questions or issues?
- Check Troubleshooting section above
- Review logs in `experiments/outputs/runs/`
- File an issue with debug output included
