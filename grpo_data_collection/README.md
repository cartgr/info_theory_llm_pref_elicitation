# GRPO Data Collection for Question-Asking Policy

This directory contains the implementation for collecting training data to train a question-asking policy using Group Relative Policy Optimization (GRPO).

## Quick Start

```bash
# Run data collection with default settings
python run_grpo_collection.py

# Run with custom parameters
python run_grpo_collection.py --num-generations 16 --max-depth 3 --branch-factor 3
```

## What This Does

1. **Generates Multiple Questions**: At each turn, generates N candidate questions (default: 8)
2. **Evaluates All Candidates**: Gets PLLM answers and computes rewards for ALL questions
3. **Calculates Group Advantages**: Computes relative advantages for GRPO training
4. **Builds Tree Structure**: Explores multiple dialogue paths (max depth: 3)

## Key Files

- `run_grpo_collection.py` - Main entry point for data collection
- `grpo_collector.py` - Core GRPO collection logic
- `question_generator.py` - Generates diverse candidate questions
- `export_grpo_dataset.py` - Exports data in various formats
- `train_with_trl.py` - Example script for training with HuggingFace TRL
- `configs/grpo_config.yml` - Configuration file

## Configuration

Edit `configs/grpo_config.yml` to customize:

```yaml
grpo_collection:
  num_generations: 8       # Questions per turn
  max_depth: 3            # Tree depth
  branch_factor: 2        # How many top questions to branch from
  beginning_prompt: "Discover the user's preferences about cars"
  domain: "cars"
  persona: "likes fast cars but is on a budget"
```

## Output Format

After running, you'll find data in `outputs/run_YYYYMMDD_HHMMSS/`:

```
outputs/run_20240109_143022/
├── config.yml           # Configuration used
├── dataset.json        # Complete dataset
├── dataset.jsonl       # JSONL format (one example per line)
├── trl_format/         # Ready for TRL GRPOTrainer
│   ├── train.json
│   ├── prompts.csv
│   ├── completions.npy
│   └── rewards.npy
├── splits/             # Train/test splits
│   ├── train.jsonl
│   └── test.jsonl
└── analysis.csv        # For analysis and visualization
```

## Data Structure

Each data point contains:
```python
{
  "prompt": "Context: Q: What's your budget? A: Around $25k\nTask: Ask follow-up",
  "completions": [
    "Do you prefer new or used cars?",
    "What's most important: performance or reliability?",
    ...  # 8 total questions
  ],
  "rewards": [0.23, 0.45, ...],     # Information gain for each
  "advantages": [-0.5, 0.8, ...],   # Group-relative advantages
  "metadata": {
    "persona": "likes fast cars but is on a budget",
    "depth": 1,
    "answers": [...]  # PLLM answers for each question
  }
}
```

## How Rewards Are Computed

1. **Information-Theoretic Score**:
   ```
   F(s) = (1/|E|) Σ log p(y_ij | s)
   ```
   Measures how well transcript `s` predicts preferences

2. **Reward = Information Gain**:
   ```
   r = F(s_new) - F(s_old)
   ```
   How much a question improves predictions

3. **Advantages = Group Normalization**:
   ```
   A = (r - mean(r)) / std(r)
   ```
   Relative quality within the group

## Training with Collected Data

After collection, train your question-asking policy:

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# Load collected data
with open("outputs/latest/trl_format/train.json") as f:
    data = json.load(f)

dataset = Dataset.from_dict(data)

# Configure GRPO training
config = GRPOConfig(
    num_generations=8,  # Match collection
    learning_rate=1e-5,
    num_train_epochs=3
)

# Train
trainer = GRPOTrainer(
    model="your-base-model",
    train_dataset=dataset,
    config=config
)
trainer.train()
```

See `train_with_trl.py` for a complete example.

## Tree Structure Visualization

The collection builds a tree of dialogues:
```
Root
├── Q: What's your budget?
│   ├── Q: New or used preference?
│   │   ├── Q: Specific brands in mind?
│   │   └── Q: Daily commute distance?
│   └── Q: Performance vs reliability?
│       ├── Q: Manual or automatic?
│       └── Q: Fuel efficiency important?
└── Q: What type of driving?
    └── ...
```

Only top-K questions (by advantage) are expanded at each level.

## Customization

### Different Personas
```bash
python run_grpo_collection.py --persona "values luxury and comfort"
```

### More Questions Per Turn
```bash
python run_grpo_collection.py --num-generations 16
```

### Deeper Trees
```bash
python run_grpo_collection.py --max-depth 5 --branch-factor 3
```

## Dependencies

- `together` - Together AI API client
- `pydantic` - Data validation
- `numpy` - Numerical operations
- `pandas` - Data export
- `pyyaml` - Configuration
- HuggingFace `trl` - For training (optional)

## Troubleshooting

1. **Empty API responses**: Check Together API key in `.env`
2. **Low rewards**: Normal - information gain per question is typically small
3. **Negative advantages**: Expected - half will be below average
4. **Memory issues**: Reduce `num_generations` or `branch_factor`

## Next Steps

1. Collect data across multiple personas
2. Expand to other domains (restaurants, movies)
3. Train policy with TRL
4. Evaluate against baselines
5. Deploy trained policy