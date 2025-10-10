# PLLM MVE - Generative Active Preference Learning

Minimal Viable Experiment for testing whether a greedy question selection policy improves a log-scoring objective when eliciting preferences from a Participant LLM (PLLM).

## Overview

This project implements an information-theoretic approach to preference elicitation where:
- A **PLLM** (Participant LLM) simulates a user with a fixed persona
- An **Evaluator D** predicts preferences from dialogue transcripts
- A **greedy policy** selects questions to maximize expected information gain
- Performance is measured via a log-scoring objective

The key insight is that the reward `r_{t+1} = F(S_{t+1}) - F(S_t)` is proportional to the mutual information `I(Y; O_{t+1} | S_t, A_t)` when the evaluator is Bayes-optimal.

## Installation

```bash
# Clone the repository
cd pllm_mve

# Install the package
pip install -e .
```

## Setup

1. Create a `.env` file in the project root with your Together API key:
```bash
TOGETHER_API_KEY=your_api_key_here
```

2. Activate the conda environment (if using the specified setup):
```bash
conda activate llm-finetune
```

## Running Experiments

### Quick Start (Debug Mode)

Run a single seed for testing:
```bash
cd pllm_mve
python run_experiment.py --debug
```

### Full Experiment

Run the complete experiment with 10 seeds:
```bash
python run_experiment.py --config experiments/mve_one_persona.yml
```

### Custom Configuration

Modify `experiments/mve_one_persona.yml` or create your own config:
```yaml
name: mve_one_persona
output_dir: experiments/outputs/runs/mve_one_persona
num_seeds: 10

episode:
  persona: "likes fast cars but is on a budget"
  domain: cars
  num_items: 10      # K items to generate
  num_pairs: 20      # M pairs for evaluation
  num_turns: 5       # T dialogue turns
  model: "openai/gpt-oss-20b"
  temperature: 0.1
```

## Project Structure

```
pllm_mve/
├── src/pllm_mve/
│   ├── types.py              # Core data structures
│   ├── config.py             # Configuration management
│   ├── together_client.py    # Together API with JSON mode
│   ├── personas.py           # Persona management
│   ├── eval_items.py         # Item generation for domains
│   ├── pllm.py              # Participant LLM
│   ├── evaluator.py         # Evaluator D
│   ├── scoring.py           # Log-scoring functions
│   ├── qllm_policy.py       # Question selection policies
│   ├── rollout.py           # Episode execution
│   └── io_utils.py          # Logging and I/O
├── experiments/
│   ├── mve_one_persona.yml  # Default experiment config
│   └── outputs/             # Experiment results
└── run_experiment.py        # Main experiment runner
```

## Key Components

### PLLM (Participant LLM)
- Simulates a user with a fixed persona
- Provides consistent preference labels Y for evaluation pairs
- Answers questions during dialogue

### Evaluator D
- Reads only the answers from the transcript
- Outputs P(A > B | transcript) for any pair
- Uses JSON mode for structured outputs

### Scoring
The log-scoring objective:
```
F(s) = (1/|E|) * Σ_{(i,j) ∈ E} log p(y_{ij} | s)
```

### Policies
- **Greedy**: Selects questions that maximize expected score improvement
- **Random**: Baseline that selects questions randomly

## Output

Experiments generate:
- Episode logs with turn-by-turn scores and rewards
- Comparison between greedy and random policies
- Aggregated statistics across seeds
- CSV files for analysis

Results are saved to timestamped directories under `experiments/outputs/runs/`.

## Success Metrics

The experiment is successful if:
1. F(S_T) - F(S_0) > 0 (positive total reward)
2. Greedy policy outperforms random baseline over multiple seeds
3. Information gain aligns with reward improvements

## Notes

- Uses `openai/gpt-oss-20b` model via Together API
- Implements JSON mode with Pydantic schemas for reliability
- Probabilities are clipped to [1e-3, 1-1e-3] to avoid log(0)
- Labels are cached for consistency within episodes