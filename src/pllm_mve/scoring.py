"""Scoring functions for preference learning."""

import math
from typing import Dict, Tuple
from .types import EpisodeEvalSet, Transcript
from .evaluator import EvaluatorD


def log_score(
    eval_set: EpisodeEvalSet,
    transcript: Transcript,
    evaluator: EvaluatorD,
    verbose: bool = False
) -> float:
    """
    Compute log-scoring objective F(s).

    F(s) = (1/|E|) * sum_{(i,j) in E} log p(y_{ij} | s)
    """
    if not eval_set.pairs:
        return 0.0

    total_log_prob = 0.0
    predictions = []  # Track for verbose output

    for pair in eval_set.pairs:
        i, j = pair
        # Get evaluator's probability
        p = evaluator.evaluate_pair(i, j, eval_set.items, transcript)

        # Get true label
        y = eval_set.labels.get(pair, 0)

        # Track prediction
        predictions.append({
            'pair': pair,
            'true_label': y,
            'pred_prob': p,
            'item_a': eval_set.items[i][:40],
            'item_b': eval_set.items[j][:40]
        })

        # Compute log probability of the correct label
        if y == 1:
            # Label is 1, so we want P(A > B)
            q = p
        else:
            # Label is 0, so we want P(B > A) = 1 - P(A > B)
            q = 1.0 - p

        # Clip to avoid log(0)
        q = min(1.0 - 1e-6, max(1e-6, q))

        total_log_prob += math.log(q)

    # Print verbose output if requested
    if verbose:
        print(f"\n  Evaluation on {len(eval_set.pairs)} pairs:")
        print(f"  {'Pair':<8} {'True':<6} {'Pred':<6} {'Item A vs Item B'}")
        print(f"  {'-'*70}")
        for pred in predictions:
            pair_str = f"({pred['pair'][0]},{pred['pair'][1]})"
            true_str = "A>B" if pred['true_label'] == 1 else "B>A"
            pred_str = f"{pred['pred_prob']:.3f}"
            print(f"  {pair_str:<8} {true_str:<6} {pred_str:<6} {pred['item_a']}... vs {pred['item_b']}...")

        # Calculate accuracy if we threshold at 0.5
        correct = sum(1 for p in predictions if (p['pred_prob'] > 0.5) == (p['true_label'] == 1))
        accuracy = correct / len(predictions)
        print(f"\n  Accuracy @ 0.5: {accuracy:.2%} ({correct}/{len(predictions)})")
        print(f"  Average score: {total_log_prob / len(eval_set.pairs):.4f}\n")

    # Return average log probability
    return total_log_prob / len(eval_set.pairs)


def compute_reward(
    score_new: float,
    score_old: float
) -> float:
    """
    Compute reward as difference in scores.

    r_{t+1} = F(S_{t+1}) - F(S_t)
    """
    return score_new - score_old


def compute_information_gain_approx(
    eval_set: EpisodeEvalSet,
    transcript_before: Transcript,
    transcript_after: Transcript,
    evaluator: EvaluatorD
) -> float:
    """
    Approximate information gain from adding a turn.

    This is proportional to the reward if evaluator is Bayes-optimal.
    """
    score_before = log_score(eval_set, transcript_before, evaluator)
    score_after = log_score(eval_set, transcript_after, evaluator)
    return compute_reward(score_after, score_before)