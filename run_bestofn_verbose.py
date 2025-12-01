#!/usr/bin/env python3
"""
Best-of-N experiment with VERBOSE logging.

This version logs everything: prompts, personas, candidates, scoring details.
All output goes to both console and a log file.
"""

import argparse
import random
from pathlib import Path
import sys
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pllm_mve.config import load_config, save_config
from pllm_mve.types import EpisodeConfig, EpisodeEvalSet, Transcript
from pllm_mve.together_client import TogetherChat
from pllm_mve.eval_items import generate_items_for_domain
from pllm_mve.pllm import PLLM
from pllm_mve.evaluator import EvaluatorD
from pllm_mve.scoring import log_score
from pllm_mve.io_utils import create_experiment_dir, EpisodeLogger, save_json
from pllm_mve.responder import ResponderLLM
from pllm_mve.verbose_logger import init_logger, get_logger, close_logger


def log_experiment_header(config):
    """Log experiment configuration."""
    L = get_logger()
    L.header("BEST-OF-N EXPERIMENT - VERBOSE MODE")
    L.blank()
    L.key_value("Timestamp", datetime.now().isoformat())
    L.key_value("Experiment Name", config.name)
    L.key_value("Number of Seeds", config.num_seeds)
    L.blank()
    L.subheader("Episode Configuration")
    L.key_value("Domain", config.episode.domain)
    L.key_value("Persona", config.episode.persona)
    L.key_value("Number of Items", config.episode.num_items)
    L.key_value("Number of Pairs (E)", config.episode.num_pairs)
    L.key_value("Number of Turns", config.episode.num_turns)
    L.key_value("Model", config.episode.model)
    L.key_value("Temperature", config.episode.temperature)
    L.key_value("Max Tokens", config.episode.max_tokens)


def log_items(items: List[str]):
    """Log all generated items."""
    L = get_logger()
    L.subheader(f"Generated Items ({len(items)} total)")
    L.numbered_list(items)


def log_pairs(pairs: List[tuple], items: List[str], label: str = "Pairs"):
    """Log pairs with item names."""
    L = get_logger()
    L.subheader(f"{label} ({len(pairs)} pairs)")
    for i, (a, b) in enumerate(pairs):
        L.log(f"  Pair {i}: ({a}, {b}) = '{items[a][:40]}...' vs '{items[b][:40]}...'")


def log_labels(labels: Dict[tuple, int], items: List[str]):
    """Log all preference labels."""
    L = get_logger()
    L.subheader(f"Preference Labels ({len(labels)} labels)")
    L.table_row(["Pair", "Label", "Preferred Item"], [12, 8, 50])
    L.divider()
    for (a, b), label in labels.items():
        preferred = items[a] if label == 1 else items[b]
        label_str = "A>B" if label == 1 else "B>A"
        L.table_row([f"({a},{b})", label_str, preferred[:48]], [12, 8, 50])


def log_persona_prompt(persona: str):
    """Log the persona system prompt."""
    L = get_logger()
    L.subheader("Persona System Prompt")
    system = (
        f"You are a human participant with this persona:\n"
        f"{persona}\n\n"
        f"ACT LIKE A HUMAN. Answer naturally and conversationally. "
        f"Be authentic and stay in character."
    )
    L.prompt_block("SYSTEM", system)


def log_question_generation_prompt(domain: str, items: List[str], transcript: Transcript, pool_size: int):
    """Log the question generation prompt."""
    L = get_logger()
    L.subheader("Question Generation Prompt")

    # Build context
    context = ""
    if transcript.turns:
        context = "Previous conversation:\n"
        for i, turn in enumerate(transcript.turns):
            context += f"Q{i+1}: {turn.question}\n"
            context += f"A{i+1}: {turn.answer}\n\n"

    system = (
        f"You are an expert interviewer trying to learn someone's preferences about {domain}. "
        f"Generate informative questions that will help uncover their preferences."
    )

    user = (
        f"{context}\n"
        f"Generate {pool_size} diverse questions to ask next. Each question should:\n"
        f"- Help reveal preferences about {domain}\n"
        f"- Be natural and conversational\n"
        f"- Be different from the others (diverse approaches)\n\n"
        f"Return ONLY the questions, one per line, numbered 1-{pool_size}."
    )

    L.prompt_block("SYSTEM", system)
    L.prompt_block("USER", user)


def log_evaluator_prompt(answers: str, option_a: str, option_b: str):
    """Log the evaluator prompt."""
    L = get_logger()
    system = (
        "You are a calibrated evaluator. Based on the participant's answers, "
        "determine if they prefer option A over option B. "
        "Respond with only a single letter: A or B."
    )
    user = (
        f"Transcript (participant answers only):\n{answers}\n\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n\n"
        f"Which option does the participant prefer? Answer with only 'A' or 'B'."
    )
    L.prompt_block("EVALUATOR SYSTEM", system)
    L.prompt_block("EVALUATOR USER", user)


def generate_candidate_questions_verbose(
    domain: str,
    items: List[str],
    transcript: Transcript,
    pool_size: int,
    client: TogetherChat
) -> List[str]:
    """Generate candidate questions with verbose logging."""
    L = get_logger()
    L.section("Generating Candidate Questions")

    # Log the prompt
    log_question_generation_prompt(domain, items, transcript, pool_size)

    # Build context
    context = ""
    if transcript.turns:
        context = "Previous conversation:\n"
        for i, turn in enumerate(transcript.turns):
            context += f"Q{i+1}: {turn.question}\n"
            context += f"A{i+1}: {turn.answer}\n\n"

    system = (
        f"You are an expert interviewer trying to learn someone's preferences about {domain}. "
        f"Generate informative questions that will help uncover their preferences."
    )

    user = (
        f"{context}\n"
        f"Generate {pool_size} diverse questions to ask next. Each question should:\n"
        f"- Help reveal preferences about {domain}\n"
        f"- Be natural and conversational\n"
        f"- Be different from the others (diverse approaches)\n\n"
        f"Return ONLY the questions, one per line, numbered 1-{pool_size}."
    )

    # Generate
    response = client.chat(system=system, user=user, temperature=0.8, max_tokens=500)

    L.subheader("Raw LLM Response")
    L.log(response)

    # Parse questions
    import re
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip('"\'')
        if line and (line.endswith('?') or len(line) > 10):
            questions.append(line)

    while len(questions) < pool_size:
        questions.append(f"What's most important to you when choosing {domain}?")

    questions = questions[:pool_size]

    L.subheader("Parsed Candidate Questions")
    L.numbered_list(questions)

    return questions


def score_with_logging(
    eval_set: EpisodeEvalSet,
    transcript: Transcript,
    evaluator: EvaluatorD,
    label: str = "Scoring"
) -> Tuple[float, float]:
    """Score transcript with detailed logging."""
    L = get_logger()
    L.section(f"{label} on {len(eval_set.pairs)} pairs")

    if not eval_set.pairs:
        L.log("No pairs to evaluate")
        return 0.0, 0.0

    import math
    total_log_prob = 0.0

    # Log current transcript state
    if transcript.turns:
        L.log("Current transcript:")
        with L.indent():
            for i, turn in enumerate(transcript.turns):
                L.log(f"Q{i+1}: {turn.question}")
                L.log(f"A{i+1}: {turn.answer}")
    else:
        L.log("Transcript: (empty)")

    L.blank()
    L.table_row(["Pair", "True", "P(A>B)", "Correct?", "LogP", "Items"], [10, 6, 8, 8, 10, 40])
    L.divider(width=82)

    correct_count = 0
    for pair in eval_set.pairs:
        i, j = pair
        p = evaluator.evaluate_pair(i, j, eval_set.items, transcript)
        y = eval_set.labels.get(pair, 0)

        # Compute log probability
        if y == 1:
            q = p
        else:
            q = 1.0 - p
        q = min(1.0 - 1e-6, max(1e-6, q))
        log_p = math.log(q)
        total_log_prob += log_p

        # Check correctness
        pred_correct = (p > 0.5) == (y == 1)
        if pred_correct:
            correct_count += 1

        true_str = "A>B" if y == 1 else "B>A"
        correct_str = "YES" if pred_correct else "NO"
        items_str = f"{eval_set.items[i][:18]}.. vs {eval_set.items[j][:18]}.."

        L.table_row(
            [f"({i},{j})", true_str, f"{p:.3f}", correct_str, f"{log_p:.4f}", items_str],
            [10, 6, 8, 8, 10, 40]
        )

    avg_score = total_log_prob / len(eval_set.pairs)
    accuracy = correct_count / len(eval_set.pairs)

    L.divider(width=82)
    L.key_value("Total Log Prob", f"{total_log_prob:.4f}")
    L.key_value("Average Score", f"{avg_score:.4f}")
    L.key_value("Accuracy @ 0.5", f"{accuracy:.2%} ({correct_count}/{len(eval_set.pairs)})")

    return avg_score, accuracy


def run_bestofn_episode_verbose(
    config: EpisodeConfig,
    persona_pllm: PLLM,
    responder: ResponderLLM,
    evaluator: EvaluatorD,
    eval_set_E: EpisodeEvalSet,
    logger: EpisodeLogger,
    num_candidates: int = 5,
    num_samples: int = 3,
    use_pllm_responder: bool = False,
) -> Tuple[Transcript, Dict]:
    """Best-of-N episode with verbose logging."""
    L = get_logger()
    transcript = Transcript()

    L.header("BEST-OF-N EPISODE")
    L.key_value("Number of turns", config.num_turns)
    L.key_value("Candidates per turn (k)", num_candidates)
    L.key_value("Samples per candidate (t)", num_samples)

    # Log persona prompt
    log_persona_prompt(config.persona)

    # Initial baseline
    L.header("INITIAL BASELINE (empty transcript)")
    f_prev, _ = score_with_logging(eval_set_E, transcript, evaluator, "Initial Score on E")

    for turn in range(config.num_turns):
        L.header(f"TURN {turn + 1} / {config.num_turns}", char="*")

        # Generate candidates (only show items that appear in E's pairs)
        items_in_E = eval_set_E.get_items_in_pairs()
        questions = generate_candidate_questions_verbose(
            domain=config.domain,
            items=items_in_E,
            transcript=transcript,
            pool_size=num_candidates,
            client=evaluator.chat,
        )

        baseline_score = f_prev
        best_q = None
        best_gain_stat = float("-inf")
        all_candidate_results = []

        L.section("Evaluating Each Candidate Question")

        # Evaluate each candidate
        for q_idx, q in enumerate(questions):
            L.subheader(f"Candidate {q_idx + 1}: {q}")

            # Get all samples in ONE API call (batched)
            L.log(f"\n  Sampling {num_samples} hypothetical answers (batched)...")
            sampled_answers = responder.sample_answers(q, num_samples=num_samples)

            gains = []
            sample_details = []

            for s_idx, a_sample in enumerate(sampled_answers):
                L.log(f"\n  Sample {s_idx + 1}/{num_samples}:")
                L.log(f"    Hypothetical answer: {a_sample}")

                # Create trial transcript
                trial = Transcript(turns=transcript.turns.copy())
                trial.add_turn(q, a_sample)

                # Score on E
                new_score = log_score(eval_set_E, trial, evaluator, verbose=False)
                gain = new_score - baseline_score

                L.log(f"    Trial score: {new_score:.4f}, Gain: {gain:.4f}")

                gains.append(gain)
                sample_details.append({
                    'answer': a_sample,
                    'score': new_score,
                    'gain': gain
                })

            # Aggregate: use mean for PLLM responder (same persona), max for generic responder
            if use_pllm_responder:
                gain_stat = float(np.mean(gains)) if gains else 0.0
                agg_type = "Mean"
            else:
                gain_stat = float(np.max(gains)) if gains else 0.0
                agg_type = "Max"

            L.log(f"\n  Summary for candidate {q_idx + 1}:")
            L.log(f"    Gains: {[f'{g:.4f}' for g in gains]}")
            L.log(f"    {agg_type} gain: {gain_stat:.4f}")

            all_candidate_results.append({
                'question': q,
                'gain': gain_stat,
                'agg_type': agg_type,
                'samples': sample_details
            })

            if gain_stat > best_gain_stat:
                best_gain_stat = gain_stat
                best_q = q

        # Summary of candidate selection
        L.subheader("Candidate Selection Summary")
        agg_label = "Mean Gain" if use_pllm_responder else "Max Gain"
        L.table_row(["#", agg_label, "Question"], [4, 12, 60])
        L.divider(width=76)
        for i, result in enumerate(all_candidate_results):
            marker = " <-- SELECTED" if result['question'] == best_q else ""
            L.table_row(
                [str(i+1), f"{result['gain']:.4f}", result['question'][:55] + "..." + marker],
                [4, 12, 60]
            )

        # Ask real question to PLLM
        L.section(f"Asking Selected Question to PLLM")
        L.log(f"Question: {best_q}")

        real_answer = persona_pllm.answer_question(best_q)
        L.log(f"Real Answer: {real_answer}")

        transcript.add_turn(best_q, real_answer)

        # Score after real answer
        L.section("Scoring After Real Answer")
        f_new, _ = score_with_logging(eval_set_E, transcript, evaluator, f"Score on E after turn {turn + 1}")

        reward = f_new - f_prev

        L.subheader(f"Turn {turn + 1} Summary")
        L.key_value("Previous score", f"{f_prev:.4f}")
        L.key_value("New score", f"{f_new:.4f}")
        L.key_value("Actual reward", f"{reward:.4f}")
        L.key_value("Expected max gain", f"{best_gain_stat:.4f}")
        L.key_value("Gain vs Expected", f"{reward - best_gain_stat:.4f}")

        logger.log_turn(
            turn=turn,
            question=best_q,
            answer=real_answer,
            score=f_new,
            reward=reward,
            expected_gain=best_gain_stat,
            policy="bestofn",
        )

        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary


def run_direct_episode_verbose(
    config: EpisodeConfig,
    persona_pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set_E: EpisodeEvalSet,
    logger: EpisodeLogger,
    client: TogetherChat,
    num_candidates: int = 5,
) -> Tuple[Transcript, Dict]:
    """Direct baseline episode with verbose logging."""
    L = get_logger()
    transcript = Transcript()

    L.header("DIRECT BASELINE EPISODE")
    L.key_value("Number of turns", config.num_turns)
    L.key_value("Candidates per turn", num_candidates)
    L.log("(Selecting randomly from candidates, no scoring)")

    # Initial baseline
    L.header("INITIAL BASELINE (empty transcript)")
    f_prev, _ = score_with_logging(eval_set_E, transcript, evaluator, "Initial Score on E")

    for turn in range(config.num_turns):
        L.header(f"DIRECT TURN {turn + 1} / {config.num_turns}", char="*")

        # Generate candidates (only items in E's pairs)
        items_in_E = eval_set_E.get_items_in_pairs()
        questions = generate_candidate_questions_verbose(
            domain=config.domain,
            items=items_in_E,
            transcript=transcript,
            pool_size=num_candidates,
            client=client,
        )

        # Random selection
        question = random.choice(questions)
        L.section("Random Selection")
        L.log(f"Selected question: {question}")

        # Ask PLLM
        answer = persona_pllm.answer_question(question)
        L.log(f"Answer: {answer}")

        transcript.add_turn(question, answer)

        # Score
        L.section("Scoring After Answer")
        f_new, _ = score_with_logging(eval_set_E, transcript, evaluator, f"Score on E after turn {turn + 1}")

        reward = f_new - f_prev

        L.subheader(f"Turn {turn + 1} Summary")
        L.key_value("Previous score", f"{f_prev:.4f}")
        L.key_value("New score", f"{f_new:.4f}")
        L.key_value("Reward", f"{reward:.4f}")

        logger.log_turn(
            turn=turn,
            question=question,
            answer=answer,
            score=f_new,
            reward=reward,
            policy="direct",
        )

        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary


def compare_policies_verbose(
    config: EpisodeConfig,
    persona_pllm: PLLM,
    responder: ResponderLLM,
    evaluator: EvaluatorD,
    eval_set_E: EpisodeEvalSet,
    eval_set_T: EpisodeEvalSet,
    output_dir: Path,
    num_candidates: int = 5,
    num_samples: int = 3,
    client: TogetherChat = None,
    use_pllm_responder: bool = False,
) -> Dict:
    """Compare policies with verbose logging."""
    L = get_logger()

    if client is None:
        client = TogetherChat(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    # Best-of-N
    L.header("RUNNING BEST-OF-N POLICY", char="#", width=80)
    bestofn_logger = EpisodeLogger(output_dir)
    bestofn_transcript, bestofn_summary = run_bestofn_episode_verbose(
        config=config,
        persona_pllm=persona_pllm,
        responder=responder,
        evaluator=evaluator,
        eval_set_E=eval_set_E,
        logger=bestofn_logger,
        num_candidates=num_candidates,
        num_samples=num_samples,
        use_pllm_responder=use_pllm_responder,
    )
    bestofn_logger.save(suffix="_bestofn")

    # Direct baseline
    L.header("RUNNING DIRECT BASELINE", char="#", width=80)
    direct_logger = EpisodeLogger(output_dir)
    direct_transcript, direct_summary = run_direct_episode_verbose(
        config=config,
        persona_pllm=persona_pllm,
        evaluator=evaluator,
        eval_set_E=eval_set_E,
        logger=direct_logger,
        client=client,
        num_candidates=num_candidates,
    )
    direct_logger.save(suffix="_direct")

    # Final evaluation on T
    L.header("FINAL EVALUATION ON T (all pairs)")

    L.subheader("Best-of-N Final Transcript")
    for i, turn in enumerate(bestofn_transcript.turns):
        L.log(f"Q{i+1}: {turn.question}")
        L.log(f"A{i+1}: {turn.answer}")
    L.blank()
    bestofn_final_T, bestofn_accuracy_T = score_with_logging(eval_set_T, bestofn_transcript, evaluator, "Best-of-N on T")

    L.subheader("Direct Final Transcript")
    for i, turn in enumerate(direct_transcript.turns):
        L.log(f"Q{i+1}: {turn.question}")
        L.log(f"A{i+1}: {turn.answer}")
    L.blank()
    direct_final_T, direct_accuracy_T = score_with_logging(eval_set_T, direct_transcript, evaluator, "Direct on T")

    # Comparison
    total_reward_diff = bestofn_summary["total_reward"] - direct_summary["total_reward"]
    final_score_diff = bestofn_final_T - direct_final_T
    accuracy_diff = bestofn_accuracy_T - direct_accuracy_T

    comparison = {
        "bestofn": {
            **bestofn_summary,
            "final_logscore_T": bestofn_final_T,
            "final_accuracy_T": bestofn_accuracy_T,
        },
        "direct": {
            **direct_summary,
            "final_logscore_T": direct_final_T,
            "final_accuracy_T": direct_accuracy_T,
        },
        "improvement": {
            "total_reward": total_reward_diff,
            "final_score": final_score_diff,
            "total_reward_E": total_reward_diff,
            "final_logscore_T": final_score_diff,
            "final_accuracy_T": accuracy_diff,
        },
        "config": {
            "num_candidates": num_candidates,
            "num_samples": num_samples,
        },
    }

    L.header("COMPARISON RESULTS")
    L.key_value("Best-of-N total reward (E)", f"{bestofn_summary['total_reward']:.4f}")
    L.key_value("Direct total reward (E)", f"{direct_summary['total_reward']:.4f}")
    L.key_value("Improvement (reward E)", f"{total_reward_diff:.4f}")
    L.blank()
    L.key_value("Best-of-N final score (T)", f"{bestofn_final_T:.4f}")
    L.key_value("Direct final score (T)", f"{direct_final_T:.4f}")
    L.key_value("Improvement (score T)", f"{final_score_diff:.4f}")
    L.blank()
    L.key_value("Best-of-N accuracy (T)", f"{bestofn_accuracy_T:.2%}")
    L.key_value("Direct accuracy (T)", f"{direct_accuracy_T:.2%}")
    L.key_value("Improvement (accuracy T)", f"{accuracy_diff:+.2%}")

    return comparison


def setup_episode_verbose(config: EpisodeConfig, seed: int = None) -> Tuple[EpisodeEvalSet, EpisodeEvalSet]:
    """Setup episode with verbose logging."""
    L = get_logger()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    L.header(f"EPISODE SETUP (seed={seed})")

    # Build items
    temp_chat = TogetherChat(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    items = generate_items_for_domain(
        domain=config.domain,
        persona=config.persona,
        num_items=config.num_items,
        chat_client=temp_chat,
    )

    log_items(items)

    # Build pairs
    all_pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
    L.key_value("Total possible pairs (T)", len(all_pairs))

    # Sample E
    num_pairs_E = min(config.num_pairs, len(all_pairs))
    pairs_E = random.sample(all_pairs, num_pairs_E)
    L.key_value("Eval subset size (E)", len(pairs_E))

    log_pairs(pairs_E, items, "Evaluation Pairs (E)")

    eval_set_E = EpisodeEvalSet(items=items, pairs=pairs_E, labels={})
    eval_set_T = EpisodeEvalSet(items=items, pairs=all_pairs, labels={})

    return eval_set_E, eval_set_T


def run_single_seed_verbose(
    config,
    seed: int,
    output_dir: Path,
    num_candidates: int = 5,
    num_samples: int = 3,
    use_logprobs: bool = False,
    include_questions: bool = False,
    use_pllm_responder: bool = False,
) -> dict:
    """Run single seed with verbose logging."""
    L = get_logger()

    L.header(f"SEED {seed}", char="#", width=80)

    # Setup episode
    eval_set_E, eval_set_T = setup_episode_verbose(config.episode, seed)

    # Save episode metadata
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    episode_data = {
        "seed": seed,
        "items": eval_set_T.items,
        "pairs_E": eval_set_E.pairs,
        "pairs_T": eval_set_T.pairs,
    }
    save_json(episode_data, seed_dir / "episode_data.json")

    # Initialize PLLM
    L.section("Initializing Persona PLLM")
    chat_persona = TogetherChat(
        model=config.episode.model,
        max_tokens=config.episode.max_tokens,
        temperature=config.episode.temperature,
    )
    persona_pllm = PLLM(chat_persona)
    persona_pllm.initialize_persona(config.episode.persona, eval_set_T)

    L.log(f"Persona: {config.episode.persona}")
    L.log(f"Model: {config.episode.model}")

    # Label pairs
    L.section("Labeling All Pairs with Persona PLLM")
    L.log("This calls the LLM for each pair to get ground truth preferences...")

    # Log the preference prompt format
    L.subheader("Preference Labeling Prompt Format")
    system = (
        f"You are simulating a participant with the following persona:\n"
        f"{config.episode.persona}\n"
        f"Answer consistently with this persona across the entire episode. Be decisive.\n"
        f"Always respond with valid JSON."
    )
    user_template = (
        f"Based on the persona, choose between option A or B:\n\n"
        f"A: [ITEM A]\n"
        f"B: [ITEM B]\n\n"
        f'Return JSON in this exact format: {{"choice": "A"}} or {{"choice": "B"}}'
    )
    L.prompt_block("SYSTEM", system)
    L.prompt_block("USER (template)", user_template)

    L.log("\nLabeling E pairs...")
    for pair in eval_set_E.pairs:
        label = persona_pllm.label_eval_question(pair[0], pair[1])
        eval_set_E.labels[pair] = label

    L.log("Labeling T pairs...")
    for pair in eval_set_T.pairs:
        label = persona_pllm.label_eval_question(pair[0], pair[1])
        eval_set_T.labels[pair] = label

    log_labels(eval_set_E.labels, eval_set_E.items)

    # Initialize evaluator
    L.section("Initializing Evaluator D (Judge)")
    evaluator = EvaluatorD(
        chat_persona,
        include_questions=include_questions,
        use_logprobs=use_logprobs
    )
    L.key_value("Use logprobs", use_logprobs)
    L.key_value("Include questions", include_questions)

    # Log evaluator prompt format
    L.subheader("Evaluator Prompt Format")
    if include_questions:
        eval_system = (
            "You are a calibrated evaluator. Based on the conversation transcript, "
            "determine if the participant prefers option A over option B."
        )
        eval_user_template = (
            f"Conversation transcript:\n[Q&A]\n\n"
            f"Option A: [ITEM A]\n"
            f"Option B: [ITEM B]\n\n"
        )
    else:
        eval_system = (
            "You are a calibrated evaluator. Based on the participant's answers, "
            "determine if they prefer option A over option B."
        )
        eval_user_template = (
            f"Transcript (participant answers only):\n[ANSWERS]\n\n"
            f"Option A: [ITEM A]\n"
            f"Option B: [ITEM B]\n\n"
        )

    if use_logprobs:
        eval_system += " Respond with only a single letter: A or B."
        eval_user_template += "Which option does the participant prefer? Answer with only 'A' or 'B'."
    else:
        eval_system += " Always respond with valid JSON."
        eval_user_template += 'Return the probability P(A>B | transcript) as JSON: {"probability": 0.X}'
    L.prompt_block("SYSTEM", eval_system)
    L.prompt_block("USER (template)", eval_user_template)

    # Initialize ResponderLLM
    L.section("Initializing ResponderLLM (Proposal Model P)")
    if use_pllm_responder:
        # Oracle mode: use PLLM as responder (for debugging)
        L.log("*** ORACLE MODE: Using PLLM as responder ***")
        L.log("This is 'cheating' - the responder knows the true persona")
        L.log("Useful for debugging to verify Best-of-N works when responder matches user")
        responder = persona_pllm  # Use PLLM directly
    else:
        # Normal mode: generic responder
        chat_responder = TogetherChat(
            model=config.episode.model,
            max_tokens=config.episode.max_tokens,
            temperature=config.episode.temperature,
        )
        responder = ResponderLLM(chat_responder, use_persona=False)
        responder.initialize(domain=config.episode.domain)
        L.log(f"ResponderLLM initialized WITHOUT persona (generic user mode)")
        L.log(f"Domain: {config.episode.domain}")
        L.log("This generates diverse hypothetical answers without biasing toward the true persona")
        L.log("Answers are batched: multiple samples per question in ONE API call")

    # Run comparison
    comparison = compare_policies_verbose(
        config=config.episode,
        persona_pllm=persona_pllm,
        responder=responder,
        evaluator=evaluator,
        eval_set_E=eval_set_E,
        eval_set_T=eval_set_T,
        output_dir=seed_dir,
        num_candidates=num_candidates,
        num_samples=num_samples,
        client=chat_persona,
        use_pllm_responder=use_pllm_responder,
    )

    save_json(comparison, seed_dir / "comparison.json")
    return comparison


def main():
    """Main entry point with verbose logging."""
    parser = argparse.ArgumentParser(description="Run Best-of-N experiment (VERBOSE)")
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/mve_one_persona.yml"),
        help="Path to experiment config file"
    )
    parser.add_argument(
        "--num-candidates", type=int, default=3,
        help="Number of candidate questions (k)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of samples per candidate (t)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of seeds to run"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for logs"
    )
    parser.add_argument(
        "--no-console", action="store_true",
        help="Disable console output (only write to log file)"
    )
    parser.add_argument(
        "--use-logprobs", action="store_true",
        help="Use logprobs for probability estimation (default: text-based probability)"
    )
    parser.add_argument(
        "--include-questions", action="store_true",
        help="Include questions (not just answers) in evaluator context"
    )
    parser.add_argument(
        "--use-pllm-responder", action="store_true",
        help="Use PLLM (persona) as responder instead of generic ResponderLLM (for debugging/oracle mode)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config.num_seeds = args.num_seeds

    # Create experiment directory
    exp_dir = create_experiment_dir(
        base_dir=args.output_dir or Path("experiments/outputs/runs"),
        experiment_name=f"bestofn_verbose_{config.name}"
    )
    config.output_dir = exp_dir

    # Initialize logger
    log_file = exp_dir / "verbose_trace.log"
    init_logger(log_file=log_file, console=not args.no_console)
    L = get_logger()

    L.log(f"Log file: {log_file}")

    # Save config
    save_config(config, exp_dir / "config.yml")

    # Log experiment header
    log_experiment_header(config)
    L.key_value("Candidates (k)", args.num_candidates)
    L.key_value("Samples (t)", args.num_samples)
    L.key_value("Use logprobs", args.use_logprobs)
    L.key_value("Include questions", args.include_questions)
    L.key_value("Use PLLM as responder", args.use_pllm_responder)

    # Run seeds
    all_results = []
    for seed in range(config.num_seeds):
        try:
            comparison = run_single_seed_verbose(
                config, seed, exp_dir,
                num_candidates=args.num_candidates,
                num_samples=args.num_samples,
                use_logprobs=args.use_logprobs,
                include_questions=args.include_questions,
                use_pllm_responder=args.use_pllm_responder
            )
            all_results.append(comparison)
        except Exception as e:
            L.log(f"ERROR in seed {seed}: {e}")
            import traceback
            L.log(traceback.format_exc())
            raise

    # Aggregate results
    if all_results:
        bestofn_rewards = [r["bestofn"]["total_reward"] for r in all_results]
        direct_rewards = [r["direct"]["total_reward"] for r in all_results]
        improvements = [r["improvement"]["total_reward"] for r in all_results]

        # Accuracy on T
        bestofn_accuracies = [r["bestofn"]["final_accuracy_T"] for r in all_results]
        direct_accuracies = [r["direct"]["final_accuracy_T"] for r in all_results]
        accuracy_improvements = [r["improvement"]["final_accuracy_T"] for r in all_results]

        aggregate = {
            "num_seeds": len(all_results),
            "bestofn": {
                "mean_reward": float(np.mean(bestofn_rewards)),
                "std_reward": float(np.std(bestofn_rewards)),
                "all_rewards": bestofn_rewards,
                "mean_accuracy": float(np.mean(bestofn_accuracies)),
                "std_accuracy": float(np.std(bestofn_accuracies)),
                "all_accuracies": bestofn_accuracies,
            },
            "direct": {
                "mean_reward": float(np.mean(direct_rewards)),
                "std_reward": float(np.std(direct_rewards)),
                "all_rewards": direct_rewards,
                "mean_accuracy": float(np.mean(direct_accuracies)),
                "std_accuracy": float(np.std(direct_accuracies)),
                "all_accuracies": direct_accuracies,
            },
            "improvement": {
                "mean": float(np.mean(improvements)),
                "std": float(np.std(improvements)),
                "all": improvements,
                "success_rate": sum(1 for x in improvements if x > 0) / len(improvements),
                "mean_accuracy": float(np.mean(accuracy_improvements)),
                "std_accuracy": float(np.std(accuracy_improvements)),
                "all_accuracy": accuracy_improvements,
            },
        }

        save_json(aggregate, exp_dir / "aggregate_results.json")

        L.header("AGGREGATE RESULTS")
        L.key_value("Seeds completed", len(all_results))
        L.blank()
        L.subheader("Accuracy @ 0.5 (on T)")
        L.key_value("Best-of-N mean accuracy", f"{aggregate['bestofn']['mean_accuracy']:.2%} +/- {aggregate['bestofn']['std_accuracy']:.2%}")
        L.key_value("Direct mean accuracy", f"{aggregate['direct']['mean_accuracy']:.2%} +/- {aggregate['direct']['std_accuracy']:.2%}")
        L.key_value("Accuracy improvement", f"{aggregate['improvement']['mean_accuracy']:+.2%} +/- {aggregate['improvement']['std_accuracy']:.2%}")
        L.blank()
        L.subheader("Log Score (reward)")
        L.key_value("Best-of-N mean reward", f"{aggregate['bestofn']['mean_reward']:.4f} +/- {aggregate['bestofn']['std_reward']:.4f}")
        L.key_value("Direct mean reward", f"{aggregate['direct']['mean_reward']:.4f} +/- {aggregate['direct']['std_reward']:.4f}")
        L.key_value("Reward improvement", f"{aggregate['improvement']['mean']:.4f} +/- {aggregate['improvement']['std']:.4f}")
        L.key_value("Success rate (reward > 0)", f"{aggregate['improvement']['success_rate']:.2%}")

    L.header("EXPERIMENT COMPLETE")
    L.log(f"Results saved to: {exp_dir}")
    L.log(f"Verbose log: {log_file}")

    close_logger()


if __name__ == "__main__":
    main()
