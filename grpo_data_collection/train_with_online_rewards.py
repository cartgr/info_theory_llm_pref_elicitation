#!/usr/bin/env python
"""Train GRPO with online reward computation."""

import warnings
import json
import torch
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from online_reward import create_online_reward_function

# Suppress warnings from external libraries that we can't fix
warnings.filterwarnings("ignore", category=UserWarning, message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*aten::isin.*MPS.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*None of the inputs have requires_grad.*")


def load_prompts(data_path: str):
    """Load just the prompts from dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    prompts = data['prompt'] if isinstance(data, dict) else [d['prompt'] for d in data]
    return Dataset.from_dict({'prompt': prompts})


def main():
    parser = argparse.ArgumentParser(description="Train with GRPO using online rewards")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSON")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--output", default="grpo_trained_models/online_rewards", help="Output directory")
    parser.add_argument("--persona", default="likes fast cars but is on a budget", help="PLLM persona")
    parser.add_argument("--domain", default="cars", help="Domain")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-generations", type=int, default=4, help="Generations per prompt")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")

    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training with Online Rewards")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Persona: {args.persona}")
    print(f"Domain: {args.domain}")
    print(f"Generations per prompt: {args.num_generations}")
    print("=" * 60)

    # Load prompts
    print("\nLoading prompts...")
    dataset = load_prompts(args.prompts)
    print(f"Loaded {len(dataset)} prompts")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set pad token before loading model to avoid config mismatch warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Detect device
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    if use_mps:
        print("Using MPS (Apple Silicon) with float32")
        dtype = torch.float32
        device_map = None
    elif use_cuda:
        print("Using CUDA with bfloat16")
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        print("Using CPU with float32")
        dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device_map
    )

    if use_mps:
        model = model.to("mps")

    # Create online reward function
    print("\nInitializing online reward calculator...")
    print("This will make API calls to Together AI for each generated question.")
    print(f"Approximate API calls per step: {args.num_generations * 2} (PLLM + Evaluator)")

    reward_func = create_online_reward_function(
        persona=args.persona,
        domain=args.domain
    )

    # GRPO Configuration
    training_args = GRPOConfig(
        output_dir=args.output,

        # GRPO-specific
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,  # Must be divisible by num_generations

        # Training
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,

        # Generation settings
        max_completion_length=50,  # Short questions
        temperature=0.8,  # Some diversity

        # Logging
        logging_steps=1,
        save_steps=10,
        save_total_limit=3,
        report_to=[],  # Disable tensorboard for now

        # Efficiency
        bf16=use_cuda,
        fp16=False,
        gradient_checkpointing=True,
    )

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training with online rewards...")
    print("This will be SLOW due to API calls for each generated question!")
    print("=" * 60 + "\n")

    trainer.train()

    # Save model with absolute path
    output_path = Path(args.output).absolute() / "final_model"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to {output_path}")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Verify files were saved
    required_files = ["config.json", "model.safetensors"]
    for fname in required_files:
        if not (output_path / fname).exists():
            print(f"WARNING: {fname} not found in {output_path}")
        else:
            print(f"  ✓ Saved {fname}")

    print(f"\nModel saved successfully to: {output_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Test generation with proper chat format
    print("\nTesting trained model...")

    # Create test message in chat format
    test_messages = [
        {"role": "system", "content": "You are interviewing someone about their cars preferences."},
        {"role": "user", "content": "Ask one specific question to learn what they're looking for."}
    ]

    # Apply chat template
    test_prompt = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only new tokens
    question = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"\nSample generation:")
    print(f"  Task: Ask a car preference question")
    print(f"  Generated: {question}")
    print(f"\nFull output saved to: {output_path}")


if __name__ == "__main__":
    main()