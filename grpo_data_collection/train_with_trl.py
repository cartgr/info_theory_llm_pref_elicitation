#!/usr/bin/env python
"""Example script for training a question-asking policy with TRL using collected GRPO data."""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


def load_grpo_dataset(data_path: str):
    """Load the collected GRPO dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    # GRPO just needs prompts - it will generate completions during training
    # We'll store the pre-computed rewards separately for the reward function
    prompts = data['prompt']

    # Create a simple dataset with just prompts
    dataset = Dataset.from_dict({'prompt': prompts})

    # Store pre-computed data for reward lookup
    global PRECOMPUTED_DATA
    PRECOMPUTED_DATA = data

    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train with GRPO using collected data")
    parser.add_argument("--data", help="Path to dataset JSON file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("--output", default="grpo_trained_models/question_policy", help="Output directory")
    args = parser.parse_args()

    # Configuration
    model_name = args.model
    data_path = args.data
    output_dir = args.output

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Use float32 for MPS (M2 Mac), bfloat16 for CUDA
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    if use_mps:
        print("Using MPS (Apple Silicon) with float32")
        dtype = torch.float32
        device_map = None  # MPS doesn't support device_map="auto"
    elif use_cuda:
        print("Using CUDA with bfloat16")
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        print("Using CPU with float32")
        dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map
    )

    if use_mps:
        model = model.to("mps")

    print("Loading dataset...")
    dataset = load_grpo_dataset(data_path)

    print(f"Dataset size: {len(dataset)}")
    print(f"First prompt: {dataset['prompt'][0][:100]}...")

    # GRPO Configuration
    training_args = GRPOConfig(
        output_dir=output_dir,

        # GRPO-specific parameters
        num_generations=8,  # Should match data collection

        # Training parameters
        learning_rate=1e-5,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        gradient_accumulation_steps=16,

        # Optimization
        warmup_steps=100,
        weight_decay=0.01,

        # Generation parameters
        max_completion_length=64,
        temperature=0.8,

        # Logging and saving
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        push_to_hub=False,
        report_to=["tensorboard"],

        # Efficiency
        bf16=use_cuda,  # Only use bf16 on CUDA
        fp16=False,
        gradient_checkpointing=True,
    )

    # Reward function - for now just return length as a simple reward
    # TODO: Use pre-computed rewards by matching completions
    def reward_func(completions, **kwargs):
        """Reward based on completion length (simple baseline)."""
        return [float(len(completion)) / 100.0 for completion in completions]

    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    print(f"Training complete! Model saved to {output_dir}/final_model")

    # Example generation with trained model
    print("\nExample generation with trained model:")
    prompt = "Context: [empty]\nTask: Discover the user's preferences about cars"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated question: {generated_question}")


if __name__ == "__main__":
    main()