#!/usr/bin/env python
"""Interactive chat with the trained question-asking model."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Chat with trained model")
    parser.add_argument(
        "--model-path",
        default="grpo_trained_models/question_policy",
        help="Path to trained model"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    # Convert to absolute path and validate
    model_path = Path(args.model_path).absolute()

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return

    required_files = ["config.json", "tokenizer_config.json"]
    missing = [f for f in required_files if not (model_path / f).exists()]
    if missing:
        print(f"Error: Missing required files in {model_path}:")
        for f in missing:
            print(f"  - {f}")
        return

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    # Detect device
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    if use_mps:
        device = "mps"
        dtype = torch.float32
    elif use_cuda:
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True
    ).to(device)

    model.eval()

    print("\n" + "="*60)
    print("Question-Asking Model Interactive Chat")
    print("="*60)
    print("The model will generate questions to discover your preferences.")
    print("Type 'quit' to exit\n")

    # Example prompts
    print("Example tasks:")
    print("  1. Context: [empty]\n     Task: Discover the user's preferences about cars")
    print("  2. Context: Q: What's your budget? A: Around $25k\n     Task: Ask a follow-up question")
    print("\n" + "="*60 + "\n")

    while True:
        print("Enter your prompt (or 'quit' to exit):")
        user_input = input("> ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Tokenize
        inputs = tokenizer(user_input, return_tensors="pt").to(device)

        # Generate
        print("\nGenerating question...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the new part (after the prompt)
        new_text = generated_text[len(user_input):].strip()

        print("\n" + "-"*60)
        print("Generated question:")
        print(new_text)
        print("-"*60 + "\n")


if __name__ == "__main__":
    main()
