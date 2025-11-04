#!/usr/bin/env python
"""Interactive chat with the trained question-asking model."""

import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


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

    # Example usage
    print("Example usage:")
    print("  1. 'Ask about cars' - Initial question")
    print("  2. 'Budget: $25k' - Follow-up question with context")
    print("  3. 'Q: What's your budget? A: Around $25k' - Multi-turn context")
    print("\nTip: Provide context for better follow-up questions!")
    print("\n" + "="*60 + "\n")

    conversation_history = []

    while True:
        print("Enter context/instruction (or 'quit' to exit, 'reset' to clear history):")
        user_input = input("> ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'reset':
            conversation_history = []
            print("Conversation history cleared.\n")
            continue

        if not user_input:
            continue

        # Build chat messages
        if conversation_history:
            # Multi-turn: use history as system context
            context = "\n".join(conversation_history)
            messages = [
                {"role": "system", "content": f"You are interviewing someone about their cars preferences. Conversation so far:\n{context}"},
                {"role": "user", "content": "Ask one follow-up question to learn more."}
            ]
        else:
            # First turn: use input as instruction
            messages = [
                {"role": "system", "content": "You are interviewing someone about their cars preferences."},
                {"role": "user", "content": f"Ask one specific question to {user_input}"}
            ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        print("\nGenerating question...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only new tokens
        question = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        print("\n" + "-"*60)
        print("Generated question:")
        print(question)
        print("-"*60 + "\n")

        # Ask if user wants to add this to history
        add_to_history = input("Add answer to conversation? (y/n, or type answer): ").strip()
        if add_to_history.lower() == 'y':
            answer = input("User's answer: ").strip()
            conversation_history.append(f"Q: {question}")
            conversation_history.append(f"A: {answer}")
        elif add_to_history.lower() != 'n' and add_to_history:
            # User provided answer directly
            conversation_history.append(f"Q: {question}")
            conversation_history.append(f"A: {add_to_history}")
        print()


if __name__ == "__main__":
    main()
