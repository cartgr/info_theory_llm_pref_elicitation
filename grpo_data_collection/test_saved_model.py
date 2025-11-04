#!/usr/bin/env python
"""Test a saved model's generation quality."""

import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")

def test_model(model_path):
    """Test model generation."""
    model_path = Path(model_path).absolute()

    print(f"Loading model from: {model_path}")

    # Check files exist
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        local_files_only=True
    ).to(device)

    model.eval()

    # Test prompts in chat format
    test_conversations = [
        [
            {"role": "system", "content": "You are interviewing someone about their cars preferences."},
            {"role": "user", "content": "Ask one specific question to learn what they're looking for."}
        ],
        [
            {"role": "system", "content": "You are interviewing someone about their cars preferences. Here's the conversation so far:\n\nQ: What's your budget?\nA: Around $25,000"},
            {"role": "user", "content": "Ask one follow-up question to learn more."}
        ],
    ]

    print("\n" + "=" * 60)
    print("TESTING MODEL GENERATION")
    print("=" * 60)

    for i, messages in enumerate(test_conversations, 1):
        print(f"\n### Test {i} ###")
        print(f"Messages: {messages[0]['content'][:60]}...")

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"Formatted prompt:\n{prompt}")
        print("-" * 60)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        full_generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the new part
        if prompt in full_generated:
            question = full_generated.split(prompt)[-1].strip()
        else:
            # Fallback: decode only new tokens
            question = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        print(f"Generated (new tokens only): {question}")
        print(f"Full output: {full_generated[-200:]}")  # Show last 200 chars
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_saved_model.py <model_path>")
        sys.exit(1)

    test_model(sys.argv[1])
