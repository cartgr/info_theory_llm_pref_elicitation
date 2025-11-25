"""Together API client with JSON mode support."""

import os
import json
import time
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from together import Together
import together.error

from .config import get_api_key


class ABLabel(BaseModel):
    """Schema for A/B preference labels."""
    label: str = Field(
        pattern="^(A|B)$",
        description="Return exactly 'A' or 'B'"
    )


class EvalProb(BaseModel):
    """Schema for evaluator probability output."""
    p_a_over_b: float = Field(
        ge=0.0,
        le=1.0,
        description="P(A > B | transcript)"
    )


class TogetherChat:
    """Wrapper for Together Chat API with JSON mode support."""

    def __init__(
        self,
        model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Use Mixtral which supports JSON mode
        max_tokens: int = 256,
        temperature: float = 0.1
    ):
        self.client = Together(api_key=get_api_key())
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(
        self,
        system: str,
        user: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 5
    ) -> str:
        """Send a chat completion request with retry logic."""
        # Add JSON instruction if using JSON mode
        if response_format:
            system = system + "\nOnly respond in valid JSON matching the provided schema."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)

                if not response or not response.choices:
                    print(f"DEBUG: No response or choices. Response: {response}")
                    return ""

                content = response.choices[0].message.content
                if content is None:
                    print(f"DEBUG: Content is None. Response: {response}")
                    return ""

                return content.strip()
            except together.error.ServiceUnavailableError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                    print(f"API overloaded (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error in Together API call after {max_retries} attempts: {e}")
                    print(f"DEBUG: kwargs were: {kwargs}")
                    raise
            except Exception as e:
                print(f"Error in Together API call: {e}")
                print(f"DEBUG: kwargs were: {kwargs}")
                raise

    def get_ab_preference(
        self,
        persona: str,
        option_a: str,
        option_b: str
    ) -> int:
        """Get A/B preference as 0 or 1 (1 if A preferred)."""
        system = (
            f"You are simulating a participant with the following persona:\n"
            f"{persona}\n"
            f"Answer consistently with this persona across the entire episode. Be decisive.\n"
            f"Always respond with valid JSON."
        )

        # Simpler prompt that works better with JSON mode
        user = (
            f"Based on the persona, choose between option A or B:\n\n"
            f"A: {option_a}\n"
            f"B: {option_b}\n\n"
            f'Return JSON in this exact format: {{"choice": "A"}} or {{"choice": "B"}}'
        )

        # Use simple json_object mode without schema
        response_format = {"type": "json_object"}

        response = self.chat(
            system=system,
            user=user,
            response_format=response_format,
            max_tokens=50
        )

        try:
            result = json.loads(response)
            # Try different keys the model might use
            if "choice" in result:
                return 1 if result["choice"] == "A" else 0
            elif "label" in result:
                return 1 if result["label"] == "A" else 0
            elif "answer" in result:
                return 1 if str(result["answer"]).upper().startswith("A") else 0
            else:
                # Look for A or B in any value
                for value in result.values():
                    if str(value).upper() == "A":
                        return 1
                    elif str(value).upper() == "B":
                        return 0
            # If still not found, check the raw response
            if "A" in response and "B" not in response:
                return 1
            elif "B" in response and "A" not in response:
                return 0
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing preference response: {e}")
            print(f"Response: {response[:100]}")

        # Default to random choice on error
        import random
        return random.choice([0, 1])

    def get_eval_probability(
        self,
        answers: str,
        option_a: str,
        option_b: str
    ) -> float:
        """Get evaluator probability P(A > B | transcript)."""
        system = (
            "You are a calibrated evaluator. You will be given a set of the participant's "
            "previous answers (no questions). From these, infer the probability the persona "
            "prefers option A over option B. Always respond with valid JSON."
        )

        user = (
            f"Transcript (participant answers only):\n"
            f"{answers}\n\n"
            f"Option A: {option_a}\n"
            f"Option B: {option_b}\n\n"
            f'Return the probability P(A>B | transcript) as JSON: {{"probability": 0.X}}'
        )

        # Use simple json_object mode
        response_format = {"type": "json_object"}

        response = self.chat(
            system=system,
            user=user,
            response_format=response_format,
            max_tokens=50
        )

        try:
            result = json.loads(response)
            # Try different keys
            if "probability" in result:
                p = float(result["probability"])
            elif "p_a_over_b" in result:
                p = float(result["p_a_over_b"])
            elif "prob" in result:
                p = float(result["prob"])
            else:
                # Try to find any numeric value
                for value in result.values():
                    try:
                        p = float(value)
                        if 0 <= p <= 1:
                            break
                    except (ValueError, TypeError):
                        continue
                else:
                    p = 0.5  # Default if no valid probability found

            # Clip to avoid log(0)
            return min(1.0 - 1e-3, max(1e-3, p))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing probability response: {e}")
            print(f"Response: {response[:100]}")
            # Default to uniform on error
            return 0.5

    def get_eval_probability_logprobs(self, answers: str, option_a: str, option_b: str) -> float:
        """Get evaluator probability using logprobs (more efficient and accurate).
        Falls back gracefully if the Together API response lacks top_logprobs.
        """
        import math

        def _clip(p: float) -> float:
            return min(0.999, max(0.001, float(p)))

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

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                logprobs=5
            )
        except Exception as e:
            print(f"Error requesting logprobs: {e}")
            return self.get_eval_probability(answers, option_a, option_b)

        if not response or not response.choices:
            return 0.5

        choice = response.choices[0]
        if not getattr(choice, "logprobs", None):
            # Model/endpoint doesn't support logprobs
            return self.get_eval_probability(answers, option_a, option_b)

        logprob_a = None
        logprob_b = None

        # Extract first-token logprobs robustly
        top0 = None
        if hasattr(choice.logprobs, "top_logprobs") and choice.logprobs.top_logprobs:
            top0 = choice.logprobs.top_logprobs[0]

        # Normalize to list of (token, logprob) pairs
        items = []
        if isinstance(top0, dict):
            items = list(top0.items())
        elif isinstance(top0, list):
            for entry in top0:
                if isinstance(entry, dict):
                    tok = entry.get("token", "")
                    lp = entry.get("logprob")
                    if tok and lp is not None:
                        items.append((tok, lp))

        for token, logprob in items:
            t = (token or "").strip().upper()
            if t == "A":
                logprob_a = logprob
            elif t == "B":
                logprob_b = logprob

        # Handle missing values
        if logprob_a is None and logprob_b is None:
            return self.get_eval_probability(answers, option_a, option_b)
        if logprob_a is None:
            return _clip(0.001)
        if logprob_b is None:
            return _clip(0.999)

        prob_a = math.exp(logprob_a)
        prob_b = math.exp(logprob_b)
        p_a_over_b = prob_a / (prob_a + prob_b + 1e-12)
        return _clip(p_a_over_b)


    def answer_question(
        self,
        persona: str,
        question: str
    ) -> str:
        """Get PLLM answer to a question."""
        system = (
            f"You are a human participant with this persona:\n"
            f"{persona}\n\n"
            f"ACT LIKE A HUMAN. Answer naturally and conversationally. Be authentic and stay in character."
        )

        user = (
            f"Question: {question}\n\n"
            f"Answer in ONLY about 20 words. Be conversational like a real person talking to someone. "
            f"Don't over-explain - just answer directly and naturally."
        )

        response = self.chat(
            system=system,
            user=user,
            temperature=0.7,  # Slightly higher for more natural answers
            max_tokens=50  # Reduced to enforce brevity (~20 words)
        )

        return response