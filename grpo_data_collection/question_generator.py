"""Generate diverse candidate questions for GRPO training."""

from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pllm_mve.together_client import TogetherChat
from src.pllm_mve.types import Transcript


class QuestionGenerator:
    """Generate diverse questions for preference elicitation."""

    def __init__(self, chat_client: Optional[TogetherChat] = None):
        """Initialize with chat client."""
        self.chat = chat_client or TogetherChat(temperature=0.8)  # Higher temp for diversity

    def generate_initial_questions(
        self,
        beginning_prompt: str,
        domain: str,
        num_questions: int = 8
    ) -> List[str]:
        """Generate initial questions given a beginning prompt."""
        system = (
            f"You are interviewing someone to discover their preferences about {domain}. "
            f"Ask clear, specific questions to understand what they're looking for."
        )

        user = (
            f"Generate {num_questions} different questions to ask someone about their {domain} preferences.\n\n"
            f"Each question should:\n"
            f"- Be a single, direct question\n"
            f"- Explore different aspects (budget, features, use case, priorities, etc.)\n"
            f"- Be conversational and natural\n\n"
            f"Return ONLY the questions, one per line, no numbering."
        )

        response = self.chat.chat(
            system=system,
            user=user,
            temperature=0.8,
            max_tokens=512
        )

        # Parse questions from response
        questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line:
                # Remove common prefixes
                line = line.lstrip('- •123456789.')
                line = line.strip()
                if line and '?' in line:  # Basic validation
                    questions.append(line)

        # Ensure we have enough questions
        while len(questions) < num_questions:
            questions.append(f"What aspects of {domain} are most important to you?")

        return questions[:num_questions]

    def generate_followup_questions(
        self,
        transcript: Transcript,
        domain: str,
        num_questions: int = 8
    ) -> List[str]:
        """Generate follow-up questions based on transcript history."""
        # Format transcript for context
        context = ""
        for turn in transcript.turns:
            context += f"Q: {turn.question}\nA: {turn.answer}\n\n"

        system = (
            f"You are interviewing someone about their {domain} preferences. "
            f"Ask follow-up questions that dig deeper based on their previous answers."
        )

        user = (
            f"Previous conversation:\n{context}\n"
            f"Generate {num_questions} different follow-up questions based on what they said.\n\n"
            f"Each question should:\n"
            f"- Build on their previous answers\n"
            f"- Explore new aspects or dig deeper\n"
            f"- Be specific and conversational\n\n"
            f"Return ONLY the questions, one per line, no numbering."
        )

        response = self.chat.chat(
            system=system,
            user=user,
            temperature=0.8,
            max_tokens=512
        )

        # Parse questions
        questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line:
                line = line.lstrip('- •123456789.')
                line = line.strip()
                if line and '?' in line:
                    questions.append(line)

        # Add fallback questions if needed
        fallbacks = [
            f"What specific features in {domain} matter most to you?",
            f"How would you prioritize different aspects of {domain}?",
            f"What's your ideal scenario for using this?",
            f"Are there any deal-breakers for you?",
            f"What compromises would you be willing to make?"
        ]

        for fallback in fallbacks:
            if len(questions) >= num_questions:
                break
            if fallback not in questions:
                questions.append(fallback)

        return questions[:num_questions]

    def format_as_prompt(self, transcript: Transcript, domain: str) -> str:
        """Format transcript as a prompt for GRPO training."""
        if not transcript.turns:
            # Initial question prompt
            return (
                f"You are interviewing someone about their {domain} preferences. "
                f"Ask one specific question to learn what they're looking for.\n\n"
                f"Question:"
            )

        # Follow-up question prompt
        context = ""
        for turn in transcript.turns:
            context += f"Q: {turn.question}\nA: {turn.answer}\n"

        return (
            f"You are interviewing someone about their {domain} preferences. "
            f"Here's the conversation so far:\n\n{context}\n"
            f"Ask one follow-up question to learn more.\n\n"
            f"Question:"
        )