"""Persona management for PLLM experiments."""

from typing import List, Dict


def get_default_personas() -> Dict[str, str]:
    """Get default personas for different domains."""
    return {
        "cars": "likes fast cars but is on a budget",
        "restaurants": "vegetarian who enjoys spicy food and casual dining",
        "movies": "enjoys sci-fi and action movies but dislikes horror",
        "travel": "prefers adventure travel to beach resorts, budget-conscious",
        "tech": "early adopter who values open-source and privacy",
        "fashion": "minimalist style, prefers comfort over trends",
        "music": "enjoys indie rock and jazz, dislikes heavy metal",
        "books": "loves mystery novels and historical non-fiction"
    }


def get_persona_for_domain(domain: str) -> str:
    """Get the appropriate persona for a domain."""
    personas = get_default_personas()
    return personas.get(domain, personas["cars"])  # Default to cars


def format_persona_prompt(persona: str) -> str:
    """Format persona for system prompt."""
    return (
        f"You are simulating a participant with the following persona:\n"
        f"{persona}\n"
        f"Answer consistently with this persona across the entire episode. Be decisive."
    )