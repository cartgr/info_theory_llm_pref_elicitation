"""Generate domain-specific evaluation items."""

from typing import List, Dict
import random
from .together_client import TogetherChat


def get_car_items() -> List[str]:
    """Get predefined car items for evaluation."""
    return [
        "Honda Civic Si (2020, manual)",
        "Used BMW M3 (2011, high mileage)",
        "Mazda MX-5 Miata (2019)",
        "Ford Mustang EcoBoost (2021)",
        "Subaru WRX (2020)",
        "Volkswagen GTI (2021)",
        "Used Porsche Boxster (2008)",
        "Hyundai Veloster N (2022)",
        "Toyota GR86 (2022)",
        "Used Nissan 370Z (2015)"
    ]


def get_restaurant_items() -> List[str]:
    """Get predefined restaurant items."""
    return [
        "Thai vegetarian restaurant with spicy curries",
        "Italian pizzeria with wood-fired oven",
        "Indian street food with chaat and dosas",
        "Mexican taqueria with veggie options",
        "Mediterranean grill with falafel and hummus",
        "Japanese ramen shop with vegetable broth",
        "Ethiopian restaurant with injera and lentils",
        "Vietnamese pho place with tofu options",
        "Greek taverna with grilled vegetables",
        "Chinese Sichuan restaurant with mapo tofu"
    ]


def get_movie_items() -> List[str]:
    """Get predefined movie items."""
    return [
        "Blade Runner 2049 (sci-fi, dystopian)",
        "Mad Max: Fury Road (action, post-apocalyptic)",
        "The Matrix (sci-fi, action)",
        "Inception (sci-fi, thriller)",
        "Interstellar (sci-fi, space)",
        "John Wick (action, revenge)",
        "Edge of Tomorrow (sci-fi, time loop)",
        "The Martian (sci-fi, survival)",
        "Mission Impossible: Fallout (action, spy)",
        "Arrival (sci-fi, alien contact)"
    ]


def generate_items_for_domain(
    domain: str,
    persona: str,
    num_items: int,
    chat_client: TogetherChat
) -> List[str]:
    """Generate items for a specific domain using PLLM."""
    # Use predefined items for consistency in MVE
    if domain == "cars":
        items = get_car_items()
    elif domain == "restaurants":
        items = get_restaurant_items()
    elif domain == "movies":
        items = get_movie_items()
    else:
        # Generate items dynamically for other domains
        system = f"You are helping generate items for preference learning experiments."
        user = (
            f"Generate {num_items} specific items in the domain of '{domain}' "
            f"that would be relevant for someone with the persona: {persona}. "
            f"Return a simple list, one item per line. Be specific and detailed."
        )
        response = chat_client.chat(system=system, user=user, max_tokens=512)
        items = [line.strip() for line in response.strip().split('\n') if line.strip()]
        items = [item.lstrip('- •123456789.') for item in items]  # Clean up list formatting
        items = items[:num_items]

    # Ensure we have exactly num_items
    if len(items) < num_items:
        items.extend([f"{domain} item {i}" for i in range(len(items), num_items)])
    elif len(items) > num_items:
        items = items[:num_items]

    return items


def generate_item_pairs(
    num_items: int,
    num_pairs: int,
    seed: int = None
) -> List[tuple]:
    """Generate pairs of item indices for comparison."""
    if seed is not None:
        random.seed(seed)

    all_pairs = [(i, j) for i in range(num_items) for j in range(i + 1, num_items)]

    if num_pairs >= len(all_pairs):
        return all_pairs

    # Sample without replacement
    selected_pairs = random.sample(all_pairs, num_pairs)
    return selected_pairs