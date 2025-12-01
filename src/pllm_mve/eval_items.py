"""Generate domain-specific evaluation items."""

from typing import List, Dict
import random
from .together_client import TogetherChat


def get_car_items() -> List[str]:
    """Get predefined car items for evaluation (expanded set)."""
    return [
        # Budget performance (under $30k)
        "Honda Civic Si (2020, manual)",
        "Mazda MX-5 Miata (2019)",
        "Ford Mustang EcoBoost (2021)",
        "Subaru WRX (2020)",
        "Volkswagen GTI (2021)",
        "Hyundai Veloster N (2022)",
        "Toyota GR86 (2022)",
        "Used Nissan 370Z (2015)",
        "Honda Accord Sport (2022)",
        "Mazda3 Turbo (2021)",

        # Used performance/luxury
        "Used BMW M3 (2011, high mileage)",
        "Used Porsche Boxster (2008)",
        "Used Audi S4 (2013)",
        "Used Lexus IS 350 (2016)",
        "Used Mercedes C43 AMG (2017)",
        "Used Corvette C6 (2010)",

        # New luxury (over $40k)
        "BMW 330i (2023)",
        "Mercedes-Benz C-Class (2023)",
        "Audi A4 (2023)",
        "Lexus ES 350 (2023)",
        "Genesis G70 (2023)",

        # SUVs - Budget
        "Honda CR-V (2022)",
        "Mazda CX-5 (2023)",
        "Toyota RAV4 (2022)",
        "Subaru Forester (2023)",
        "Hyundai Tucson (2023)",

        # SUVs - Luxury
        "BMW X3 (2023)",
        "Audi Q5 (2023)",
        "Lexus RX 350 (2023)",
        "Mercedes-Benz GLC (2023)",
        "Acura RDX (2023)",

        # Electric/Hybrid
        "Tesla Model 3 (2023)",
        "Chevrolet Bolt EV (2023)",
        "Hyundai Ioniq 5 (2023)",
        "Nissan Leaf (2023)",
        "Toyota Prius (2023)",
        "Honda Accord Hybrid (2023)",
        "Kia Niro EV (2023)",

        # Practical/Commuter
        "Toyota Corolla (2023)",
        "Honda Civic (2023)",
        "Mazda3 (2022)",
        "Hyundai Elantra (2023)",
        "Subaru Impreza (2023)",

        # Trucks
        "Ford F-150 (2023)",
        "Toyota Tacoma (2023)",
        "Chevrolet Silverado (2023)",
        "Ram 1500 (2023)",
        "Honda Ridgeline (2023)",

        # Minivans
        "Honda Odyssey (2023)",
        "Toyota Sienna Hybrid (2023)",
        "Chrysler Pacifica (2023)",

        # Specialty
        "Subaru Outback (2023)",
        "Jeep Wrangler (2023)",
        "Ford Bronco (2023)",
        "Mini Cooper S (2023)",
        "Fiat 500 Abarth (2022)",
    ]


def get_restaurant_items() -> List[str]:
    """Get predefined restaurant items (expanded set)."""
    return [
        # Italian
        "Upscale Italian trattoria, handmade pasta, $$$",
        "Casual Neapolitan pizza, wood-fired, $$",
        "Italian-American red sauce joint, family-style, $$",
        "Modern Italian small plates, wine bar, $$$",
        "Quick Italian deli, sandwiches and salads, $",

        # Asian - Japanese
        "Omakase sushi bar, 12-course, $$$$",
        "Casual ramen shop, pork tonkotsu, $$",
        "Japanese izakaya, small plates and sake, $$",
        "Fast-casual poke bowl, build-your-own, $",
        "Hibachi grill, tableside cooking, $$$",

        # Asian - Chinese
        "Dim sum palace, weekend carts, $$",
        "Sichuan restaurant, very spicy, $$",
        "Chinese-American takeout, General Tso's, $",
        "Upscale Cantonese, Peking duck, $$$",
        "Hand-pulled noodle shop, casual, $",

        # Asian - Other
        "Thai restaurant, curries and pad thai, $$",
        "Vietnamese pho house, big bowls, $",
        "Korean BBQ, tabletop grilling, $$$",
        "Indian curry house, buffet lunch, $$",
        "Filipino restaurant, adobo and lumpia, $$",

        # Mexican/Latin
        "Taqueria, street tacos, cash only, $",
        "Upscale Mexican, mole and mezcal, $$$",
        "Tex-Mex cantina, fajitas and margaritas, $$",
        "Peruvian rotisserie chicken, casual, $$",
        "Cuban cafe, sandwiches and coffee, $",

        # American
        "Classic steakhouse, dry-aged beef, $$$$",
        "Craft burger joint, local beef, $$",
        "Southern comfort food, fried chicken, $$",
        "New American farm-to-table, seasonal menu, $$$",
        "Classic diner, breakfast all day, $",
        "BBQ smokehouse, brisket and ribs, $$",
        "Seafood shack, lobster rolls, $$",

        # European
        "French bistro, steak frites, $$$",
        "Spanish tapas bar, sangria, $$",
        "Greek taverna, lamb and mezze, $$",
        "German beer hall, schnitzel and pretzels, $$",
        "British gastropub, fish and chips, $$",

        # Middle Eastern/Mediterranean
        "Lebanese restaurant, shawarma and mezze, $$",
        "Falafel counter, fast-casual, $",
        "Turkish kebab house, grilled meats, $$",
        "Israeli hummus bar, fresh pita, $",
        "Moroccan restaurant, tagine and couscous, $$$",

        # Vegetarian/Health
        "Vegan cafe, plant-based bowls, $$",
        "Farm-to-table vegetarian, seasonal, $$$",
        "Juice bar and salads, healthy fast-casual, $",
        "Raw vegan restaurant, organic, $$$",

        # Brunch/Breakfast
        "Trendy brunch spot, avocado toast, $$",
        "Classic breakfast diner, pancakes, $",
        "French bakery cafe, croissants, $$",

        # Fast Food/Quick
        "Chipotle-style burrito chain, $",
        "Fried chicken sandwich chain, $",
        "Shake Shack-style burger, $$",
        "Sweetgreen-style salad bar, $$",

        # Special Occasion
        "Chef's tasting menu, fine dining, $$$$",
        "Rooftop restaurant, views and cocktails, $$$",
        "Historic landmark restaurant, classic, $$$",
        "Celebrity chef restaurant, trendy, $$$$",
    ]


def get_movie_items() -> List[str]:
    """Get predefined movie items (expanded set)."""
    return [
        # Sci-fi
        "Blade Runner 2049 (sci-fi, dystopian)",
        "The Matrix (sci-fi, action)",
        "Inception (sci-fi, thriller)",
        "Interstellar (sci-fi, space)",
        "Arrival (sci-fi, alien contact)",
        "Dune (2021) (sci-fi, epic)",
        "Ex Machina (sci-fi, AI)",
        "Her (sci-fi, romance)",
        "Annihilation (sci-fi, horror)",
        "The Prestige (sci-fi, mystery)",

        # Action/Thriller
        "Mad Max: Fury Road (action, post-apocalyptic)",
        "John Wick (action, revenge)",
        "Mission Impossible: Fallout (action, spy)",
        "Edge of Tomorrow (sci-fi, time loop)",
        "The Dark Knight (action, superhero)",
        "Heat (action, crime)",
        "Sicario (thriller, crime)",
        "No Country for Old Men (thriller, western)",
        "Drive (action, noir)",
        "The Raid (action, martial arts)",

        # Drama
        "The Shawshank Redemption (drama, prison)",
        "Schindler's List (drama, historical)",
        "Parasite (drama, thriller)",
        "Whiplash (drama, music)",
        "The Social Network (drama, biography)",
        "There Will Be Blood (drama, period)",
        "Manchester by the Sea (drama, family)",
        "Moonlight (drama, coming-of-age)",
        "12 Years a Slave (drama, historical)",
        "The Godfather (drama, crime)",

        # Comedy
        "The Grand Budapest Hotel (comedy, quirky)",
        "Superbad (comedy, teen)",
        "The Big Lebowski (comedy, cult)",
        "In Bruges (comedy, dark)",
        "Hot Fuzz (comedy, action)",
        "Step Brothers (comedy, absurd)",
        "Bridesmaids (comedy, ensemble)",
        "What We Do in the Shadows (comedy, horror)",
        "The Nice Guys (comedy, noir)",
        "Knives Out (comedy, mystery)",

        # Horror
        "Get Out (horror, social)",
        "Hereditary (horror, supernatural)",
        "The Witch (horror, period)",
        "A Quiet Place (horror, survival)",
        "Midsommar (horror, folk)",
        "It Follows (horror, supernatural)",
        "The Conjuring (horror, supernatural)",
        "28 Days Later (horror, zombie)",
        "Pan's Labyrinth (horror, fantasy)",
        "The Babadook (horror, psychological)",

        # Romance/Rom-com
        "Eternal Sunshine of the Spotless Mind (romance, sci-fi)",
        "Before Sunrise (romance, dialogue)",
        "La La Land (romance, musical)",
        "Pride and Prejudice (2005) (romance, period)",
        "Crazy Rich Asians (romance, comedy)",
        "The Notebook (romance, drama)",
        "500 Days of Summer (romance, indie)",
        "When Harry Met Sally (romance, comedy)",
        "About Time (romance, sci-fi)",
        "Silver Linings Playbook (romance, drama)",

        # Animation
        "Spider-Man: Into the Spider-Verse (animation, superhero)",
        "Spirited Away (animation, fantasy)",
        "The Incredibles (animation, superhero)",
        "Coco (animation, family)",
        "Wall-E (animation, sci-fi)",
        "Ratatouille (animation, comedy)",
        "Akira (animation, sci-fi)",
        "Princess Mononoke (animation, fantasy)",
        "Toy Story (animation, family)",
        "How to Train Your Dragon (animation, adventure)",

        # Documentary
        "Free Solo (documentary, adventure)",
        "Won't You Be My Neighbor? (documentary, biography)",
        "The Act of Killing (documentary, crime)",
        "Jiro Dreams of Sushi (documentary, food)",
        "13th (documentary, social)",
    ]


def generate_items_for_domain(
    domain: str,
    persona: str,
    num_items: int,
    chat_client: TogetherChat
) -> List[str]:
    """Generate items for a specific domain using PLLM.

    Items are randomly sampled from the predefined pool (using the current
    random state, so set random.seed() before calling for reproducibility).
    """
    # Use predefined items for consistency in MVE
    if domain == "cars":
        pool = get_car_items()
    elif domain == "restaurants":
        pool = get_restaurant_items()
    elif domain == "movies":
        pool = get_movie_items()
    else:
        # Generate items dynamically for other domains
        system = f"You are helping generate items for preference learning experiments."
        user = (
            f"Generate {num_items} specific items in the domain of '{domain}' "
            f"that would be relevant for someone with the persona: {persona}. "
            f"Return a simple list, one item per line. Be specific and detailed."
        )
        response = chat_client.chat(system=system, user=user, max_tokens=512)
        pool = [line.strip() for line in response.strip().split('\n') if line.strip()]
        pool = [item.lstrip('- •123456789.') for item in pool]  # Clean up list formatting

    # Randomly sample num_items from the pool
    if len(pool) <= num_items:
        items = pool
    else:
        items = random.sample(pool, num_items)

    # Ensure we have exactly num_items (fallback for small pools)
    if len(items) < num_items:
        items.extend([f"{domain} item {i}" for i in range(len(items), num_items)])

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