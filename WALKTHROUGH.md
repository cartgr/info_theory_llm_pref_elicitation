# Best-of-N Experiment: Detailed Walkthrough

This document walks through the entire experiment step-by-step with actual prompts.

---

## PHASE 1: SETUP

### Step 1.1: Load Config

From `experiments/mve_one_persona.yml`:

```yaml
episode:
  persona: |
    Marcus, a 34-year-old software engineer living in Austin, Texas...
  domain: cars
  num_items: 15
  num_pairs: 25
  num_turns: 5
  model: mistralai/Mixtral-8x7B-Instruct-v0.1
```

### Step 1.2: Generate Items

**Code:** `eval_items.py:generate_items_for_domain()`

For domain="cars", we use predefined items (no LLM call). The first 15 items from `get_car_items()`:

```
items = [
  0: "Honda Civic Si (2020, manual)"
  1: "Mazda MX-5 Miata (2019)"
  2: "Ford Mustang EcoBoost (2021)"
  3: "Subaru WRX (2020)"
  4: "Volkswagen GTI (2021)"
  5: "Hyundai Veloster N (2022)"
  6: "Toyota GR86 (2022)"
  7: "Used Nissan 370Z (2015)"
  8: "Honda Accord Sport (2022)"
  9: "Mazda3 Turbo (2021)"
  10: "Used BMW M3 (2011, high mileage)"
  11: "Used Porsche Boxster (2008)"
  12: "Used Audi S4 (2013)"
  13: "Used Lexus IS 350 (2016)"
  14: "Used Mercedes C43 AMG (2017)"
]
```

### Step 1.3: Create Pair Sets E and T

**Code:** `run_bestofn_experiment.py:setup_episode()`

```python
# T = ALL pairs (for final evaluation)
all_pairs = [(i, j) for i in range(15) for j in range(i + 1, 15)]
# T has 15*14/2 = 105 pairs

# E = SUBSET of pairs (for rewards during episode)
pairs_E = random.sample(all_pairs, 25)  # e.g., 25 pairs
```

Example E (25 pairs):
```
pairs_E = [
  (3, 14),   # Subaru WRX vs Used Mercedes C43
  (10, 13),  # Used BMW M3 vs Used Lexus IS 350
  (4, 8),    # VW GTI vs Honda Accord Sport
  (0, 6),    # Honda Civic Si vs Toyota GR86
  ... (21 more pairs)
]
```

---

## PHASE 2: LABEL ALL PAIRS WITH PLLM

**Code:** `pllm.py:label_eval_question()` → `together_client.py:get_ab_preference()`

For each pair in E and T, we ask the PLLM (simulating Marcus) which item he prefers.

### Prompt for pair (0, 6): Honda Civic Si vs Toyota GR86

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM:                                                         │
├─────────────────────────────────────────────────────────────────┤
│ You are simulating a participant with the following persona:    │
│ Marcus, a 34-year-old software engineer living in Austin,       │
│ Texas. He commutes 35 minutes each way on highways and city     │
│ streets. Married with a 2-year-old daughter, so car seat        │
│ compatibility and safety ratings are non-negotiable.            │
│                                                                 │
│ Budget is firm at $38,000 max - he's been saving for two years  │
│ and won't finance more than that. Previously owned a 2015 Honda │
│ Accord that lasted 180,000 miles, which shaped his trust in     │
│ Japanese reliability.                                           │
│                                                                 │
│ Priorities in order: safety ratings (must be 5-star NHTSA),     │
│ reliability (wants to keep the car 10+ years), fuel efficiency  │
│ (targets 30+ mpg combined), then comfort and tech features.     │
│                                                                 │
│ Dislikes: overly aggressive styling, massive infotainment       │
│ screens he finds distracting, anything that feels "cheap"       │
│ inside. Test drove a Nissan recently and hated the CVT          │
│ transmission feel.                                              │
│                                                                 │
│ Considering: Would go electric if charging was more convenient  │
│ in his apartment complex, but currently leaning toward hybrid   │
│ or efficient gas. Wants Apple CarPlay. Likes the look of wagons │
│ and hatchbacks but wife prefers SUV styling.                    │
│                                                                 │
│ Answer consistently with this persona across the entire         │
│ episode. Be decisive.                                           │
│ Always respond with valid JSON.                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ USER:                                                           │
├─────────────────────────────────────────────────────────────────┤
│ Based on the persona, choose between option A or B:             │
│                                                                 │
│ A: Honda Civic Si (2020, manual)                                │
│ B: Toyota GR86 (2022)                                           │
│                                                                 │
│ Return JSON in this exact format: {"choice": "A"} or            │
│ {"choice": "B"}                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RESPONSE:                                                       │
├─────────────────────────────────────────────────────────────────┤
│ {"choice": "A"}                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Result:** `labels[(0, 6)] = 1` (A > B, meaning Honda Civic Si preferred over Toyota GR86)

This makes sense for Marcus: Civic Si is more practical, better fuel economy, Honda reliability.

**This is repeated for all 25 pairs in E and all 105 pairs in T.**

---

## PHASE 3: INITIALIZE COMPONENTS

### 3.1 EvaluatorD (Judge)

**Code:** `evaluator.py:EvaluatorD()`

The evaluator predicts P(A > B | transcript). It does NOT know the persona - it only sees the conversation answers.

### 3.2 ResponderLLM (Proposal Model)

**Code:** `responder.py:ResponderLLM()`

Generates hypothetical answers for candidate evaluation. Does NOT have the persona - generates diverse answers from generic users.

---

## PHASE 4: INITIAL BASELINE SCORE

**Code:** `scoring.py:log_score()`

Before any conversation, we score how well the evaluator can predict preferences with NO information.

For each pair (i, j) in E:

### Evaluator Prompt (empty transcript)

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM:                                                         │
├─────────────────────────────────────────────────────────────────┤
│ You are a calibrated evaluator. You will be given a set of the  │
│ participant's previous answers (no questions). From these,      │
│ infer the probability the persona prefers option A over option  │
│ B. Always respond with valid JSON.                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ USER:                                                           │
├─────────────────────────────────────────────────────────────────┤
│ Transcript (participant answers only):                          │
│ No answers yet.                                                 │
│                                                                 │
│ Option A: Honda Civic Si (2020, manual)                         │
│ Option B: Toyota GR86 (2022)                                    │
│                                                                 │
│ Return the probability P(A>B | transcript) as JSON:             │
│ {"probability": 0.X}                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RESPONSE:                                                       │
├─────────────────────────────────────────────────────────────────┤
│ {"probability": 0.5}                                            │
└─────────────────────────────────────────────────────────────────┘
```

**Result:** P(A > B) = 0.5 (maximum uncertainty - no information)

### Log Score Calculation

For pair (0, 6) where true label = 1 (A > B):
```
P(correct) = P(A > B) = 0.5
log(P(correct)) = log(0.5) = -0.693
```

For pair (4, 8) where true label = 0 (B > A):
```
P(correct) = 1 - P(A > B) = 1 - 0.5 = 0.5
log(P(correct)) = log(0.5) = -0.693
```

**Initial average log score = -0.693** (random guessing)

---

## PHASE 5: BEST-OF-N EPISODE (Turn 1)

### Step 5.1: Generate Candidate Questions

**Code:** `qllm_policy.py:generate_candidate_questions()`

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM:                                                         │
├─────────────────────────────────────────────────────────────────┤
│ You are an expert interviewer trying to learn someone's         │
│ preferences about cars. Generate informative questions that     │
│ will help uncover their preferences.                            │
│                                                                 │
│ The items being considered are:                                 │
│ - Honda Civic Si (2020, manual)                                 │
│ - Mazda MX-5 Miata (2019)                                       │
│ - Ford Mustang EcoBoost (2021)                                  │
│ - Subaru WRX (2020)                                             │
│ - Volkswagen GTI (2021)                                         │
│ - Hyundai Veloster N (2022)                                     │
│ - Toyota GR86 (2022)                                            │
│ - Used Nissan 370Z (2015)                                       │
│ - Honda Accord Sport (2022)                                     │
│ - Mazda3 Turbo (2021)                                           │
│ ...and 5 more                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ USER:                                                           │
├─────────────────────────────────────────────────────────────────┤
│ Generate 5 diverse questions to ask next. Each question should: │
│ - Be specific and targeted                                      │
│ - Help reveal preferences about cars                            │
│ - Be natural and conversational                                 │
│ - Be different from the others (diverse approaches)             │
│                                                                 │
│ Return ONLY the questions, one per line, numbered 1-5.          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RESPONSE:                                                       │
├─────────────────────────────────────────────────────────────────┤
│ 1. What's your primary use for this car - daily commuting,      │
│    weekend fun, or a mix of both?                               │
│ 2. How important is fuel efficiency versus performance to you?  │
│ 3. Do you have a preference for manual or automatic             │
│    transmission?                                                │
│ 4. What's your budget range for this purchase?                  │
│ 5. Are you open to buying a used vehicle, or do you prefer new? │
└─────────────────────────────────────────────────────────────────┘
```

**Result:** 5 candidate questions (k=5)

### Step 5.2: Evaluate Each Candidate

For each candidate question, we sample hypothetical answers and score them.

#### Candidate 1: "What's your primary use for this car..."

**Step 5.2.1: Sample Hypothetical Answers (batched)**

**Code:** `responder.py:sample_answers()`

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM:                                                         │
├─────────────────────────────────────────────────────────────────┤
│ You are simulating 3 different people being interviewed about   │
│ cars. Each person has different preferences and backgrounds.    │
│ Generate 3 distinct, realistic answers to the question below.   │
│                                                                 │
│ Format your response EXACTLY like this:                         │
│ [1] First person's answer here                                  │
│ [2] Second person's answer here                                 │
│ ...                                                             │
│                                                                 │
│ Each answer should be ~20 words, natural and conversational.    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ USER:                                                           │
├─────────────────────────────────────────────────────────────────┤
│ Question: What's your primary use for this car - daily          │
│ commuting, weekend fun, or a mix of both?                       │
│                                                                 │
│ Generate 3 different answers:                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RESPONSE:                                                       │
├─────────────────────────────────────────────────────────────────┤
│ [1] Mostly daily commuting, about 30 miles each way. I need     │
│     something reliable and fuel-efficient for the highway.      │
│ [2] Weekend fun for sure! I want something sporty that's        │
│     exciting to drive on mountain roads.                        │
│ [3] A mix of both - I commute during the week but love taking   │
│     road trips on weekends. Need versatility.                   │
└─────────────────────────────────────────────────────────────────┘
```

**Result:** 3 hypothetical answers (t=3)

**Step 5.2.2: Score Each Trial Transcript**

For sample [1] "Mostly daily commuting...":

Create trial transcript:
```
Q1: What's your primary use for this car - daily commuting, weekend fun, or a mix of both?
A1: Mostly daily commuting, about 30 miles each way. I need something reliable and fuel-efficient for the highway.
```

Now evaluate this trial on all 25 pairs in E:

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM:                                                         │
├─────────────────────────────────────────────────────────────────┤
│ You are a calibrated evaluator. You will be given a set of the  │
│ participant's previous answers (no questions). From these,      │
│ infer the probability the persona prefers option A over option  │
│ B. Always respond with valid JSON.                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ USER:                                                           │
├─────────────────────────────────────────────────────────────────┤
│ Transcript (participant answers only):                          │
│ - Mostly daily commuting, about 30 miles each way. I need       │
│   something reliable and fuel-efficient for the highway.        │
│                                                                 │
│ Option A: Honda Civic Si (2020, manual)                         │
│ Option B: Toyota GR86 (2022)                                    │
│                                                                 │
│ Return the probability P(A>B | transcript) as JSON:             │
│ {"probability": 0.X}                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RESPONSE:                                                       │
├─────────────────────────────────────────────────────────────────┤
│ {"probability": 0.75}                                           │
└─────────────────────────────────────────────────────────────────┘
```

The evaluator now predicts P(Civic > GR86) = 0.75 because:
- "daily commuting" → practical car
- "fuel-efficient" → Civic is better
- "reliable" → Honda reputation

**This evaluation is done for all 25 pairs in E.**

**Step 5.2.3: Compute Log Score and Gain**

```
Trial score = average of log(P(correct)) over all 25 pairs
            ≈ -0.45 (better than -0.693 baseline)

Gain = trial_score - baseline_score
     = -0.45 - (-0.693)
     = +0.243
```

**Repeat for samples [2] and [3], get gains like [+0.243, +0.15, +0.18]**

**Step 5.2.4: Aggregate Gains**

```
max_gain for candidate 1 = max([0.243, 0.15, 0.18]) = 0.243
```

### Step 5.3: Repeat for All Candidates

```
Candidate 1 (primary use):        max_gain = 0.243
Candidate 2 (fuel vs performance): max_gain = 0.198
Candidate 3 (transmission):        max_gain = 0.156
Candidate 4 (budget):              max_gain = 0.312  ← BEST
Candidate 5 (new vs used):         max_gain = 0.089
```

### Step 5.4: Select Best Question

**Winner:** Candidate 4 "What's your budget range for this purchase?"

### Step 5.5: Ask PLLM for Real Answer

**Code:** `pllm.py:answer_question()` → `together_client.py:answer_question()`

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM:                                                         │
├─────────────────────────────────────────────────────────────────┤
│ You are a human participant with this persona:                  │
│ Marcus, a 34-year-old software engineer living in Austin,       │
│ Texas. He commutes 35 minutes each way on highways and city     │
│ streets. Married with a 2-year-old daughter, so car seat        │
│ compatibility and safety ratings are non-negotiable.            │
│ [... full persona ...]                                          │
│                                                                 │
│ ACT LIKE A HUMAN. Answer naturally and conversationally.        │
│ Be authentic and stay in character.                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ USER:                                                           │
├─────────────────────────────────────────────────────────────────┤
│ Question: What's your budget range for this purchase?           │
│                                                                 │
│ Answer in ONLY about 20 words. Be conversational like a real    │
│ person talking to someone. Don't over-explain - just answer     │
│ directly and naturally.                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RESPONSE:                                                       │
├─────────────────────────────────────────────────────────────────┤
│ I've got about $38,000 saved up. That's my firm ceiling -       │
│ been saving for two years and I'm sticking to it.               │
└─────────────────────────────────────────────────────────────────┘
```

### Step 5.6: Update Transcript and Score

**Real transcript after turn 1:**
```
Q1: What's your budget range for this purchase?
A1: I've got about $38,000 saved up. That's my firm ceiling - been saving for two years and I'm sticking to it.
```

**New score on E:** -0.52 (improved from -0.693)
**Reward:** -0.52 - (-0.693) = +0.173

---

## PHASE 6: TURNS 2-5

Repeat Phase 5 with updated transcript. Each turn:
1. Generate 5 candidate questions (now seeing previous Q&A)
2. For each candidate, sample 3 hypothetical answers
3. Score each trial on E
4. Select best question (max gain)
5. Ask PLLM, get real answer
6. Update transcript, compute reward

---

## PHASE 7: FINAL EVALUATION ON T

After 5 turns, evaluate the final transcript on ALL 105 pairs (T), not just E.

```
Final transcript:
Q1: What's your budget range for this purchase?
A1: I've got about $38,000 saved up...

Q2: How important is fuel efficiency versus performance?
A2: Fuel efficiency matters a lot for my commute...

Q3: Do you have any brand preferences?
A3: I trust Honda and Toyota - had great luck with my Accord...

Q4: What features are must-haves for you?
A4: Apple CarPlay is essential. And good safety ratings...

Q5: New or used - any preference?
A5: I'd consider used if it's low mileage and reliable...
```

**Final log score on T:** -0.38 (much better than -0.693 baseline)

---

## PHASE 8: DIRECT BASELINE COMPARISON

Run the same experiment but instead of selecting the best question, pick randomly from the 5 candidates.

**Direct baseline final score on T:** -0.55 (worse than best-of-n)

---

## SUMMARY

```
                    Best-of-N    Direct    Improvement
Initial score (E):    -0.693    -0.693         0
Final score (T):      -0.38     -0.55       +0.17
Total reward (E):     +0.45     +0.22       +0.23
```

Best-of-N outperforms random selection by choosing questions that maximize information gain about preferences.
