# Walu Season

- Practice reef: **6,973 kg/day** (threshold 5,300), worst day 6,217, 6 s. `python evaluation_script.py main.py --dataset train`.
- On 12 made-up reefs (same rules, different spots: one strong school, rare big schools, two close schools) it gets 89.7-94.6% of what fishing only the best spot would give.
- Strategy: UCB with a variance bonus. The variance has a floor that fades with casts, so a walu spot isn't written off after a few empty casts.
- No randomness, so the score is the same on every run. Submit `main.py`.
