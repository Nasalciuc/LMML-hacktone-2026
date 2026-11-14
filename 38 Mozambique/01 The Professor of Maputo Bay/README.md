Arena (24 maps x 5, worst game per map): 0.27 / 0.23 / 0.31 on seeds 11/12/13. Not at the 65% bar yet.
Own 96-map check: 0.40 map score, 63% of single games won, avg 36 vs 29 points.
Slowest game 4.5 s thinking on CPython (budget capped at 6 s per game), no illegal actions.
Agent: beam search over own moves (horizon 14, width 24) on a local copy of the rules, scored by delivered points plus value of products, materials near the workshop and goods matched to raw contracts.
Workshop site picked at round 0 by comparing searches for both manufacturing sites.
Weak spot: many 34-34 draws and losses by one contract; worst-of-5 punishes that.
