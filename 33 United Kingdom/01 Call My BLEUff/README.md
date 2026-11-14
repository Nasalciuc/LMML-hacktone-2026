**Flag: `SIGMOID_FRINDLOW`**

Honest seats by round: 1 2 3 | 2 3 3 | 1 3 3 | 2 2 2 | 1 2 1 | 2 1 3 | 2 2 3 | 3 2 2. The producers' BLEU winners spell SKELDRAP and are right in only 8 of 24 rounds.
Each round has one heavy bluffer and one light one. Heavy: 8 croppers (a reference copied word for word and cut short), 8 padders (reference sentences bolted on, facts said twice), 8 splicers (reference fragments stitched into non-sentences). Light: drops most facts on 3-7 cards, now and then a padded card or "more than more than".
All of it pays under this scorer: no smoothing, clipping per single reference, and a brevity penalty taken once per deck against each card's closest reference, so short, precise cards are cheap.
Metric: per card, four listener checks - missing facts (≥2 that most references agree on), padding, a cut-out of a reference, garbled text (trigram model on the other cards' references, repeated phrases). Deck score = cards failing any check.
The honest deck fails 0-1 cards in every round, the others 3-31; closest round is 15 (0 vs 3). Everything is in `call_my_bleuff.ipynb`; figures `separation.png`, `tricks_by_deck.png`.
