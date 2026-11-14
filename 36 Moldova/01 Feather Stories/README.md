# Feather Stories

Flag: `SIGMOID_XWBPHD` (next best: `SIGMOID_ANSEMO`)

`python feather_stories.py` — output in [run_output.txt](run_output.txt). Longer walk-through in [feather_stories.ipynb](feather_stories.ipynb).

- Misreads only happen inside 6 groups of look-alike letters: AKRX, BGSZ, CDOQ, EFLP, HMNW, IJTY (0 of 14,467 letters cross a group).
- Inside a group a misread is always one of 3 fixed swaps, and which swap you get depends on the site, not on the bird, observer or year (the same ring ADHIOL is read fine at one site and swapped at another).
- Sites are either clear (true letter wins, double misreads 6/4,171) or swapped (~85% of letters go through one swap).
- The only reads that match no registered bird: 24 White Stork reads, all at L5819951 (Lower Prut). Majority read RHGLWQ.
- Those reads have 5 double misreads out of 121 letters, p = 1.6e-4 for a clear site, so RHGLWQ is the swapped picture of the ring, not the ring.
- Which swap: over the 109 registered sites, the swap that hits both halves of the alphabet the same way is type b at 18 sites and type a at 4. Undoing b gives XWBPHD (0.63), undoing a gives ANSEMO (0.37).
