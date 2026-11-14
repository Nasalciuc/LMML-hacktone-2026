# The Braided River Run

- Test valley: **293 minutes** (246 actions), under the 296 limit. Practice valley: 114, same as the reference.
- `python environment.py --map test-input.txt --replay-file solution.txt` -> `valid run: 293 minutes (246 actions)`.
- A* over (cell, bridges down), plus tide phase on fords; heuristic = walking time with all bridges down and fords dry. Runs in under a second.
- 293 is the optimum: a plain Dijkstra over every (cell, bridges, minute mod 16) state, stepping with the simulator itself, gives the same number.
- Submit `main.py` (writes `solution.txt`); `solution.txt` here is its output.
