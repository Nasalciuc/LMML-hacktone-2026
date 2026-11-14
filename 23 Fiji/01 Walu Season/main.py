"""Walu Season: UCB with a variance bonus.

Each spot gets a score = average catch + a bonus that grows with the spread
seen so far and shrinks with the number of casts there. A walu spot that has
not shown a school yet still looks wide: the variance has a floor that only
fades as casts pile up, so a few dry casts do not write it off.

No randomness is used, so the same seeds always give the same total.
"""

import json
import math
import sys


class Strategy:
    INIT = 3          # casts per spot before scoring starts
    C = 0.5           # weight of the variance bonus
    VAR_FLOOR = 25.0  # assumed spread of an unseen school, fades as 1/(n+1)
    EXTRA = 4.0       # small range term, as in UCB-V

    def __init__(self, n_spots, n_casts):
        self.k = n_spots
        self.n = [0] * n_spots
        self.s = [0.0] * n_spots
        self.s2 = [0.0] * n_spots
        self.t = 0

    def choose(self):
        for i in range(self.k):
            if self.n[i] < self.INIT:
                return i
        lt = math.log(self.t + 1)
        best, arg = -1e18, 0
        for i in range(self.k):
            n = self.n[i]
            m = self.s[i] / n
            var = max(self.s2[i] / n - m * m, self.VAR_FLOOR / (n + 1))
            v = m + self.C * math.sqrt(2 * var * lt / n) + self.EXTRA * lt / n
            if v > best:
                best, arg = v, i
        return arg

    def update(self, spot, catch):
        self.t += 1
        self.n[spot] += 1
        self.s[spot] += catch
        self.s2[spot] += catch * catch


def main():
    with open("test-input.json") as f:
        rules = json.load(f)

    for _day in range(rules["n_days"]):
        agent = Strategy(rules["n_spots"], rules["n_casts"])
        for _cast in range(rules["n_casts"]):
            spot = agent.choose()
            print(spot, flush=True)
            catch = float(sys.stdin.readline())
            agent.update(spot, catch)


if __name__ == "__main__":
    main()
