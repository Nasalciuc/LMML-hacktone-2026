# What The Data?

Submit `main.py` (stdlib only). Reads `test-input.csv`, writes `test-output.csv`.

- skips the `#` / blank preamble, whatever its length
- `,` decimals -> `.`, F -> C, one decimal, half-up
- names: undoes the wrong cp1252/latin-1 decode as many times as it was applied (Ã°, ÃƒÂ° -> ð)
- duplicates by `reading_id`, keeps one, sorted

On the practice export the output is identical to `train-output.csv`. `make_messy.py` builds harder exports (17 stations incl. Þ, Ö, Á, Í, names broken up to 3 times, more F rows, longer preamble); all 8 seeds match.
