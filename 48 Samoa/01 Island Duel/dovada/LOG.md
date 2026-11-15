# Evaluation attempts log

Qualify bar: >= 308 wins of 512 AND wins in >= 24 of 32 groups. Fresh hidden maps.
Each attempt uses a NEW --output name; all evidence is preserved.

| attempt | agent | params | elapsed | wins/512 | winning_groups | result |
|---|---|---|---|---|---|---|
| attempt-1 | my_agent.py (agent4, eg10) | baked (a2_r1_mean + eg_plies10) | ~4 min | 319 (0.623) | 32/32 | QUALIFIED |

attempt-1 files: attempt-1.json (full), attempt-1.events.jsonl (journal),
attempt-1.html (sample replays), attempt-1.chart.json + attempt-1.chart.html (earned chart).
Seat split 164/155 (seat0/seat1); 11 draws. Threshold was >=308 wins and >=24 groups.

The chart is DETERMINISTIC: the exact JSON (same star coords/letters) is baked into
samoa-windows-x86_64.exe, so re-running eval yields an identical chart. No second
attempt needed for the puzzle.

Chart solved (no new eval needed): Greek-order guide+companions `TCEVNMKHEYYCNVIZU`,
Vigenere-decrypted with Samoa's motto key FAAVAEILEATUASAMOA -> OCEANICWAYFINDING.
Flag: SIGMOID_OCEANICWAYFINDING.
