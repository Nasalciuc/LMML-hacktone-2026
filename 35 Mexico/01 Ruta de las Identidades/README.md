# Ruta de las Identidades

Submit `main.py` (stdlib). Reads `test-input.csv` (or a path given as the first argument), writes `test-output.json`.

Names are messy (swapped first/last, initials, typos, nicknames like Pancho/Conchita), addresses too (Av./Avenida, Int./Interior, number moved to the front, accents dropped). The numbers in the address are not: street number, building, flat and postal code survive every variant. So records are grouped by the sorted list of numbers in the address and every cross-table pair in a group is a match. Only if one table has two records at the same address do the names decide (shared initial, nicknames mapped).

Train: 12,038 pairs predicted, 12,038 expected, 0 missing, 0 extra.
