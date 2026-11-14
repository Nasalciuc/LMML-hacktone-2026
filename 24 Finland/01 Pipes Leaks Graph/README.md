# Pipes, Leaks, and the Graph Beneath

Flag: `SIGMOID_HOTLINE`

`python clean_routes.py`

- town names: strip spaces and trailing dots, lower case, drop å/ä/ö accents
- Swedish names mapped to the Finnish ones (Helsingfors -> Helsinki, Åbo -> Turku, Villmanstrand -> Lappeenranta, ... 61 pairs)
- a connection = unordered pair of towns, counted once whatever the office or route type

Result: 116 towns, 448 connections (matches the self-check). Top 8: Helsinki 39, Orivesi 38, Tornio 33, Lieksa 29, Imatra 27, Nokia 23, Eura 22, Närpiö 21 — all different, as the task says. The raw group-by put Orivesi first because Helsinki was split between Helsinki and Helsingfors.
