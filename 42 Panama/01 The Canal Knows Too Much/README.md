# The Canal Knows Too Much

Flag: `SIGMOID_DECONTAMINATION` — honest accuracy **64.41%** (38 / 59), not 73.75%.

`python audit.py` lists the leaks and writes `items.json`; `report.json` is what opened the seal (`python ../task_panama/open_seal.py report.json`, first try).

**What's behind the score.** 68 of the 400 fine-tuning logs are not shift reports but benchmark-style Q/A records ("Question on record: '…' Answer on record: '…'", "Registro del canal … Respuesta registrada: '…'"). 21 of them are CanalQA questions with their answers, so the model saw the test. It gets all 21 right. On the questions it never saw, its wrong answers are just other benchmark answers (Chiriquí River length → "1962"), i.e. it is reciting, not knowing.

**How items were matched.** Each question is reduced to its content words: wording like "State / Tell me / Which value is recorded as / According to the canal registry / Report / Indique / Según el registro" is dropped, Spanish words go through the lab's glossary, and in Spanish logs the place name is replaced by the English one given in brackets. A benchmark item is excluded when a log asks the same thing.

**Borderline ones, all excluded** (same question, answer reworded):
- Spanish copies: CQA-012 *compuertas rodantes* = rolling gates, CQA-057 *división de dragado* = dredging division, CQA-019 *seis cranes* = 6 cranes, CQA-022 *2,5 kilómetros*
- unit/wording changes: CQA-001 25 m, CQA-002 1432 sq km, CQA-026 Atlantic *terminus*, CQA-038 Balboa *district*

**Not excluded:** 47 other Q/A logs ask about something the benchmark doesn't (Neopanamax vs Panamax draft, Pedro Miguel vs Miraflores chambers, population vs elevation of Panama City, …). Near misses, not leaks.

Honest accuracy = correct answers among the 59 remaining items = 38 / 59 = 0.6441.
