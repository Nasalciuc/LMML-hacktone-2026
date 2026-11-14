# Pura Vida, No PII

`main.py` (stdlib only) reads `test-input.json` and writes `test-output.json`; next to `train-input.json` it writes `train-pred.json` instead, so the golden file is never touched. Order: protect PV refs, then emails, IBANs, dates of birth, addresses (street/landmark/house anchors plus the city), digit runs (phone/cédula/account by digit count), then names from cues, signatures, lexicons and email local parts, and every detected name part is removed everywhere in the document.
Train (20 docs, 107 items): privacy 1.000, utility proxy 1.000, output identical to the golden file once placeholders are unified.
Synthetic, `gen_synth.py` seed 31 (1000 docs, 5832 items, 30% unseen names): privacy 1.000; seed 32 with only unseen names (5773 items): privacy 1.000; non-PII words kept 100% on both.
Hand-written free-form messages (`hard_cases.py`, 18 docs, 71 items): privacy 0.986, the miss is an unknown lowercase name after "my son".
`evaluate.py` takes items from the input/golden diff and counts an item as leaked if any variant survives (name parts, email local part, phone digits with/without 506, glued cédula/IBAN, DOB in other formats, address chunks); the "squashed substring" score is lower only because letters of `[NAME]` glue into short names, which also hits the golden file. `sh run_all.sh` reproduces everything.
