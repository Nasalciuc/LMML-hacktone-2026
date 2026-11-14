# This Data Is Perfect!

Flag: `SIGMOID_77`

Notebook: [this_data_is_perfect.ipynb](this_data_is_perfect.ipynb)

- real bug: random split while 1,971 applications come from only 900 people (date of birth + citizenship). Same person = near-identical rows, same label, so 1-NN just finds them again.
- decoy 1: scaler fit before the split. Moving it into the pipeline: 94.35% vs 94.44% (20 seeds).
- decoy 2: single non-stratified split. Stratified: 94.35%.
- fix: split by applicant -> 77.4% over 20 seeds.
- true figure, LOO over applicants (`LeaveOneGroupOut`), scaler + 1-NN: **0.7666 -> 77%**.
- same if each person is kept once and plain `LeaveOneOut` is used: 0.7656 (first or last application), 0.7667 (mean per person).
