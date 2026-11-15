# Bihainim Angkol: `task_papua_new_guinea` (LMML / Sigmoid)

**Țara: Papua Noua Guinee.** Povestea se petrece în Kimbe Bay, provincia West New Britain. „Bihainim angkol” înseamnă „urmează-l pe unchiul” în Tok Pisin.

## TL;DR

| | |
|---|---|
| **Fișierul de urcat** | `submission/main.py`, sha256 `d67f8f5d89468224588ca1215cafc8936ecf237bd463e07d6bad3fc7dc8bfd15` |
| **NU urca** | `task_papua_new_guinea/main.py`: e template-ul gol al organizatorilor (WAIT peste tot, 0 pești) |
| **Rezultat pe cele 400 de zile de practică** | 5.963 pești/zi față de 2.490 la angkol, adică **2.395×** (ținta e 1.5×). Comanda oficială afișează `above the target` |
| **Cât de bun e** | Politica e optimul exact, verificat cu fracții, deci în medie nu se poate face mai bine. Batem cu 1.88× chiar și un angkol *perfect* |
| **Cum re-verifici tot** | `python run_all.py` (~10 s, doar stdlib), apoi compari cu `run_all_output.txt` |
| **Flag** | `SIGMOID_<WORD>`, apare automat când `main.py` trece evaluarea |

---

## 1. Ce cere task-ul

Laguna, cu rândul 0 spre larg și rândul 5 spre țărm:

```
    0 1 2 3 4 5 6
 0  . . . . T T .     H  plaja satului: aici se descarcă peștele
 1  # # # # ~ # #     s  seagrass: 20% șansă de pește la FISH
 2  . . . . s s .     T  recif exterior: 90% șansă de pește la FISH
 3  . . . . s s .     #  barieră de recif: nu se poate trece
 4  . . . . . . .     ~  pasajul prin recif: valul (40%) împinge barca o celulă la sud
 5  . . H . . . .     .  apă
```

- O zi are 24 de pași. Pornești de pe plajă (5,2) cu cala goală, care încape maximum 6 pești.
- Starea e `(row, col, hold, steps_left)`. Acțiunile sunt NORTH, SOUTH, EAST, WEST, FISH și WAIT.
- Peștele se descarcă automat când barca termină un pas pe H. Ce a rămas în barcă la sfârșitul zilei se pierde.
- Toate evenimentele aleatorii vin din seed-ul zilei: `random.Random(seed)` generează 24 de numere pentru val și 24 pentru pește. Același seed cu aceleași decizii dă aceeași zi.
- **Scoring:** media capturii pe zile *nevăzute* trebuie să fie ≥ 1.5× media lui angkol pe aceleași zile, ca să iei PASS și flag-ul. Primești 0 dacă `main.py` crapă sau depășește 60 de secunde, ori dacă în `test-output.csv` lipsește o stare, se repetă una sau apare o acțiune necunoscută.
- **Constrângeri:** Python 3.12, doar stdlib plus numpy 2.2.6, fără internet. `main.py` rulează singur, doar cu `test-input.csv` lângă el.

## 2. Ce am aflat din logbook (`check/analyze_logbook.py`)

- **Logbook-ul e exact simulatorul `lagoon.py` cu seed = numărul zilei.** Am rejucat toate cele 400 de zile × 24 de pași: 0 divergențe, iar captura e identică cu `train-output.json` în fiecare zi. Concluzia: mediul de evaluare e cel din `lagoon.py`.
- **Ce face angkol:** N, N, E, E până la (3,4) pe seagrass, apoi FISH până mai rămân ~6 pași, apoi acasă. Descarcă de obicei la `steps_left = 3`.
- **Nu trece niciodată de pasaj și nu pescuiește niciodată la recif.** N-a fost nici măcar o dată pe rândurile 0–1. Din 5,000 de acțiuni FISH, 4,954 sunt pe seagrass, iar restul sunt zgomot.
- **E stocastic:** 4.76% din decizii diferă de acțiunea lui obișnuită în acea stare, adică ~5.7% acțiuni aleatorii.
- **Capturile lui:** 0 pești în 46 de zile, 1 în 59, 2 în 109, 3 în 85, 4 în 57, 5 în 28 și 6 în 16. Media e **2.49**.

Concluzie: angkol stă într-un loc unde peștele trage 20% din timp, deși cu 3 pași mai încolo (prin pasaj) e un loc cu 90%. Dacă îl imiți, rămâi la ~2.5 pești pe zi. Ca să-l depășești, trebuie să treci pasajul spre recif.

## 3. Soluția (`submission/main.py`)

Mediul e cunoscut complet: harta, probabilitățile și regulile sunt în `lagoon.py`, iar logbook-ul confirmă că exact așa se comportă. Deci problema e un **MDP cu orizont finit** și se rezolvă exact prin **programare dinamică (backward induction)**, fără nicio învățare aproximativă.

- **Stări:** 36 de celule de apă × 7 valori pentru cală × 24 de pași = **6,048**, exact cât are `train-input.csv`.
- **Tranziții:** FISH pe s/T cu cala < 6 adaugă un pește cu p = 0.20 / 0.90. Dacă barca termină pasul pe `~`, cu p = 0.40 e împinsă o celulă la sud. Pe H se descarcă tot.
- **Bellman**, cu t = `steps_left`, fără discount:
  `V_0(s) = 0`, `V_t(s) = max_a Σ_s' P(s'|s,a) · [pești_descărcați(s,a,s') + V_{t-1}(s')]`
- Se calculează pentru t = 1 … 24 și se ia argmax. Calculul se face la fiecare rulare a `main.py` (~0.1 s în Python pur), deci fișierul nu depinde de nimic extern: importă doar `csv`.
- **De ce e suficientă o politică pe `(row, col, hold, steps_left)`:** numerele aleatorii ale zilei nu depind de decizii și sunt independente de la un pas la altul. Politica nu vede seed-ul, deci istoricul nu spune nimic despre viitor, iar politica Markov optimă e optimul global.
- **Tie-break-uri**, care nu schimbă captura așteptată (verificat exact, vezi §5):
  1. captura așteptată, cu toleranță 1e-9;
  2. descărcarea peștelui cât mai devreme (maximizează E[Σ pești × steps_left la descărcare]), ca barca plină să vină direct acasă;
  3. ordinea WAIT > NORTH > SOUTH > EAST > WEST > FISH, ca barca să stea pe plajă când nu mai e nimic de prins.

**Ce face politica:** pleacă N, N, N, E, E până la (2,4) și intră în pasaj (dacă valul o împinge înapoi, mai încearcă). Iese la (0,4) și pescuiește cu 90% până umple cala, apoi coboară; valul o împinge adesea gratis din (1,4) în (2,4). Urmează S, S, S, W, W, descarcă 6 pești și apoi WAIT. Drumul de întoarcere de la recif durează cel mult 7 pași, determinist, pentru că valul împinge doar spre casă. De aceea politica **nu pierde niciodată pește pe mare** și aduce **6 pești în 97.3% din zile**.

Ziua 1, rejucată cu `lagoon.py --day 1`:

```
24 (5,2) NORTH  23 (4,2) NORTH  22 (3,2) NORTH  21 (2,2) EAST  20 (2,3) EAST
19 (2,4) NORTH  18 (1,4) NORTH  17..12 (0,4) FISH x6 -> cala 6
11 (0,4) SOUTH  [valul o duce din (1,4) în (2,4)]  10 (2,4) SOUTH  9 (3,4) SOUTH
 8 (4,4) SOUTH   7 (5,4) WEST   6 (5,3) WEST -> descarcă 6   5..1 WAIT pe plajă
```

## 4. De ce trece

| | Pești/zi | Raport |
|---|---|---|
| angkol (logbook, 400 de zile) | 2.49 | 1.00× |
| **cel mai bun pescar posibil care nu trece pasajul** (limita teoretică pentru stilul lui angkol) | 3.16 | |
| `main.py`, optim exact (fracții) | **5.9575** | 2.39× față de 2.49 |
| `main.py`, 400 de zile de practică (compare oficial) | 5.963 | **2.395×** |
| `main.py`, 20,000 de zile nevăzute (seed 401–20400) | 5.959 | 2.52× față de clona lui angkol |
| `main.py` față de angkol *perfect* (3.16) | | **1.88×** |

Chiar dacă „angkol” din evaluare ar fi un pescar perfect de seagrass, tot trecem de 1.5×.

## 5. Verificările (`python run_all.py` le refă pe toate)

Coloana „Pas” e numărul secțiunii `=== N. ... ===` din output-ul lui `run_all.py`.

| Pas | Ce verifică | Script | Rezultat |
|---|---|---|---|
| 1 | Logbook-ul = `lagoon.py` cu seed = zi | `check/analyze_logbook.py` | 0 divergențe în 400 de zile, captura identică |
| 2 | `main.py` rulat ca la grader: singur, într-un folder gol, `python ./main.py` | `run_all.py` | exit 0, ~0.1 s. `run_all.py` îl rulează cu Python-ul tău și, dacă îl ai, cu `python3.12`. Separat am verificat și pe 3.11, 3.12.3, 3.12.13 și 3.13: output identic byte cu byte |
| 3 | Comanda oficială din enunț | `lagoon.py --compare` | 5.963 față de 2.490, **2.395×, above the target** |
| 4 | DP independent scris pe funcția de tranziție din `lagoon.py` (`_advance`) | `check/verify.py` | 0 stări diferite din 6,048; validatorul oficial (`check_policy_file`) nu găsește probleme |
| 4 | Zile nevăzute contra unei clone a lui angkol (acțiunea majoritară plus 5.7% zgomot) | `check/verify.py` | 0 din 1,000 de grupuri de 20 de zile sub 1.5× (minimul e 1.74×) |
| 5 | Optimalitate cu **aritmetică exactă** (`fractions.Fraction`) | `check/exact_check.py` | V\* = 1489374012481/250000000000 = 5.957496…, iar politica e **identică cu V\*** și e argmax exact în toate cele 6,048 de stări |
| 6 | Puține zile de test, pește pierdut, constante diferite | `check/stress.py` | vezi §6; pește pierdut pe mare: 0 din 10,000 de zile |
| 7 | Format față de unealta organizatorilor | `check/format_parity.py` | identic cu template-ul (header, ordine, CRLF); un `main.py` generat cu `lagoon.export_main(politica noastră)` dă un `test-output.csv` identic byte cu byte |
| 8 | Input-uri neobișnuite | `check/input_robustness.py` | rânduri amestecate, subset, BOM+CRLF, header ciudat, `5.0`, stări invalide, `inf`/`nan`, doar header: toate OK, fără crash |

Toate scripturile folosesc seed-uri fixe. Rulat la tine, `run_all.py` trebuie să dea exact aceleași numere ca `run_all_output.txt`; doar timpii diferă.

### Verificare manuală, ca în enunț (Linux, macOS sau Git Bash)

```bash
mkdir -p /tmp/png && cp submission/main.py /tmp/png/ && cp task_papua_new_guinea/train-input.csv /tmp/png/test-input.csv
(cd /tmp/png && python main.py)
python task_papua_new_guinea/lagoon.py --policy /tmp/png/test-output.csv --compare task_papua_new_guinea/train-output.json
# -> ratio 2.395 x angkol (target 1.5 x): above the target
```

Atenție: nu rula `python main.py` direct în `task_papua_new_guinea/`, pentru că acolo `main.py` e template-ul gol.

## 6. Riscuri rămase (sincer)

1. **Un set de test foarte mic.** Am luat la întâmplare subseturi din cele 400 de zile reale, cu capturile adevărate ale lui angkol. Probabilitatea ca raportul să iasă sub 1.5× e 11.6% la 1 zi, 4.2% la 3 zile, 1.6% la 5 zile, 0.1% la 10 zile și **0% de la 20 de zile în sus**. Nicio politică nu poate elimina riscul ăsta: într-o zi în care angkol aduce 5–6 pești, nimeni nu poate aduce 1.5× pentru că în cală încap 6. Setul de practică are 400 de zile, deci testul e aproape sigur de aceeași mărime.
2. **Șansa de pește la recif (0.9) și valul (0.4) nu pot fi confirmate din logbook**, pentru că angkol n-a fost niciodată acolo. Vin din `lagoon.py` și din enunț („under the same tide and bites”). Chiar dacă ar fi altele, aceeași politică trece: cu 0.5 la recif iese 1.92×, cu val de 0.8 iese 2.02×, cu ambele mai slabe (0.6 și 0.6) iese 2.06×. Iar cu 0.3 pe seagrass, unde angkol ar profita, iese 1.71×.
3. **Clona lui angkol e aproximativă:** 2.43 față de 2.49 real pe zilele de practică. Contează doar la testul pe zile nevăzute. Verificările pe date reale (compare oficial, subseturile din §6.1) nu depind de ea.

## 7. Cum se trimite

1. Urci **`submission/main.py`**. Verifică sha256 `d67f8f5d…bfd15`: pe Linux cu `sha256sum`, pe Windows cu `certutil -hashfile main.py SHA256`.
2. Dacă trece, flag-ul `SIGMOID_<WORD>` apare automat.
3. Context: submisia anterioară a echipei (14/11/2026 14:36) a picat cu `ModuleNotFoundError: No module named 'environment'`, pentru că importa un modul care nu există la grader. Noul `main.py` importă doar `csv` și își calculează singur politica la pornire.

## 8. Conținutul arhivei

```
papua_new_guinea_handoff/
├── README.md                  acest fișier
├── submission/main.py         SINGURUL fișier de urcat
├── run_all.py                 refă toate verificările: python run_all.py [folder_task]
├── run_all_output.txt         output-ul rulării noastre, pentru comparație
├── check/
│   ├── analyze_logbook.py     replay-ul logbook-ului + comportamentul lui angkol
│   ├── verify.py              DP independent + validator + 20,000 de zile nevăzute
│   ├── exact_check.py         optimalitate exactă + limita pentru stilul lui angkol
│   ├── stress.py              puține zile, pește pierdut, constante diferite
│   ├── format_parity.py       format față de template-ul și export_main ale organizatorilor
│   └── input_robustness.py    input-uri neobișnuite
└── task_papua_new_guinea/     copie neschimbată din arhiva organizatorilor
```

Ca să confirmi că folderul `task_papua_new_guinea/` e identic cu ce ai descărcat tu, compară SHA-256 (sau rulează `python run_all.py <folderul tău de task>`):

| Fișier | SHA-256 |
|---|---|
| `task_papua_new_guinea-participant.zip` (originalul) | `9aaa871596fa21340a6cc97471277fffce0b9e1635fb7580f8d84bd06ac78c89` |
| `lagoon.py` | `ee1f12a159a203289e2c552c0a2b7087103c7ed4464c0de18454661889b4f34c` |
| `main.py` (template) | `302725e2958aeb2dfa02b4d606f951aa991e061895f118f652868da0566dbc5f` |
| `train-input.csv` | `42b16248cd5704fb002d0b571f438b61d96e5451037b925c645c93c375d02c39` |
| `train-output.json` | `295c6d256995994b8474d9bdd1774414a6d69f09aead74f3f11c2cbf3f5b1405` |
| `dataset/logbook.csv` | `dac4b3461543d5f94b28d784dd3f462be0388be785d2f598bd5056b67ade5317` |
