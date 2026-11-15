# Verificare independentă (15 nov. 2026)

Am testat arhiva `papua_new_guinea_handoff.zip` separat de scripturile ei, pe Windows, cu Python 3.12.10 și 3.13.

**Concluzie:** `submission/main.py` e corect și gata de trimis. Pe platformă nu a fost încă trimis; ultimul rezultat de acolo e eroarea veche din 14 nov., 14:36 (`No module named 'environment'`).

## Ce am verificat

| Verificare | Rezultat |
|---|---|
| sha256 al `submission/main.py` | `d67f8f5d89468224588ca1215cafc8936ecf237bd463e07d6bad3fc7dc8bfd15`, ca în README |
| Fișierele organizatorilor din arhivă | identice cu cele descărcate de noi (aceleași sha256 pentru `lagoon.py`, template, `train-input.csv`, `train-output.json`, `logbook.csv` și pentru zip-ul original) |
| Ce face codul | importă doar `csv`; citește `test-input.csv`, scrie `test-output.csv`; fără rețea, fără alte fișiere |
| Rulat singur într-un folder gol (`python ./main.py`) | exit 0, 0,09 s (limita e 60 s) |
| Python 3.12.10 și 3.13 | `test-output.csv` identic |
| Format | 6.048 de rânduri, aceleași stări în aceeași ordine ca intrarea, CRLF; `lagoon.check_policy_file` nu găsește probleme |
| Comanda oficială `lagoon.py --compare train-output.json` | 5,963 față de 2,490 pești pe zi, raport 2,395 (ținta 1,5) |
| Valoarea așteptată, calculată de mine pe regulile din `lagoon.py` | 5,957496 pești pe zi, egală cu a politicii mele optime calculate separat |
| 20.000 de zile nevăzute (seed 401–20400) | 5,959 pești pe zi; cala plină în 97,3% din zile |
| Intrări neobișnuite (rânduri amestecate, doar antet, primele 10 rânduri) | exit 0, acțiuni corecte |
| `run_all.py` rulat aici cu Python 3.12.10 | toate cele 8 secțiuni până la capăt; toate numerele identice cu `run_all_output.txt` (diferă doar timpii și versiunea de Python) |

Față de varianta mea veche (`varianta_veche/`), politica diferă la 2.068 din 6.048 de stări, dar numai acolo unde două acțiuni au aceeași valoare. Captura așteptată e aceeași.

## Cum repeți verificarea

În acest folder, task-ul e un nivel mai sus, deci:

```
python run_all.py "..\task_papua_new_guinea"
```

În pachetul pentru coleg, folderul `task_papua_new_guinea` e inclus, deci merge direct `python run_all.py`.

## De trimis pe platformă

Un singur fișier: `submission/main.py`.
