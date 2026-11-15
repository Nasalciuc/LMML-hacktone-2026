# Etiopia: The Azmari's Borrowed Melody

**Ce trimiți: `main.py` din acest folder.**

Dacă platforma îl respinge, a doua încercare e `a_doua_incercare/main.py`. Spune-mi atunci care test a picat (test-1, test-2 sau test-3). Platforma arată doar primul test picat.

## Ce cere taskul

Primești o înregistrare de 45 până la 60 de secunde cu un instrument etiopian, masenqo (cu arcuș) sau washint (fluier). În ea a fost lipită o bucată de câteva secunde din altă înregistrare. Trebuie să spui la ce secundă începe bucata lipită și la ce secundă se termină.

Treci un test dacă ambele margini sunt la cel mult 1,5 secunde de adevăr. Sunt trei înregistrări ascunse. Avem un singur exemplu cu răspuns: bucata lipită e între 14,96 și 20,97 secunde.

## Ideea

Bucata lipită vine din altă înregistrare. A fost făcută cu alt microfon, în altă cameră, cu alt zgomot de fond. Deci nu căutăm „altă melodie”, căutăm „alt fel de a suna”.

## Pașii

```
sunet -> bucăți scurte -> pentru fiecare: cum sună înregistrarea -> blocul care diferă cel mai mult de rest -> început, sfârșit
```

1. Sunetul e tăiat în bucăți scurte, de câteva sutimi de secundă.
2. Pentru fiecare bucată se calculează măsuri despre cum a fost înregistrată: cât zgomot e pe frecvențele joase, cum e împărțită energia pe frecvențe, cât de tare e fondul.
3. Se caută blocul continuu, de cel mult 20 de secunde, care diferă cel mai mult de restul fișierului.
4. Marginile blocului se scriu în CSV.

Față de detectorul primit de la coleg am schimbat două lucruri:

- Am scos din căutare măsurile care urmăresc notele cântate. Pe alte melodii ele mutau marginile sau alegeau un pasaj normal al instrumentului.
- Blocul căutat are cel mult 20 de secunde, nu 40. Enunțul spune „câteva secunde”, iar blocurile mai lungi au fost mereu alegeri greșite.

## Cât de bine merge

Pe exemplul dat: răspunde 14,965 și 20,912. Adevărul e 14,960 și 20,965.

Am făcut 344 de înregistrări de test cu răspuns cunoscut: 32 din exemplul dat și 312 sintetice.

| Detector | Trecute din 344 |
|---|---|
| `main.py` (acesta) | 334 |
| Cel primit de la coleg, neschimbat | 327 |
| `a_doua_incercare/main.py` | 312 |

La cazurile trecute, greșeala de margine e de obicei 0,04 secunde.

## Ce nu e sigur

- Bucăți foarte scurte, sub 2,5 secunde: 25 din 30.
- Același instrument, înregistrat la fel ca restul: 98 din 104. Aici diferă doar felul de a cânta, iar urma e slabă.
- Dacă fișierul are două zone ciudate, o alege pe cea mai puternică.
- Detectorul răspunde mereu cu un bloc, chiar dacă nu e nimic lipit.
- Toate cifrele sunt pe înregistrări făcute de mine, nu pe cele trei ascunse.

## Verificare

Rulare: `python main.py test-1-input.wav`. Scrie un CSV cu începutul și sfârșitul.

- `main.py`: sha256 `55d236f83f1d9c058434f7042c0dd60dd5f4797cfca0d7a17af720f71bf176a1`
- `a_doua_incercare/main.py`: sha256 `f6e1b1b5a6a40a1c82eb15dd28271afa6b88cd829828cd677632c6209b3d62ba`

## Repo

Soluția e și în repo-ul local, în `22 Ethiopia\01 The Azmari's Borrowed Melody`, cu același README și cu descrierea lungă în `DETALII.md`. Nu e comisă și nu e urcată pe GitHub.
