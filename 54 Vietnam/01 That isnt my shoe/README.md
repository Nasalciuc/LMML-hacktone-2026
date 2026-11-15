# Vietnam: That isn't my shoe!

**Ce trimiți: `sherlock_labels.pt` din acest folder.**

Platforma nu spune în ce format vrea răspunsul. Dacă dă eroare de format, încearcă pe rând fișierele din `alte_formate/`. E același model, doar răspunsul e scris altfel:

1. `sherlock_probs4.pt`: 4 numere pe poză, în ordinea nike, adidas, converse, unknown.
2. `sherlock_probs4_a-c-n.pt`: la fel, în ordine alfabetică (adidas, converse, nike, unknown).
3. `sherlock_logits3.pt`: 3 numere pe poză, fără „unknown”.

## Ce cere taskul

Avem 735 de poze cu pantofi sport de la trei mărci: Nike, Adidas, Converse. La test apar și alte mărci, pe care modelul nu le-a văzut: Puma, Reebok, Vans, Asics. Pentru ele răspunsul corect e `unknown`.

Treci dacă sunt îndeplinite amândouă:

- la mărcile cunoscute, cel puțin 85% răspunsuri corecte;
- din pozele cu alte mărci, cel puțin 70% primesc `unknown`.

Se trimite un singur fișier `.pt` de cel mult 10 MB. Platforma îl rulează singură.

## Ideea

Să ghicești marca dintre trei e partea ușoară. Partea grea e să-ți dai seama că o poză nu e din niciuna.

Modelul face întâi o ghicire, apoi două verificări. Dacă verificările ies slab, răspunde `unknown`.

```
poză -> ghicește marca (nike / adidas / converse)
     -> verificarea 1: seamănă cu vreo poză din cele 735?
     -> verificarea 2: a fost mărită la fel ca cele 735?
     -> ambele destul de bune: marca ghicită.  Altfel: unknown
```

**Ghicirea.** O rețea mică de imagini (MobileNetV3), antrenată pe cele 735 de poze.

**Verificarea 1: seamănă?** Modelul ține minte „amprenta” fiecărei poze din cele 735. Pentru o poză nouă caută cea mai apropiată amprentă. Dacă nici cea mai apropiată nu e aproape, poza e ciudată.

**Verificarea 2: a fost mărită la fel?** Pozele de antrenare au fost mărite de la 240 la 256 de pixeli. Mărirea lasă un tipar care se repetă la fiecare 16 pixeli. Autorii au presupus că pozele cu alte mărci vin din altă sursă și nu au tiparul. Verificarea asta face cea mai mare parte din treabă.

## Cât de bine merge

| Ce | Rezultat | Prag |
|---|---|---|
| Mărci cunoscute, corecte (testat pe poze ținute deoparte la antrenare) | 0,891 | 0,85 |
| Alte mărci respinse, după autori (testat pe poze care nu sunt pantofi) | 0,86–0,92 | 0,70 |
| Alte mărci respinse, măsurat de mine pe 1.750 de poze reale Puma, Reebok, Vans, Asics | 0,57–0,66 | 0,70 |
| La fel, dar dacă pozele au fost și ele mărite de la 240 la 256 | 0,02–0,04 | 0,70 |

Mărime: 6,95 MB. Timp: 17 până la 40 de milisecunde pe poză. Dă același răspuns pentru toate formatele de intrare încercate.

Modelul nostru vechi (`vietnam_model.pt`) respinge doar 29–38% din aceleași poze, deci acesta e cel mai bun pe care îl avem.

## Ce nu e sigur

- **Poarta de respingere.** Pe pantofi reali am măsurat sub 0,70. Pozele mele de test sunt miniaturi mici de pe internet. Pe cele 48 cu originalul mare, peste 400 de pixeli, respingerea a fost 0,71–0,79. Deci depinde de unde au luat organizatorii pozele.
- **Verificarea 2 se bazează pe felul în care a fost făcut setul de date**, nu pe cum arată pantoful. Dacă pozele cu alte mărci au trecut prin aceeași mărire, respingerea cade aproape la zero.
- **Marja la mărcile cunoscute e mică**: 0,891 față de 0,85.
- Un prag mai strict ar muta echilibrul: la 96–97% cunoscute păstrate am măsurat 0,875–0,882 la cunoscute și 0,73–0,79 la respingere. Varianta asta nu e făcută.

## Verificare

Fișierul se încarcă cu `torch.jit.load("sherlock_labels.pt")` pe torch 2.11. Primește poze 256×256 și întoarce o listă de etichete.

## Repo

Soluția e și în repo-ul local, în `54 Vietnam\01 That isnt my shoe`, cu același README și cu descrierea lungă în `DETALII.md`. Nu e comisă și nu e urcată pe GitHub.
