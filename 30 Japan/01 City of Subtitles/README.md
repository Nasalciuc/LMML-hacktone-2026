# Japonia: City of Subtitles

**Nu e nimic de trimis. Cuvântul secret nu a fost găsit.**

Prima jumătate a taskului e făcută și e sigură. A doua jumătate, cea care dă flag-ul, lipsește.

## Ce cere taskul

Șase fotografi postează poze din oraș, fiecare cu o subtitrare. În săptămânile 1–7 pozele sunt semnate: 108 poze. În săptămânile 8–12 sunt 145 de poze fără nume.

Trebuie să afli cine a făcut fiecare poză nesemnată, să găsești pozele lui Toorisugari („trecătorul”) și să scoți din ele un cuvânt ascuns. Flag-ul e `SIGMOID_<CUVÂNT>`.

Indiciul din enunț: pozele unui simplu trecător ar putea ști mai mult decât arată, și poate acolo se ascunde identitatea unui fotograf necunoscut.

## Partea făcută: cine a făcut fiecare poză

Fiecare senzor de cameră are mici defecte de fabricație. Ele lasă în orice poză același zgomot foarte slab, ca o amprentă. Din pozele semnate se scoate amprenta fiecărui fotograf. Apoi fiecare poză nesemnată se compară cu cele șase amprente.

```
poze semnate -> amprenta fiecărui fotograf
poză nesemnată -> zgomotul ei -> cu care amprentă se potrivește cel mai bine? -> fotograful
```

Capcana: o parte din poze au fost făcute cu telefonul ținut invers și postate drept. La ele amprenta apare rotită cu 180 de grade. Fără să verifici și varianta rotită, pozele acelea par făcute de o cameră necunoscută. După ce verifici ambele orientări, totul se așază.

Rezultat, în `attribution_table.csv`:

- Toorisugari are 45 de poze, exact 9 în fiecare săptămână.
- Fiecare dintre ceilalți cinci are 20 de poze, exact 4 pe săptămână.
- Pe pozele semnate metoda nimerește 105 din 108.
- Două sesiuni independente au ajuns la același fotograf pentru toate cele 145 de poze.

Cele mai slabe poze sunt `w12_016` și `w10_013`, întunecate, puse la amayadori ca să iasă 4 pe săptămână.

## Partea lipsă: cuvântul

Am căutat cuvântul în multe locuri și nu l-am găsit:

- în zgomotul camerei lui Toorisugari: un al doilea tipar ascuns, imagini în amprentă, diferențe între poze;
- în pixeli și în fișier: metadate, date ascunse după sfârșitul fișierului, steganografie JPEG, benzile negre;
- în subtitrările de pe poze: citite automat pe toate, apoi căutate acrostihuri, caractere rare, lungimi de rânduri;
- în ordinea și orientarea pozelor: care sunt rotite, pe ce poziții stau în săptămână, citite ca biți.

Singurul lucru pus clar în date și nefolosit încă e împărțirea de 9 poze pe săptămână la Toorisugari față de 4 la ceilalți.

## Fișiere

- `attribution_table.csv`: fotograful fiecărei poze nesemnate, dacă e rotită și cât de sigură e potrivirea.

Tot codul și lista completă a verificărilor sunt în repo, în `30 Japan\01 City of Subtitles\README.md`.

## Repo

Soluția e și în repo-ul local, în `30 Japan\01 City of Subtitles`, cu același README și cu descrierea lungă în `DETALII.md`. Nu e comisă și nu e urcată pe GitHub.
