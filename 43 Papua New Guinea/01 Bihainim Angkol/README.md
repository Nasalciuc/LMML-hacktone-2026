# Papua Noua Guinee: Bihainim Angkol

**Ce trimiți: `main.py` din acest folder.** Flag-ul apare singur pe platformă când fișierul trece.

Nu urca `main.py` din folderul organizatorilor (`task_papua_new_guinea`). Acela e șablonul gol, care stă pe loc și prinde 0 pești.

## Ce cere taskul

O barcă pescuiește într-o lagună. O zi are 24 de pași. Barca pleacă de pe plajă cu cala goală, iar în cală încap cel mult 6 pești.

```
    0 1 2 3 4 5 6
 0  . . . . T T .     H  plaja: aici se descarcă peștele
 1  # # # # ~ # #     s  iarbă de mare: 20% șansă de pește la o încercare
 2  . . . . s s .     T  reciful din larg: 90% șansă de pește
 3  . . . . s s .     #  barieră: nu se poate trece
 4  . . . . . . .     ~  trecerea prin barieră: valul împinge barca înapoi în 40% din cazuri
 5  . . H . . . .     .  apă
```

La fiecare pas alegi o mișcare: nord, sud, est, vest, pescuiește sau așteaptă. Peștele contează doar dacă ajunge pe plajă. Ce rămâne în barcă la sfârșitul zilei se pierde.

Trebuie să spui ce face barca în orice situație posibilă: poziție, câți pești are în cală și câți pași au rămas. Sunt 6.048 de situații. Treci dacă prinzi în medie de cel puțin 1,5 ori mai mult decât unchiul („angkol”) din jurnalul dat ca exemplu.

## Ce face unchiul

Merge la iarba de mare, la 4 pași de plajă, și pescuiește acolo toată ziua. Nu trece niciodată bariera. Prinde în medie 2,49 pești pe zi.

Greșeala lui: la 3 pași mai departe, dincolo de trecere, peștele mușcă în 90% din cazuri, nu în 20%.

## Soluția

Regulile lagunei sunt cunoscute complet, sunt în fișierul `lagoon.py` al organizatorilor. Am verificat că jurnalul unchiului se poate rejuca exact cu ele: 400 de zile, nicio diferență.

Când regulile sunt cunoscute, cea mai bună mișcare se poate calcula exact, fără nimic învățat. Calculul merge de la sfârșitul zilei spre început:

1. Cu 0 pași rămași nu mai poți câștiga nimic.
2. Cu 1 pas rămas, pentru fiecare situație alegi mișcarea care aduce cei mai mulți pești în medie.
3. Cu 2 pași rămași faci la fel, folosind ce ai aflat la pasul anterior.
4. Continui până la 24 de pași.

Metoda se numește programare dinamică. `main.py` face tot calculul la fiecare rulare, în cam 0,1 secunde, și folosește doar Python simplu.

Ce iese: barca merge direct la trecere, iese în larg, umple cala la recif și se întoarce. Valul împinge doar spre casă, deci drumul înapoi durează cel mult 7 pași și barca nu rămâne niciodată cu pește pe mare.

## Cât de bine merge

| | Pești pe zi |
|---|---|
| Unchiul | 2,490 |
| Soluția | 5,963 |
| Raport | 2,395 (trebuie 1,5) |

Măsurat pe cele 400 de zile de practică. Barca aduce 6 pești în 97,3% din zile. Calculul fiind exact, în medie nu se poate mai bine.

## Ce nu e sigur

Aproape nimic. Singura presupunere e că zilele ascunse folosesc aceleași reguli ca `lagoon.py`. Jurnalul confirmă asta.

## Verificare

`main.py` are sha256 `d67f8f5d89468224588ca1215cafc8936ecf237bd463e07d6bad3fc7dc8bfd15`.

Rulare: pui `test-input.csv` lângă `main.py`, apoi `python main.py`. Iese `test-output.csv`.

## Repo

Soluția e și în repo-ul local, în `43 Papua New Guinea\01 Bihainim Angkol`, cu același README și cu descrierea lungă în `DETALII.md`. Nu e comisă și nu e urcată pe GitHub.
