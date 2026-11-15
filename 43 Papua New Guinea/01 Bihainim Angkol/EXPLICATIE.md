# Papua Noua Guinee: Bihainim Angkol. Explicația completă

Documentul e scris pentru un coleg care nu a lucrat la task. După citire ar trebui să poți explica unui mentor ce s-a cerut și cum am rezolvat.

Două convenții. Zecimalele sunt scrise cu punct, ca în fișierele de verificare: 5.963 înseamnă aproape șase pești. Căile de fișiere pornesc din folderul taskului, `43 Papua New Guinea/01 Bihainim Angkol`. Folderul organizatorilor e cu un nivel mai sus, deci apare ca `../task_papua_new_guinea`.

## Ce se pune pe site

| Ce | Valoare |
|---|---|
| Țara și taskul | Papua Noua Guinee, Bihainim Angkol |
| Fișierul de urcat | `submission/main.py` |
| sha256 al fișierului | `d67f8f5d89468224588ca1215cafc8936ecf237bd463e07d6bad3fc7dc8bfd15` |
| Ce nu se urcă | `../task_papua_new_guinea/main.py`, șablonul organizatorilor |
| Flag | De forma `SIGMOID_<WORD>`. Apare pe platformă când fișierul trece. Încă nu îl știm. |
| Link de GitHub către descriere | Se cere pe site. Adresa nu apare în surse. |

Atenție la nume. Și în folderul organizatorilor există un `main.py`. Acela e șablonul gol: răspunde WAIT în orice situație. Barca nu pleacă de pe plajă și prinde 0 pești.

Verifici că ai fișierul bun cu sha256. Pe Windows: `certutil -hashfile submission\main.py SHA256`. Pe Linux: `sha256sum submission/main.py`. Trebuie să iasă șirul din tabel.

Starea după surse: `VERIFICARE.md`, din 15 noiembrie 2026, spune că fișierul nu fusese încă trimis. Ultimul rezultat de pe platformă era o eroare veche, din 14 noiembrie, ora 14:36. Nicio sursă nu spune că fișierul a fost între timp trimis sau acceptat.

## Ce cere taskul

Povestea se petrece în Kimbe Bay, provincia West New Britain. „Bihainim angkol” înseamnă „urmează-l pe unchiul” în limba Tok Pisin. Unchiul e un pescar. Primim jurnalul lui de pescuit ca exemplu.

Textul enunțului de pe site nu e salvat în folder. Regulile de mai jos vin din `../task_papua_new_guinea/lagoon.py`, simulatorul scris de organizatori. Punctajul și limitele tehnice vin din `DETALII.md`, care citează enunțul.

### Laguna

```
    0 1 2 3 4 5 6
 0  . . . . T T .     H  plaja: aici se descarcă peștele
 1  # # # # ~ # #     s  iarbă de mare: 20% șansă de pește la o încercare
 2  . . . . s s .     T  reciful din larg: 90% șansă de pește
 3  . . . . s s .     #  barieră: nu se poate trece
 4  . . . . . . .     ~  trecerea prin barieră: aici bate valul
 5  . . H . . . .     .  apă
```

Pozițiile se scriu (rând, coloană). Rândul 0 e spre larg, la nord. Rândul 5 e țărmul, la sud. Plaja e la (5,2). Harta are 42 de celule. Șase sunt barieră, deci barca poate sta pe 36.

### Regulile unei zile

1. O zi are 24 de pași. Barca pornește de pe plajă cu cala goală. În cală încap cel mult 6 pești.
2. La fiecare pas alegi o acțiune din șase. NORTH, SOUTH, EAST și WEST mută barca o celulă. FISH e o încercare de pescuit. WAIT înseamnă că barca stă.
3. Dacă mutarea ar duce în barieră sau în afara hărții, barca rămâne pe loc. Pasul se consumă oricum.
4. FISH aduce un pește cu șansă de 20% pe iarba de mare și de 90% pe recif. În alte celule nu aduce nimic. Cu cala plină nu aduce nimic.
5. Dacă barca termină pasul pe trecere, valul o împinge o celulă la sud în 40% din cazuri. Adică înapoi în lagună.
6. Dacă barca termină pasul pe plajă, toți peștii din cală se descarcă și se numără. Se întâmplă automat.
7. Peștii rămași în cală la sfârșitul zilei se pierd.

Norocul unei zile e fixat dinainte. Fiecare zi are un număr de pornire, numit seed. Din el se scot 24 de numere pentru val și 24 pentru pește, câte unul pe pas. Același seed cu aceleași decizii dă exact aceeași zi. Numerele nu depind de ce face barca.

### Ce trebuie predat

O stare e o fotografie a situației bărcii la un pas: rândul, coloana, câți pești are în cală și câți pași mai are ziua. În fișiere apare ca `row,col,hold,steps_left`. Sunt 36 de celule, 7 valori pentru cală și 24 pentru pașii rămași. În total 36 × 7 × 24 = 6048 de stări.

Predăm un singur fișier, `main.py`. Evaluatorul platformei îl rulează singur într-un folder, lângă un fișier `test-input.csv`. Programul trebuie să scrie `test-output.csv`. Intrarea are câte o stare pe rând. Ieșirea repetă starea și adaugă acțiunea:

```
test-input.csv              test-output.csv
row,col,hold,steps_left     row,col,hold,steps_left,action
5,2,0,24                    5,2,0,24,NORTH
```

Deci nu predăm o barcă ce se mișcă. Predăm un tabel de decizii: pentru fiecare stare, o acțiune. Evaluatorul plimbă barca după tabel. În cod și în documentele mai vechi tabelul se numește „policy” sau „politică”.

Șablonul avertizează că `main.py` rulează singur. Nu poate importa `lagoon.py` și nu poate citi jurnalul. Tot ce îi trebuie stă în fișier.

De la organizatori primim simulatorul, șablonul, lista stărilor, zilele de practică și jurnalul. Fișierele sunt descrise la „Unde e fiecare lucru”.

### Cum se dă nota

1. Evaluatorul joacă zile pe care nu le vedem, după tabelul nostru.
2. Calculează media peștilor aduși pe zi.
3. O împarte la media unchiului pe aceleași zile.
4. Dacă raportul e cel puțin 1.5, taskul trece și apare flag-ul.

Punctajul e 0 dacă `main.py` se oprește cu eroare sau depășește 60 de secunde. La fel dacă din ieșire lipsește o stare, dacă o stare apare de două ori sau dacă apare o acțiune necunoscută. Limite tehnice: Python 3.12, biblioteca standard plus numpy 2.2.6, fără internet.

## Ce am aflat din date

Datele sunt jurnalul unchiului, `../task_papua_new_guinea/dataset/logbook.csv`. Are 9600 de rânduri: 400 de zile a câte 24 de pași. Un rând spune ziua, starea, acțiunea unchiului și câți pești a descărcat la acel pas. Analiza o face `check/analyze_logbook.py`. Cifrele de mai jos sunt din `run_all_output.txt`, secțiunea 1.

### Jurnalul se poate rejuca exact

Problema: nu știam dacă `lagoon.py` e chiar simulatorul care a produs jurnalul. Dacă este, cunoaștem regulile complet.

1. Pentru fiecare zi din jurnal pornim simulatorul cu seed egal cu numărul zilei.
2. Îi dăm pe rând acțiunile unchiului.
3. La fiecare pas comparăm starea și peștii descărcați cu ce scrie în jurnal.

Au ieșit 0 zile cu diferențe din 400. Captura fiecărei zile e identică cu cea din `train-output.json`. Deci laguna din jurnal e cea din `lagoon.py`, iar seed-ul unei zile e numărul ei.

### Ce face unchiul

```
    0 1 2 3 4 5 6
 0  . . . . T T .
 1  # # # # ~ # #
 2  . . . . s s .
 3  . . > > X s .     X    locul unde pescuiește unchiul, (3,4)
 4  . . ^ . . . .     ^ >  drumul lui de dimineață
 5  . . H . . . .
```

1. Pleacă de pe plajă: de două ori NORTH, apoi de două ori EAST. Ajunge la (3,4), pe iarba de mare, când mai are 20 de pași.
2. Pescuiește acolo. Cel mai des pleacă spre casă când mai are 6 pași.
3. Descarcă de obicei când mai are 3 pași. Așa a fost în 284 de zile din 400. În 46 de zile nu a descărcat nimic.

Din cele 9600 de rânduri, 4996 îl arată la (3,4). Alte 346 îl arată la (3,5) și 305 la (2,4), tot pe iarba de mare. Nicio celulă din rândurile 0 și 1 nu apare în jurnal. Unchiul nu a intrat niciodată pe trecere și nu a ajuns niciodată la recif.

Din 5000 de acțiuni FISH, 4954 sunt pe iarba de mare. Restul de 46 sunt în apă goală sau pe plajă, unde nu se prinde nimic.

Unchiul nu e perfect consecvent. A trecut prin 422 de stări diferite. În 120 dintre ele a ales acțiuni diferite în zile diferite. La plecare, de exemplu, a mers la nord în 377 de zile din 400. În 10 zile a făcut FISH pe plajă. `check/verify.py` a măsurat abaterea: 4.758% din decizii diferă de acțiunea lui obișnuită. O acțiune la întâmplare nimerește acțiunea obișnuită o dată din șase. De aceea abaterea corespunde cu o acțiune la întâmplare cam la 5.7% din pași.

Capturile lui pe cele 400 de zile:

| Pești aduși într-o zi | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Număr de zile | 46 | 59 | 109 | 85 | 57 | 28 | 16 |

Media e 2.49 pești pe zi.

### Ce rezultă de aici

Unchiul stă într-un loc unde o încercare reușește în 20% din cazuri. La 3 pași mai la nord, dincolo de trecere, o încercare reușește în 90% din cazuri. Cine îl copiază rămâne la nivelul lui, iar ținta cere de 1.5 ori mai mult.

Jurnalul are o limită. Unchiul nu a fost niciodată la trecere sau la recif. Deci jurnalul nu poate confirma cele două cifre de acolo: 90% pentru pește și 40% pentru val. Ele vin doar din `lagoon.py`.

## Ce am încercat, în ordine

Ordinea e cea care reiese din surse. Dată exactă are doar prima trimitere.

### 1. Prima trimitere pe platformă a picat

- Ce s-a încercat: echipa a urcat un `main.py` pe 14 noiembrie 2026, la 14:36.
- Ce a ieșit: eroarea `ModuleNotFoundError: No module named 'environment'`. Programul importa un modul numit `environment`, care nu există la evaluator.
- Ce nu se știe: fișierul acela nu e în folder. Sursele nu spun ce metodă folosea.
- Ce a rămas din asta: `main.py` trebuie să conțină tot ce îi trebuie. Fișierul final importă doar `csv`.

### 2. Copierea unchiului: respinsă pe cifre

Numele taskului te îndeamnă să faci ca unchiul. În surse nu există niciun model antrenat pe jurnal. Ideea a fost respinsă din analiză, cu trei cifre:

- Unchiul aduce 2.49 pești pe zi. O copie a lui aduce cam la fel, deci raportul ar fi în jur de 1.
- Ținta cere 1.5 × 2.49 = 3.735 pești pe zi.
- `check/exact_check.py` a calculat limita stilului lui. Cel mai bun pescar posibil care nu trece niciodată bariera aduce 3.1647 pești pe zi. E sub 3.735.

Deci nicio strategie care rămâne în lagună nu atinge în medie ținta. Barca trebuie să iasă la recif.

### 3. Prima soluție: tabel calculat separat și împachetat în fișier

Se află în `varianta_veche/`.

- Ce s-a încercat: `varianta_veche/solve.py` calculează cea mai bună acțiune pentru fiecare stare. Metoda e programarea dinamică, explicată în secțiunea următoare. Scriptul folosește direct funcțiile din `lagoon.py`. La sfârșit cheamă `lagoon.export_main`, unealta organizatorilor. Ea scrie un `main.py` care are tabelul de decizii comprimat înăuntru. Tabelul e salvat și ca `varianta_veche/policy.csv`.
- Ce a ieșit, după `varianta_veche/README.md`: 5.963 pești pe zi pe zilele 1-400, adică 2.39x. Pe zilele 401-5400 au ieșit 5.965. Captura așteptată pentru o zi întreagă: 5.957. Noțiunea e explicată în secțiunea următoare.
- Prima diferență față de soluția finală: ce se întâmplă la egalitate. Când două acțiuni aduc la fel de mult, `solve.py` o ia pe prima din ordinea `FISH, NORTH, SOUTH, EAST, WEST, WAIT`. Urmarea se vede în `varianta_veche/policy.csv`. La recif, cu cala plină și 11 pași rămași, tabelul vechi spune FISH. Barca stă în larg deși nu mai poate prinde nimic. Captura nu scade, pentru că drumul spre casă are durată sigură.
- A doua diferență: `main.py` vechi citește intrarea strict, cu `int(...)`. O valoare scrisă `5.0` l-ar opri cu eroare.
- De ce a fost înlocuită: sursele nu dau motivul într-o frază. `VERIFICARE.md` compară cele două variante. Tabelele diferă în 2068 de stări din 6048, numai acolo unde două acțiuni au aceeași valoare. Captura așteptată e aceeași. Varianta veche a rămas în folder ca arhivă.

### 4. Soluția finală: calcul la fiecare rulare, cu reguli de departajare

Se află în `submission/main.py`.

- Ce s-a schimbat: calculul se face chiar în `main.py`, la fiecare rulare. Nu mai există tabel stocat. Singurul import e `csv`.
- Ce s-a adăugat: două reguli de departajare la egalitate și o citire tolerantă a intrării.
- Ce a ieșit: 5.963 pești pe zi pe zilele de practică, raport 2.395. Rularea durează 0.10 secunde.
- De ce a rămas: are aceeași captură așteptată ca varianta veche. Barca plină merge direct acasă. Fișierul nu depinde de nimic din afară. A trecut verificările din `run_all.py` și o verificare independentă, pe 15 noiembrie 2026.

## Soluția finală, pas cu pas

Regulile lagunei sunt cunoscute complet. Atunci nu trebuie învățat nimic din exemple. Cea mai bună acțiune din fiecare stare se poate calcula.

### Două noțiuni

**Captura așteptată** e media peștilor aduși pe plajă dacă ai relua aceeași situație de foarte multe ori, de fiecare dată cu alt noroc.

Un exemplu socotit de mână, direct din reguli. Barca e pe iarba de mare la (3,4), cu cala goală și 5 pași rămași. Drumul până la plajă are 4 pași: doi spre sud și doi spre vest. Dacă face FISH, prinde un pește în 20% din cazuri și îi rămân exact 4 pași de drum. Captura așteptată e 0.2 × 1 = 0.2 pești. Orice altă acțiune dă 0, pentru că nu mai rămâne timp și de pescuit, și de drum.

**Programarea dinamică** înseamnă că rezolvi întâi situațiile simple și le folosești la cele grele. Aici situațiile simple sunt cele de la sfârșitul zilei. De aceea calculul merge de la ultimul pas spre primul. În engleză metoda se mai numește „backward induction”.

### Calculul, de la sfârșitul zilei spre început

1. Cu 0 pași rămași nu mai poți aduce nimic. Fiecare situație valorează 0.
2. Cu 1 pas rămas încerci pe rând cele șase acțiuni. Pentru fiecare vezi câți pești ajung pe plajă. Exemplu socotit de mână: barca e la (5,3) cu 4 pești. WEST o duce pe plajă și descarcă 4. Orice altceva dă 0. Acolo acțiunea e WEST și valoarea e 4.
3. Cu 2 pași rămași faci la fel. După primul pas barca ajunge într-o situație cu 1 pas rămas. Valoarea aceleia o știi deja de la punctul 2. O aduni la ce s-a descărcat acum.
4. Repeți până la 24 de pași rămași. La fiecare rundă folosești doar valorile rundei dinainte.

Norocul intră așa: o acțiune poate avea mai multe urmări. FISH la recif dă un pește în 90% din cazuri și nimic în 10%. Valoarea acțiunii e media urmărilor, fiecare înmulțită cu șansa ei. Scris pe scurt:

```
valoarea unei stări cu t pași rămași =
    cea mai mare, dintre cele 6 acțiuni, din
    suma pe urmări de:  șansă × (pești descărcați acum + valoarea stării următoare, cu t-1 pași)
```

La sfârșit, valoarea stării de start este captura așteptată a unei zile întregi. Iese 5.957496 pești.

### Ce face `submission/main.py`, bucată cu bucată

1. **Constantele.** Harta și cifrele jocului, aceleași ca în regulile de mai sus. Sunt copiate din `lagoon.py`, pentru că `main.py` nu îl poate importa.
2. **`advance`.** Un pas în care norocul e deja hotărât. Intră poziția, cala și acțiunea. Mai intră două răspunsuri de tip da sau nu: dacă a mușcat peștele și dacă a împins valul. Ies noua poziție, noua cală și numărul de pești descărcați. E copia funcției `_advance` din `lagoon.py`.
3. **`outcomes`.** Toate urmările posibile ale unei decizii, cu șansele lor. Combină „mușcă sau nu” cu „împinge sau nu”. Ies cel mult 4 urmări.
4. **`solve`.** Programarea dinamică de mai sus. Pentru fiecare număr de pași rămași, de la 1 la 24, trece prin cele 36 de celule și cele 7 valori de cală. Calculează valoarea fiecărei acțiuni și o alege pe cea mai bună. Rezultatul e tabelul de decizii, cu 6048 de stări.
5. **`parse_state`.** Citește o stare dintr-un rând al intrării. Acceptă și valori scrise ca `5.0`. Dacă o valoare nu e număr, starea rămâne necunoscută.
6. **`main`.** Cheamă `solve`, citește `test-input.csv` și scrie `test-output.csv`. Păstrează rândurile în aceeași ordine. Pentru o stare necunoscută scrie WAIT.

Citirea tolerantă de la punctele 5 și 6 rezolvă o problemă concretă. Un `main.py` care se oprește cu eroare primește 0. De aceea programul nu se oprește nici la antet cu majuscule, nici la spații în plus, nici la valori stricate.

Tot calculul durează cam 0.1 secunde. De aceea nu a fost nevoie să stocăm tabelul în fișier.

### Regulile de departajare

Problema: în multe stări, mai multe acțiuni au aceeași captură așteptată. Exemplu: o barcă plină la recif, cu timp destul. Aduce 6 pești fie că pleacă acum, fie mai târziu. Fără o regulă, barca ar putea sta degeaba în larg. `solve` alege în trei trepte:

1. Captura așteptată cea mai mare. Două valori mai apropiate de 1e-9 una de alta se socotesc egale.
2. La egalitate câștigă acțiunea care aduce peștele pe plajă mai devreme. Pentru asta codul ține un al doilea scor, numit `early`. Fiecare pește descărcat se înmulțește cu numărul de pași rămași în acel moment. Un pește adus devreme primește un scor mai mare.
3. Dacă tot e egalitate, câștigă prima acțiune din ordinea `WAIT, NORTH, SOUTH, EAST, WEST, FISH`. Așa barca stă pe plajă când nu mai are nimic de câștigat.

Efectul se vede în ziua 1, descrisă mai jos. La recif, cu cala plină și 11 pași rămași, tabelul final spune SOUTH. Tabelul vechi spunea FISH.

Regulile de departajare nu costă niciun pește. `check/verify.py` a calculat captura așteptată cu ele și fără ele. A ieșit 5.957496 în ambele cazuri.

### De ce ajunge un tabel care vede doar starea

Tabelul nu vede seed-ul zilei și nu ține minte ce s-a întâmplat mai devreme. E destul, din două motive. Numerele de noroc nu depind de decizii. Numărul de la un pas nu are legătură cu cel de la alt pas. Deci trecutul nu spune nimic despre ce urmează. Tot ce contează pentru viitor e deja în stare.

### O zi întreagă: ziua 1

Traseul e cel din `DETALII.md`, obținut cu `lagoon.py --day 1`. Săgeata arată acțiunea făcută din acea celulă.

```
Dus                         Întors
    0 1 2 3 4 5 6               0 1 2 3 4 5 6
 0  . . . . X T .            0  . . . . v T .
 1  # # # # ^ # #            1  # # # # v # #
 2  . . > > ^ s .            2  . . . . v s .
 3  . . ^ . s s .            3  . . . . v s .
 4  . . ^ . . . .            4  . . . . v . .
 5  . . H . . . .            5  . . H < < . .

 X  locul de pescuit, (0,4)
```

| Pași rămași | Unde e barca | Pești în cală | Acțiune | Ce se întâmplă |
|---|---|---|---|---|
| 24 | (5,2), plaja | 0 | NORTH | urcă la (4,2) |
| 23 | (4,2) | 0 | NORTH | urcă la (3,2) |
| 22 | (3,2) | 0 | NORTH | urcă la (2,2) |
| 21 | (2,2) | 0 | EAST | merge la (2,3) |
| 20 | (2,3) | 0 | EAST | merge la (2,4), sub trecere |
| 19 | (2,4) | 0 | NORTH | intră pe trecere. Valul nu o împinge. |
| 18 | (1,4) | 0 | NORTH | iese la recif, (0,4) |
| de la 17 la 12 | (0,4) | crește de la 0 la 6 | FISH de 6 ori | prinde de fiecare dată |
| 11 | (0,4) | 6 | SOUTH | intră pe trecere. Valul o împinge până la (2,4). |
| 10 | (2,4) | 6 | SOUTH | coboară la (3,4) |
| 9 | (3,4) | 6 | SOUTH | coboară la (4,4) |
| 8 | (4,4) | 6 | SOUTH | coboară la (5,4) |
| 7 | (5,4) | 6 | WEST | merge la (5,3) |
| 6 | (5,3) | 6 | WEST | ajunge pe plajă și descarcă 6 pești |
| de la 5 la 1 | (5,2), plaja | 0 | WAIT | stă pe plajă |

Barca a adus 6 pești și a terminat cu 5 pași nefolosiți. În aceeași zi unchiul a adus 2 pești, prima valoare din `train-output.json`.

### Ce se schimbă în alte zile

- La intrarea pe trecere valul poate împinge barca înapoi la (2,4). Barca încearcă din nou.
- La recif pescuiește până umple cala. `varianta_veche/README.md` notează că pentru 6 pești e nevoie de cam 7 încercări.
- La întoarcere valul ajută, pentru că împinge spre casă. Drumul de la recif la plajă durează cel mult 7 pași. Fiindcă durata e sigură, barca știe exact până când poate pescui.
- Socoteala pașilor, numărați pe hartă: 7 la dus și cel mult 7 la întors. Rămân cel puțin 10 pași de pescuit dacă valul nu o întoarce la intrare.
- Barca nu rămâne cu pește pe mare. În 10000 de zile noi, cala a fost goală la sfârșitul fiecărei zile.

## Cât de bine merge

| Ce se măsoară | Pești pe zi | Raport |
|---|---|---|
| Unchiul, în jurnalul de 400 de zile | 2.490 | 1 |
| Cel mai bun pescar posibil care nu trece bariera | 3.1647 | |
| Soluția, captura așteptată calculată exact | 5.957496 | 2.393 față de unchi |
| Soluția, pe cele 400 de zile de practică | 5.963 | 2.395 față de unchi. Ținta e 1.5. |
| Soluția, pe 20000 de zile noi | 5.959 | 2.520 față de o copie a unchiului |
| Soluția față de cel mai bun pescar din lagună | | 1.883 |

Cum se citesc rândurile:

- 5.963 e media pe 400 de zile anume. 5.957496 e captura așteptată, adică media pe termen lung. Diferența mică dintre ele ține de norocul acelor zile.
- Rândul cu 400 de zile e chiar comanda oficială a organizatorilor. Ea afișează „above the target”.
- Zilele noi au seed de la 401 la 20400. Pentru ele nu avem jurnalul unchiului. `check/verify.py` îl înlocuiește cu o copie: în fiecare stare face acțiunea lui cea mai deasă din jurnal. La cam 5.7% din pași face o acțiune la întâmplare. Copia a adus 2.365 pești pe zi.
- Ultimul rând răspunde la o întrebare de siguranță. Dacă „unchiul” din evaluare ar fi un pescar perfect de lagună, tot am trece de 1.5.

Captura noastră pe cele 20000 de zile noi:

| Pești aduși într-o zi | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Număr de zile | 1 | 3 | 12 | 44 | 135 | 352 | 19453 |

Cala plină apare în 19453 de zile din 20000, adică 97.3%. Abaterea standard a capturii pe zi e 0.28 pești. Ea arată cât de mult diferă de obicei o zi de medie. Zilele slabe nu vin din pește pierdut, pentru că barca nu rămâne cu pește pe mare. După reguli, rămân două cauze posibile: timp pierdut la intrarea pe trecere și încercări ratate la recif.

`README.md` pune cifra de 97.3% lângă cele 400 de zile de practică. În `run_all_output.txt` ea iese din zilele noi, și așa e folosită aici. Câte zile de practică au avut cala plină nu apare în surse.

Raportul rămâne mare și pe grupuri mici de zile noi:

| Zile în grup | Cel mai mic raport găsit | Grupuri sub 1.5 |
|---|---|---|
| 20 | 1.739 | 0 din 1000 |
| 50 | 1.986 | 0 din 400 |
| 100 | 2.132 | 0 din 200 |
| 400 | 2.360 | 0 din 50 |

Tabelul de decizii e cel mai bun posibil pentru aceste reguli. În medie nu se poate aduce mai mult. Dovada e verificarea 5 din secțiunea următoare.

## Cum a fost verificată

Toate verificările pornesc cu o singură comandă, `run_all.py`. Ea rulează pe rând scripturile din `check/`. Rezultatul rulării noastre e salvat în `run_all_output.txt`. Scripturile folosesc seed-uri fixe. Oricine le rulează trebuie să obțină aceleași cifre. Diferă doar timpii și versiunea de Python.

1. **Jurnalul față de simulator** (`check/analyze_logbook.py`). Rejoacă cele 400 de zile în `lagoon.py`, cu seed egal cu ziua. Au ieșit 0 zile cu diferențe.
2. **Rulare ca la evaluator** (făcută de `run_all.py`). Copiază `main.py` într-un folder temporar. Lângă el pune doar `test-input.csv`, o copie a `train-input.csv`. Apoi rulează `python ./main.py`. Programul a terminat cu cod de ieșire 0, în 0.10 secunde. Verificarea acoperă chiar greșeala din prima trimitere: un fișier care are nevoie de altceva decât de el însuși. Cu `python3.12` ieșirea a fost identică.
3. **Comanda oficială** (`lagoon.py --compare`). Joacă cele 400 de zile de practică după tabelul nostru. A afișat 5.963 față de 2.490 și raportul 2.395.
4. **Calcul independent** (`check/verify.py`). Reface programarea dinamică pe funcția de pas din `lagoon.py`, nu pe copia din `main.py`. O greșeală la copierea regulilor ar apărea ca stări diferite. Au ieșit 0 din 6048. Validatorul organizatorilor, `check_policy_file`, caută stări lipsă, stări repetate și acțiuni necunoscute. Nu a găsit nimic. Tot aici se joacă cele 20000 de zile noi.
5. **Aritmetică exactă** (`check/exact_check.py`). Calculatorul lucrează de obicei cu zecimale rotunjite. Scriptul reface calculul cu fracții exacte: 1/5, 9/10 și 2/5. Cea mai bună captură așteptată posibilă iese 1489374012481/250000000000, adică 5.957496049924. A tabelului nostru e identică. Stări în care acțiunea aleasă nu e una dintre cele mai bune: 0 din 6048.
6. **Teste de stres** (`check/stress.py`). Trage grupuri mici din zilele de practică. Numără peștii rămași în cală la sfârșitul zilei, pe 10000 de zile noi. Joacă același tabel în lagune cu alte cifre. Peștii rămași în cală au fost 0 în toate zilele. Celelalte două rezultate sunt la „Ce nu e sigur”.
7. **Formatul ieșirii** (`check/format_parity.py`). Rulează și șablonul organizatorilor pe aceeași intrare. Antetul, coloanele de stare și ordinea rândurilor sunt identice cu ale noastre. Ambele ieșiri au 6049 de rânduri cu tot cu antet. Apoi unealta `lagoon.export_main` scrie un `main.py` din tabelul nostru. Ieșirea aceluia e identică octet cu octet cu a noastră.
8. **Intrări neobișnuite** (`check/input_robustness.py`). `main.py` primește nouă fișiere de intrare. Unul e normal. Altele au rânduri amestecate, doar 499 de stări sau doar antetul. Unul are BOM și CRLF, adică un semn invizibil la început și sfârșit de rând în stil Windows. Altele au antet scris altfel, spații în plus sau valori ca `5.0`. Ultimele au stări de pe barieră și valori `inf` sau `nan`. Programul a terminat de fiecare dată fără eroare. A scris câte un rând pentru fiecare rând primit.

### Verificarea independentă din 15 noiembrie 2026

`VERIFICARE.md` descrie un test făcut separat de scripturile din pachet. S-a lucrat pe Windows, cu Python 3.12.10 și 3.13.

- sha256 al `submission/main.py` e cel din tabelul de la început.
- Fișierele organizatorilor din pachet sunt identice cu cele descărcate direct. Lista lor de sha256 e în `DETALII.md`, secțiunea 8.
- Codul importă doar `csv`. Citește `test-input.csv` și scrie `test-output.csv`. Nu folosește rețeaua și nu atinge alte fișiere.
- Rulat singur într-un folder gol, a terminat cu cod de ieșire 0, în 0.09 secunde. Ieșirea e identică pe cele două versiuni de Python.
- Comanda oficială și captura așteptată au dat aceleași cifre ca mai sus: 2.395 și 5.957496.
- `run_all.py` a mers până la capăt. Toate cifrele au ieșit identice cu `run_all_output.txt`.

`DETALII.md` mai spune că ieșirea a fost comparată și pe Python 3.11, 3.12.3 și 3.12.13. A ieșit identică octet cu octet. Rularea salvată în `run_all_output.txt` e făcută cu Python 3.13.18.

## Ce nu e sigur

**1. Regulile zilelor ascunse.** Tot calculul presupune că zilele ascunse folosesc regulile din `lagoon.py`. `README.md` spune că jurnalul confirmă asta. Cifrele din `run_all_output.txt` arată că jurnalul acoperă doar zona în care a umblat unchiul. Șansa de 90% la recif și valul de 40% rămân luate din `lagoon.py`. De aceea `check/stress.py` a jucat același tabel de decizii în lagune cu alte cifre, pe câte 3000 de zile noi:

| Pește la recif | Val | Pește pe iarba de mare | Noi | Copia unchiului | Raport |
|---|---|---|---|---|---|
| 0.9 | 0.4 | 0.2 | 5.96 | 2.35 | 2.54 |
| 0.75 | 0.4 | 0.2 | 5.77 | 2.35 | 2.46 |
| 0.6 | 0.4 | 0.2 | 5.16 | 2.35 | 2.20 |
| 0.5 | 0.4 | 0.2 | 4.52 | 2.35 | 1.92 |
| 0.9 | 0.6 | 0.2 | 5.76 | 2.35 | 2.45 |
| 0.9 | 0.8 | 0.2 | 4.74 | 2.35 | 2.02 |
| 0.6 | 0.6 | 0.2 | 4.83 | 2.35 | 2.06 |
| 0.9 | 0.4 | 0.3 | 5.96 | 3.48 | 1.71 |

Primul rând are cifrele din `lagoon.py`. În toate cazurile încercate raportul rămâne peste 1.5. Cifre mai rele decât acestea nu au fost încercate.

**2. Câte zile are testul ascuns.** Nu se știe. `DETALII.md` presupune că sunt tot 400, ca la practică. Cu puține zile, norocul contează. `check/stress.py` a tras la întâmplare grupuri din cele 400 de zile de practică, cu capturile reale ale unchiului:

| Zile în test | 1 | 3 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| Șansa ca raportul să iasă sub 1.5 | 11.5800% | 4.2250% | 1.5700% | 0.1100% | 0.0000% | 0.0000% | 0.0000% |

Riscul acesta nu ține de tabelul nostru. Într-o zi în care unchiul aduce 5 sau 6 pești, nimeni nu poate aduce de 1.5 ori mai mult. În cală încap doar 6.

**3. Copia unchiului e aproximativă.** Pe zilele de practică aduce 2.433 pești pe zi, față de 2.490 cât a adus unchiul real. E deci puțin mai slabă, iar rapoartele calculate față de ea ies puțin mai mari. Contează doar la testele pe zile noi. Comanda oficială și testul cu grupuri mici folosesc capturile reale.

**4. Ce lipsește din surse.** Flag-ul: în `train-output.json` câmpul `flag` e gol și nicio sursă nu arată o trimitere reușită. Textul enunțului de pe site: limita de 60 de secunde, versiunea de Python și regula cu 1.5 sunt luate din `DETALII.md`. Fișierul primei trimiteri, cel cu eroarea din 14 noiembrie. Adresa de GitHub a descrierii.

## Unde e fiecare lucru

În folderul taskului:

| Fișier | La ce folosește |
|---|---|
| `submission/main.py` | fișierul de urcat pe site |
| `README.md` | descrierea scurtă, cea mai nouă |
| `DETALII.md` | descrierea lungă, mai veche. A fost scrisă pentru un pachet în care folderul organizatorilor stătea înăuntru. |
| `VERIFICARE.md` | ce a ieșit la verificarea independentă din 15 noiembrie 2026 |
| `EXPLICATIE.md` | documentul de față |
| `run_all.py` | pornește toate verificările, una după alta |
| `run_all_output.txt` | ce a afișat `run_all.py` la noi, pentru comparație |
| `check/analyze_logbook.py` | rejoacă jurnalul în simulator și descrie ce face unchiul |
| `check/verify.py` | calcul independent al tabelului, validatorul oficial, 20000 de zile noi |
| `check/exact_check.py` | verificarea cu fracții exacte și limita pentru un pescar care nu trece bariera |
| `check/stress.py` | teste cu puține zile, pește rămas în cală, lagune cu alte cifre |
| `check/format_parity.py` | compară formatul ieșirii cu șablonul și cu `lagoon.export_main` |
| `check/input_robustness.py` | dă lui `main.py` intrări stricate și verifică să nu se oprească |
| `varianta_veche/README.md` | descrierea primei soluții, în engleză |
| `varianta_veche/solve.py` | calculul primei soluții. Scrie cele două fișiere de mai jos. |
| `varianta_veche/main.py` | `main.py` vechi, cu tabelul de decizii comprimat înăuntru |
| `varianta_veche/policy.csv` | tabelul de decizii vechi, în formă lizibilă |

În folderul organizatorilor, `../task_papua_new_guinea`:

| Fișier | La ce folosește |
|---|---|
| `lagoon.py` | simulatorul. Tot el rejoacă o zi, compară cu unchiul, validează ieșirea și are `export_main`. |
| `main.py` | șablonul gol, care răspunde WAIT peste tot. Nu se urcă. |
| `train-input.csv` | cele 6048 de stări, ca intrare de probă |
| `train-output.json` | cele 400 de zile de practică: captura unchiului în fiecare zi, media lui 2.49, ținta 1.5 |
| `dataset/logbook.csv` | jurnalul unchiului, pas cu pas |

Mai există `../README.md`, care are doar titlul țării. Folderul `__pycache__` de lângă `lagoon.py` e făcut de Python la rulare și nu contează.

## Cum refaci totul

Ai nevoie doar de Python. Evaluatorul folosește 3.12. Scripturile folosesc numai biblioteca standard.

1. Deschide un terminal în folderul taskului.
2. Verifică sha256 al fișierului de trimis, cu comanda de la „Ce se pune pe site”.
3. Pornește toate verificările. Durează cam 10 secunde.

   ```
   python run_all.py "..\task_papua_new_guinea"
   ```

4. Compară ce s-a afișat cu `run_all_output.txt`. Toate cifrele trebuie să fie aceleași. Diferă doar timpii și versiunea de Python.

Argumentul de la pasul 3 e necesar în acest folder. `README.md` dă comanda fără argument. Atunci `run_all.py` caută folderul organizatorilor înăuntrul folderului taskului, unde nu există. Scriptul se oprește cu un mesaj care începe cu `missing`.

Verificarea de mână, ca la evaluator. Comenzile merg în Git Bash, pe Linux sau pe macOS:

```bash
mkdir -p /tmp/png
cp submission/main.py /tmp/png/
cp ../task_papua_new_guinea/train-input.csv /tmp/png/test-input.csv
(cd /tmp/png && python main.py)
python ../task_papua_new_guinea/lagoon.py --policy /tmp/png/test-output.csv --compare ../task_papua_new_guinea/train-output.json
```

Ultima comandă trebuie să afișeze:

```
400 days: your canoe 5.963 fish per day, angkol 2.490
ratio 2.395 x angkol (target 1.5 x): above the target
```

Aceleași comenzi apar și în `DETALII.md`, dar cu căile din pachetul vechi. Aici au `../` în față.

`lagoon.py` mai are două folosiri utile. Fără argumente desenează harta. Cu `--policy /tmp/png/test-output.csv --day 1` rejoacă ziua 1 pas cu pas.

Prima soluție se reface cu `python varianta_veche/solve.py`. Comanda rescrie `varianta_veche/main.py` și `varianta_veche/policy.csv`. Comenzile de verificare din `varianta_veche/README.md` au căi de dinainte de mutarea fișierelor în subfolder. Din `varianta_veche`, folderul organizatorilor e acum la `../../task_papua_new_guinea`.

Nu rula `python main.py` în `../task_papua_new_guinea`. Acolo `main.py` e șablonul gol.
