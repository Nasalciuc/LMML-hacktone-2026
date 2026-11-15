# Mozambic: The Professor of Maputo Bay. Explicația completă

Documentul explică taskul și tot ce s-a lucrat la el. E scris pentru un coleg care nu a făcut munca și trebuie să o poată povesti unui mentor.

Taskul nu e gata. Agentul ia 0.27 / 0.23 / 0.31 în arena oficială. Pragul e 0.65.

Numerele sunt scrise cu punct, exact ca în loguri. Căile de fișiere pornesc din folderul taskului. „Arena oficială” e scriptul `arena.py` al organizatorilor, rulat la noi pe hărți de antrenament. Nu e scorul de pe site.

## Ce se pune pe site

Nimic, deocamdată.

| Întrebare | Răspuns |
|---|---|
| Ce formă are predarea? | Un fișier, `main.py`. Șablonul organizatorilor spune „Upload only this file”. |
| Există un flag `SIGMOID_...`? | Sursele nu pomenesc niciun flag pentru acest task. |
| Ce se mai pune pe lângă fișier? | Linkul de GitHub către descriere. |
| S-a încărcat ceva până acum? | În surse nu scrie că s-a încărcat ceva. |
| De ce nu punem `main.py` acum? | Ia 0.27 / 0.23 / 0.31 pe seed 11 / 12 / 13. Pragul e 0.65. |

## Ce cere taskul

Enunțul oficial nu e în folder. Arhiva organizatorilor are șase fișiere. Trei sunt de cod: șablonul `main.py`, arena `arena.py` și motorul `mozambique-engine`. Celelalte sunt `LICENSE`, `train-input.json` și `train-output.json`. Regulile de mai jos vin din `arena.py`, din șablonul `main.py` și din copia noastră de reguli. Copia de reguli e codul scris de noi în `main.py` și în `work/rules_check.py`. Pragul de 0.65 apare doar în `README.md` și în `DETALII.md`.

### Jocul
Se joacă în doi pe o hartă. Noi jucăm contra unui program al organizatorilor, numit Profesorul. Fiecare adună materiale, face un atelier, fabrică produse și livrează contracte. Câștigă cine are mai multe puncte la final.

| Element | Ce știm |
|---|---|
| Harta | Locuri legate prin rute. Fiecare rută are o durată în runde. Scripturile noastre presupun 10 locuri. |
| Roluri de loc | Loc cu lemn, loc cu fibră, loc de atelier. Primele două le numim locuri naturale. |
| Acasă | Fiecare jucător are un loc de pornire, `home`. |
| Durata | 80 de runde. |
| Mutarea | O singură acțiune pe rundă pentru tot jucătorul. Nu câte una pe barcă. |
| Bărci | 2 la start, amândouă acasă. O barcă duce cel mult 3 unități. |
| Depozitul de acasă | 5 lemn și 3 fibră la start. |
| Stocul unui loc natural | 4 la start. Crește cu 1 la fiecare 4 runde, până la 6. |
| Ce se vede | Stocul unui loc apare doar când locul e vizibil. Altfel observația dă `None`. |
| Contracte | 8 pe hartă. Fiecare spune ce marfă cere, câtă și unde. Mai are un număr de puncte și un termen. |
| Contracte brute | Sloturile 0, 1, 2 și 3. Cer lemn sau fibră. În codul nostru valorează 5 puncte. |
| Contracte de produs | Sloturile 4, 5, 6 și 7. În codul nostru valorează 12 puncte. |
| Cine ia un contract | Cine livrează primul. După aceea contractul e închis pentru celălalt jucător. |

Rândurile despre bărci, depozit și stoc vin din copia noastră de reguli și din `research/solo.py`. Acțiunile de mai jos au numele din motorul organizatorilor. Condițiile sunt cele din `work/rules_check.py`.

| Acțiune | Ce face | Condiție |
|---|---|---|
| `wait` | Nimic. | Mereu permisă. |
| `sail` | Barca pleacă spre un loc vecin. | Drumul durează cât scrie pe rută. |
| `load` | Mută marfă din depozitul propriu pe barcă. | Loc liber pe barcă. |
| `unload` | Mută marfă de pe barcă în depozitul propriu. | Barca are marfa. |
| `gather` | Ia lemn sau fibră din stocul unui loc natural. | Stoc în loc și loc liber pe barcă. |
| `build_workshop` | Construiește atelierul. | 2 lemn și 1 fibră în depozitul din locul de atelier. |
| `craft` | Pornește un produs. | Atelier construit, niciun produs în lucru, 2 lemn și 1 fibră. |
| `deliver` | Livrează un contract. | Barca e la destinație cu marfa cerută. Contractul e liber și termenul nu a trecut. |
| `build_boat` | Face a treia barcă. | 4 lemn și 2 fibră în depozitul de acasă. Cel mult 3 bărci. |

În copia noastră de reguli un produs e gata la 3 runde după `craft`. Drumul unui produs, de la material la puncte:

```
loc cu lemn    --gather-->  barcă  --sail-->  loc de atelier  --unload-->  depozit
loc cu fibră   --gather-->  barcă  --sail-->  loc de atelier  --unload-->  depozit
depozit acasă  --load---->  barcă  --sail-->  loc de atelier  --unload-->  depozit

depozit (2 lemn + 1 fibră)  --build_workshop-->  atelier (o singură dată)
depozit (2 lemn + 1 fibră)  --craft, 3 runde-->  1 produs
produs  --load-->  barcă  --sail-->  destinație  --deliver-->  puncte
```

La finalul meciului motorul dă `scores` și `half_points`. `scores` e perechea noi, Profesorul. `half_points` e 2 la victorie, 1 la egal și 0 la înfrângere. Scorul cel mai mare din `train-output.json` e 68, adică 4 × 12 + 4 × 5.

### Cum vorbește programul cu evaluatorul
1. Evaluatorul pornește `python main.py`.
2. La fiecare decizie trimite pe stdin o linie JSON, `{"observation": ...}`.
3. Funcția `agent(observation, memory)` alege una din `observation['legal_actions']`. Acțiunea pleacă pe stdout, tot ca o linie JSON.
4. `memory` e un dicționar care se golește la runda 0 a fiecărui meci.

Se încarcă un singur fișier, scris doar cu biblioteca standard Python. Evaluatorul folosește PyPy 3.10, un interpretor Python mai rapid. Programul are 10 secunde de gândire pe meci. Primul meci al unui proces primește încă 10 secunde. La depășire arena avertizează că evaluatorul s-ar opri cu 0 puncte.

### Cum se dă nota
1. Arena joacă 24 de hărți. Harta cu numărul m folosește grupul m modulo 8.
2. Sunt 8 grupuri: 4 forme de hartă, fiecare jucată din ambele poziții de start.
3. Fiecare hartă se joacă de 5 ori, cu `game` de la 0 la 4.
4. Pentru o hartă se ia cel mai slab dintre cele 5 meciuri, adică cel mai mic `half_points`.
5. Nota e suma acestor valori împărțită la 2 × numărul de hărți.

Regula „cel mai slab din 5” înseamnă că o hartă aduce punctul întreg doar dacă toate cele 5 meciuri sunt câștigate. Dacă cel mai slab meci e un egal, harta aduce 0.5. O singură înfrângere o duce la 0.

Un exemplu din `research/arena_final.log`, a doua rulare. Agentul a câștigat 68 din 120 de meciuri. Doar 2 hărți au avut toate cele 5 meciuri câștigate. Alte 7 hărți au avut ca cel mai slab meci un egal. Nota: (2 × 2 + 7 × 1) / 48 = 0.2292. Pragul de 0.65, calculat din aceeași formulă, cere pe 24 de hărți cel puțin 32 de jumătăți de punct din 48. De exemplu 16 hărți câștigate complet.

## Ce am aflat din date

**Fișierele de exemplu.** `train-input.json` are 64 de meciuri, fiecare descris prin `group`, `seed` și `game`. `train-output.json` are aceleași meciuri jucate de „starter main.py”, care alege mereu prima acțiune legală. În toate cele 64 agentul de start are scor 0 și `half_points` 0. Profesorul are între 39 și 68 de puncte, cel mai des 56. Deci un agent care nu face nimic util pierde tot.

**Profesorul lăsat singur, 400 de hărți.** `work/probe_wait.py` joacă `wait` în fiecare rundă și notează când ia Profesorul fiecare contract. `research/an_wait.py` rezumă 400 de hărți a câte 5 meciuri în `research/an_wait.log`.

| Ce s-a măsurat | Valoare |
|---|---|
| Scor mediu al Profesorului, pe fiecare din cele 5 meciuri | 54.3 / 54.2 / 54.8 / 53.9 / 54.1 |
| Scor mediu pe cele 4 forme de hartă | 51.4 / 55.9 / 54.3 / 55.4 |
| Contracte de produs luate pe meci | între 3.2 și 3.24 |
| Contracte brute luate pe meci | între 3.03 și 3.15 |
| Runda primului contract de produs | medie între 30.0 și 30.9, minim 17 sau 18 |
| Hărți cu 5 / 4 / 3 programe de livrare diferite în cele 5 meciuri | 267 / 123 / 10 |

Ultimul rând spune că Profesorul nu joacă la fel în cele 5 meciuri de pe aceeași hartă. Primul rând spune că niciunul din cele 5 meciuri nu e mai ușor în medie. Pe sloturi de contract:

| Sloturi | Tip | În câte meciuri le ia Profesorul | Runda mediană |
|---|---|---|---|
| 0 / 1 / 2 / 3 | brut | 53% / 67% / 91% / 98% | 8 / 16 / 25 / 35 |
| 4 / 5 / 6 / 7 | produs | 52% / 79% / 94% / 97% | 31 / 42 / 49 / 56 |

**Profesorul lăsat singur, 10000 de meciuri.** `research/prof_collect.py` a strâns încă meciuri de același fel. `research/RESULTS.txt` dă rundele la care Profesorul livrează fiecare produs. „p5” e runda până la care se întâmplă în cele mai rapide 5% din meciuri. Celulele goale nu sunt în surse.

| Livrarea de produs a Profesorului | Minim | p5 | p10 | Mediană | Apare în |
|---|---|---|---|---|---|
| prima | 17 | 20 | | 30 | |
| a doua | 23 | 28 | 30 | 41 | |
| a treia | | 44 | | 57 | 89% din meciuri |
| a patra | | | | | 34% din meciuri |

**Tabelele de supraviețuire.** `research/mk_tables.py` face din cele 10000 de meciuri tabelul `research/tables.txt`. Intră slotul contractului și o distanță. Iese un șir de 81 de valori, câte una pe rundă. Fiecare valoare e șansa, la mie, ca Profesorul să nu fi livrat încă acel contract. La contractele brute distanța e drumul de la casa Profesorului la destinație. La cele de produs e drumul cel mai scurt de la casa lui printr-un loc de atelier până la destinație. `research/tables.info` are un rezumat. Câteva rânduri din el:

| Slot | Distanță | Liber la runda 20 | La runda 35 | La runda 50 | La final |
|---|---|---|---|---|---|
| 0 (brut) | 1 | 206 | 182 | 182 | 182 |
| 0 (brut) | 9 | 906 | 886 | 886 | 886 |
| 4 (produs) | 7 | 995 | 618 | 473 | 473 |
| 7 (produs) | 7 | 999 | 852 | 592 | 15 |

Contractul brut din slotul 0, aflat la distanța 1 de casa Profesorului, mai e liber la runda 20 doar în 206 meciuri din 1000. Slotul 7 la distanța 7 rămâne liber până la final doar în 15 meciuri din 1000.

**Meciul se repetă identic.** `research/sweep_ebud.log` a jucat același meci de 3 ori, de fiecare dată cu proces nou. Scorul a ieșit la fel: 34 la 34 la un meci și 46 la 22 la altul. Agentul folosit acolo nu depindea de ceas. Deci același agent pe același meci dă același rezultat.

**Ce nu se știe.** Nu se știe dacă Profesorul își schimbă jocul după mutările noastre. Există două scripturi pentru asta, `work/probe_react.py` și `research/react_probe.py`, dar rezultatul lor nu e salvat. Nu se știu nici hărțile evaluării reale. Arena generează hărți de antrenament din seed, un număr care fixează hărțile.

## Ce am încercat, în ordine

Toate notele din această secțiune sunt note pe hartă, cu regula „cel mai slab din 5”. Sunt măsurate cu verificarea noastră, `work/evalx.py`. Pentru ultimele variante s-a folosit `research/evalx2.py`, care calculează nota la fel. După `research/RESULTS.txt`, rulările sunt făcute cu PyPy 3.10 în WSL. S-au folosit două seturi de hărți: seed 11 cu 96 de hărți și seed 22 cu 192 de hărți.

### 1. Prima versiune, numită b11
E agentul din `main.py`. Caută planuri de 14 runde pe copia de reguli și e descris pas cu pas mai jos. Pe 96 de hărți a luat 0.40 la prima măsurătoare și 0.281 la una mai nouă. Diferența e discutată la „Cât de bine merge”. A rămas în `main.py` până azi.

### 2. Reglaje de parametri pe prima versiune
Patru loguri vechi au câte 96 de hărți pe rând. Seed-ul nu e scris în ele.

| Log | Etichetă și notă |
|---|---|
| `research/sweep1.log` | g95 0.224, g99 0.281 |
| `research/sweep2.log` | kb6 0.375, kb6b5 0.349, h20 0.276, kp95 0.266 |
| `research/sweep3.log` | kb75 0.234, kb6nu8 0.260, kb6u4 0.354, kb6kr1 0.323 |
| `research/sweep9.log` | base11 0.396, rs0 0.344 |

Legenda etichetelor nu e scrisă nicăieri. Ele seamănă cu numele parametrilor din `main.py`. De exemplu g cu `gamma`, kb cu `kb` și h cu `H`. Etichetele `b5`, `u4` și `rs0` nu se pot lega sigur de un parametru. În `main.py` au rămas `gamma` 0.99, `kb` 0.6 și `H` 14. Un orizont mai lung, de 20 de runde, a ieșit mai slab: 0.276.

### 3. c1: a treia barcă
Ideea era că trei bărci cară mai mult. Barca se construia de la o rundă aleasă. Pe seed 11, cu runda 0 / 12 / 24, nota a fost 0.130 / 0.271 / 0.281. Prima versiune avea 0.281 pe aceleași hărți. Nu a adus nimic și a fost lăsată deoparte. Sursele nu spun de ce barca de la runda 0 strică atât de mult.

### 4. c2: contracte brute la jumătate și un tabel de timpi ai Profesorului
Problema: agentul cheltuia mutări pe contracte brute de 5 puncte, iar meciul se decide la cele de produs, de 12 puncte. Prima idee: în căutare un contract brut valorează doar o parte din punctele lui. Punctele reale nu se schimbă. Cu factor 0.5 nota a fost 0.370, cu 69% meciuri câștigate. Cu factor 0.2 a fost 0.349. Factorul 0.5 a rămas în variantele următoare.

A doua idee: un tabel cu runda la care Profesorul ia de obicei fiecare slot. Valorile sunt medianele de mai sus. O livrare a noastră mai târzie decât runda din tabel valora mai puțin, cu factor 0.5 sau 0.3. Nota a fost 0.281 în ambele cazuri. Împreună cu brute la 0.5 a fost 0.323, sub 0.370. Ideea a fost lăsată deoparte. `research/sweep_c2.log` are doar un antet, deci cifrele pentru c2 sunt numai în `RESULTS.txt`.

### 5. c3: deschidere fixă
Problema: primul produs iese târziu. S-a scris o deschidere fără căutare. Bărcile duc lemn și fibră de acasă direct la locul de atelier. Acolo construiesc atelierul și pornesc primul produs. Apoi preia căutarea.

| Setare (seed 11) | Notă | Meciuri câștigate |
|---|---|---|
| fără deschidere fixă | 0.370 | 331/480 |
| cu deschidere fixă | 0.214 | 279/480 |
| cu deschidere fixă, `gm` 0.96 | 0.161 | 259/480 |
| cu deschidere fixă, `gm` 0.93 | 0.156 | 216/480 |
| fără deschidere fixă, `gm` 0.96 | 0.380 | 302/480 |

`gm` e o reducere pe rundă aplicată doar punctelor din contractele de produs. Cu cât e mai mică, cu atât agentul se grăbește mai tare la produse. Deschiderea fixă a ieșit clar mai slabă și a fost oprită.

### 6. c4: alte valori pentru contractele brute
Tot pe seed 11, din `research/sweep_c4.log`:

| Setare | Notă |
|---|---|
| brute la 0.35 / la 0.7 | 0.328 / 0.323 |
| brute la valoare întreagă de la runda 40 / 55 | 0.354 / 0.380 |
| `gm` 0.98 | 0.396 |
| brute la 0.35, cu valoare întreagă de la runda 50 | 0.292 |
| brute la 0.35, cu `gm` 0.96 | 0.292 |

### 7. Confirmarea pe hărți noi
Pe seed 11 păreau mai bune `gm` 0.98 și valoarea întreagă de la runda 55. Dar setările fuseseră alese pe aceleași 96 de hărți pe care erau măsurate. De aceea au fost rejucate pe 192 de hărți noi, seed 22 (`research/confirm1.log`). Toate rândurile c4 au brute la 0.5.

| Variantă | Notă | Meciuri câștigate | Egaluri | Puncte noi / Profesor |
|---|---|---|---|---|
| b11, adică `main.py` | 0.302 | 541/960 | 179 | 35.2 / 29.9 |
| c4 | 0.344 | 652/960 | 77 | 36.1 / 28.3 |
| c4, `gm` 0.98 | 0.331 | 626/960 | 98 | 35.7 / 28.9 |
| c4, brute întregi de la runda 55 | 0.276 | 616/960 | 68 | 35.2 / 29.2 |
| c4, ambele | 0.281 | 598/960 | 85 | 35.1 / 29.5 |

Cele două câștiguri de pe seed 11 au dispărut. `RESULTS.txt` le numește zgomot. A rămas un singur câștig real: brute la 0.5, de la 0.302 la 0.344.

### 8. c5: departajare la egal
Problema: multe meciuri se termină la egal. De la o rundă aleasă, contractele brute valorează iar întreg dacă nu suntem în față la scor. Pe seed 22, cu runda 50 / 60 / 70, nota a fost 0.339 / 0.349 / 0.258. Ultima rulare a fost tulburată de încărcarea mașinii. Față de 0.344 nu e un câștig, deci regula a rămas oprită.

### 9. final și final2: gardă de timp și buget fix
`research/final.py` e c5 cu setările măsurate și fără experimentele oprite. Are în plus o gardă de timp. După 8 secunde folosite într-un meci nu mai caută. Urmează planul vechi, livrează ce poate sau așteaptă. `research/final2.py` schimbă felul în care se măsoară bugetul de căutare. Problema: bugetul era în secunde, deci nota se schimba când mașina era ocupată. Soluția: bugetul se numără în stări încercate. Testul de repetare descris mai sus e făcut cu această variantă.

Prima rulare pe 192 de hărți a dat 0.073 / 0.047 / 0.167 pentru 40000 / 60000 / 80000 de stări pe meci. Mașina era supraîncărcată, cu meciuri de până la 33.94 secunde, iar garda de 8 secunde era pornită. A doua rulare a oprit garda (`research/sweep_ebud2.log`). Cu 60000 de stări nota a fost 0.323, cu 633/960 meciuri câștigate. Cu 150000 a fost 0.359, cu 652/960. E cam la nivelul lui c4. `final2` nu a fost copiat peste `main.py`, iar sursele nu spun de ce.

### 10. p1: căutare cu valori din tabelele de supraviețuire
`research/p1_body.py` păstrează căutarea, dar schimbă valoarea unui contract. Valoarea devine câștigul așteptat de scor dacă livrăm la o rundă anume. Se calculează din tabelele de supraviețuire. Pentru p1 nu există nicio notă în `RESULTS.txt` sau în loguri.

### 11. p2: planificator cu rollout
Un rollout e un joc de probă dus până la capăt cu reguli simple. Se joacă pe copia de reguli, ca să se vadă cât valorează o alegere.

```
starea de acum + câte un obiectiv pentru fiecare barcă
        |
        v
pentru fiecare barcă liberă se încearcă fiecare obiectiv posibil
        |
        v
rollout: restul meciului, până la runda 80, jucat cu reguli simple
        |
        v
valoarea de la final  -->  rămâne obiectivul cu valoarea cea mai mare
```

O barcă poate avea cinci obiective. Două țin de materiale: adu materiale și du-le la atelier. Unul e așteptarea la atelier. Două țin de livrări: livrează un produs sau servește un contract brut. Un obiectiv nou trebuie să bată planul vechi cu cel puțin 0.05.

Valoarea unei livrări la runda t are două părți. Prima: punctele noastre, înmulțite cu șansa ca acel contract să mai fie liber la runda t. A doua: punctele pe care Profesorul le-ar fi luat mai târziu și nu le mai ia. Șansele vin din tabelele de supraviețuire.

Rezultat pe seed 22: nota 0.354, 62% meciuri câștigate, 170 de egaluri. Timpul median de gândire a fost 0.6 secunde pe meci.

### 12. p3: p2 cu trei adaosuri
- Regulile simple aduc materiale pentru cel mult 2 loturi odată. Construirea atelierului se numără ca un lot.
- Când două bărci pot muta, are prioritate cea cu drumul mai lung.
- Când două bărci sunt libere, se încearcă și perechi de obiective. Se combină cele mai bune 5 ale fiecăreia.

Rezultat pe seed 22: nota 0.391, 66% meciuri câștigate, 166 de egaluri. E cea mai mare notă măsurată pe hărți noi. Are două probleme. Prima e timpul. Timpul median e 1.2 secunde, dar 1% din meciuri trec de 12 secunde, peste limita de 10. Măsurătoarea s-a făcut cu garda de timp oprită. O notă cu garda pornită nu apare în surse. A doua problemă: fabrică și mai încet. Al doilea produs iese la runda 40. Al treilea apare doar în 18% din meciuri. După `RESULTS.txt`, victoriile lui vin din contracte brute: 2 de produs și 3 brute. Nici p3 nu a fost copiat peste `main.py`.

## Agentul de acum, pas cu pas

`main.py` e prima versiune, b11. Nu are factorul 0.5 la contractele brute și nu are garda de timp.

```
observația (JSON)
      |
      v
[1] memoria meciului  -->  [2] starea, ca listă de numere
                                  |
                                  v
[4] nota unei stări   -->  [3] căutarea: 14 runde înainte, cele mai bune 24 de planuri
                                  |
                                  v
                           [5] prima acțiune din cel mai bun plan  -->  răspuns JSON
```

1. **Memoria meciului.** La runda 0 agentul calculează drumul cel mai scurt între oricare două locuri. Costul unui drum e numărul de runde pe apă plus numărul de segmente. În fiecare rundă notează stocurile pe care le vede. Pentru un loc nevăzut presupune 4 la start și plus 1 la fiecare 4 runde, până la 6.
2. **Starea.** Observația e strânsă într-o listă de numere. La început stau runda și valoarea adunată. Apoi vin atelierele și contractele închise. Urmează fiecare barcă cu poziția, ținta, rundele rămase și marfa. La urmă vin depozitele, produsele în lucru și stocurile estimate.
3. **Căutarea.** E o căutare în fascicul, în engleză beam search. Adică la fiecare pas se păstrează doar cele mai bune câteva planuri, nu toate.
   - Un plan e un șir de acțiuni proprii, câte una pe rundă.
   - Din fiecare stare păstrată se încearcă toate acțiunile proprii posibile. Fiecare e aplicată pe copia de reguli, apoi trece o rundă.
   - Fiecare stare nouă primește o notă, la pasul 4. Stările identice se unesc.
   - Rămân cele mai bune 24. Acesta e parametrul `W`, lățimea.
   - Totul se repetă de 14 ori. Acesta e parametrul `H`, orizontul, adică numărul de runde pe care agentul le vede înainte.
   - Planul din runda trecută e ținut în listă chiar dacă iese din primele 24. Așa agentul nu își schimbă planul fără motiv.
4. **Nota unei stări.** E o sumă de cinci părți. Partea de materiale scade spre zero în ultimele 25 de runde.

   | Parte | Cum se calculează |
   |---|---|
   | Contracte livrate | Punctele contractului, înmulțite cu 0.99 pentru fiecare rundă de așteptare. Acest factor e „reducerea” din rândurile următoare. |
   | Produse care există | Pentru fiecare produs se caută cel mai bun contract de produs liber la care ajunge la timp. Valorează 12 × reducerea × 0.85. |
   | Marfă brută pe barcă | Dacă acoperă un contract brut la care ajunge la timp, valorează 5 × reducerea × 0.8. |
   | Atelier construit | 3 puncte. |
   | Materiale | Fiecare unitate de lemn sau fibră primește un factor care scade cu distanța până la atelier. Din ele se estimează câte produse mai ies. Un produs estimat valorează 0.6 × 12 × reducerea. |

5. **Alegerea.** Agentul ia starea cu nota cea mai mare de la capătul căutării. Joacă prima acțiune din planul ei și ține minte planul. `README.md` spune că alege planul cu cele mai multe puncte livrate. Mai exact, alege după nota de la pasul 4, care numără și produsele și materialele.
6. **Locul atelierului.** Se alege la runda 0, dacă harta are mai multe locuri de atelier. Agentul face câte o căutare pentru fiecare loc. Rămâne locul al cărui cel mai bun plan are nota mai mare.
7. **Timpul.** Bugetul e 6 secunde pe meci. O decizie primește partea ei egală din ce a rămas, înmulțită cu 1.5. Când timpul trece, căutarea se oprește mai devreme, dar nu înainte de 3 pași.
8. **Plasa de siguranță.** Dacă acțiunea aleasă nu e în lista legală sau codul dă eroare, agentul joacă `wait`.

Ce nu face agentul de acum:

- Nu joacă în copia de reguli și mutările Profesorului. Scoate doar contractele deja luate.
- Are o estimare a rundei la care Profesorul ar livra fiecare contract. Factorul ei se numește `risk` și e 1.0, deci nu schimbă nimic.
- Nu construiește a treia barcă și nu folosește câmpurile `visible`, `enemy_boats` și `opponent_score` din observație.

## Cât de bine merge

Arena oficială, 3 rulări cu `main.py` (`research/arena_final.log`). După `DETALII.md` ordinea e seed 11, 12 și 13.

| Seed | Hărți cu 5 victorii | Hărți cu cel mai slab meci egal | Notă | Meciuri câștigate | Cel mai lent meci |
|---|---|---|---|---|---|
| 11 | 4 din 24 | 5 | 0.2708 | 62 din 120 | 4.46 s |
| 12 | 2 din 24 | 7 | 0.2292 | 68 din 120 | 3.52 s |
| 13 | 5 din 24 | 5 | 0.3125 | 65 din 120 | 2.46 s |

Verificarea noastră pe 96 de hărți: nota pe hartă 0.40, 63% din meciurile luate câte unul câștigate, în medie 36 de puncte la 29. Linia `base11` din `research/sweep9.log` are exact aceste cifre: 0.396, 304/480, 36.0 la 28.7.

Sursele nu sunt de acord între ele aici. `RESULTS.txt` e mai nou și dă pentru același `main.py` note mai mici: 0.281 pe seed 11 și 0.302 pe 192 de hărți noi. Explicația din `RESULTS.txt` e că bugetul în secunde face nota să varieze cu încărcarea mașinii. Seed-ul rulării de 0.40 nu e scris, deci nu se știe dacă erau aceleași hărți. Cele trei note din arenă sunt mai aproape de cifrele mai noi.

Pe grupuri, pe 192 de hărți, nota lui `main.py` e între 0.19 și 0.40 (`research/confirm1.log`). Niciun grup nu e aproape de 0.65, deci problema nu ține de o singură formă de hartă. Cel mai lent meci a avut 4.5 secunde de gândire pe CPython, față de limita de 10. Toate logurile de verificare arată `illegal 0`, adică nicio acțiune nepermisă.

## Cum a fost verificat

1. **Copia de reguli față de motor.** `work/rules_check.py` joacă meciuri la întâmplare. În fiecare rundă compară lista de acțiuni legale calculată de noi cu lista dată de motor. Rezultatul lui nu e salvat. Dovada indirectă e `illegal 0` din toate logurile.
2. **Arena oficială.** `arena.py` rulează `main.py` ca evaluatorul: același protocol, 5 meciuri pe hartă, cel mai slab meci. S-a rulat cu seed 11, 12 și 13. Toate trei au `within_time_limit: true`.
3. **Verificarea noastră.** `work/evalx.py` calculează nota după aceeași formulă ca arena, dar încarcă agentul direct și joacă multe hărți în paralel. Tipărește în plus meciurile câștigate, egalurile și punctele medii. Mai dă cel mai lent meci și nota pe grup. Poate schimba parametri din linia de comandă, sub forma `CHEIE=VALOARE`.
4. **Hărți noi pentru confirmare.** Reglajele s-au făcut pe seed 11. Confirmarea s-a făcut pe seed 22, cu hărți nefolosite la reglaj. Așa au fost prinse două câștiguri false.
5. **Analiza cursei.** `research/race_stats.py` citește rezultatele salvate pe meci. Scoate runda la care noi și Profesorul livrăm fiecare produs și cum se împart contractele de produs. De aici vin cifrele din secțiunea următoare.

Limitele verificării:

- Nota pe 24 de hărți se mișcă mult de la un seed la altul. Cele trei rulări au dat între 0.23 și 0.31.
- Rulările făcute cu mașina ocupată nu se pot compara cu celelalte. Două sunt marcate așa: c5 cu runda 70 și prima rulare a lui `final2`.
- Fișierele cu rezultate pe meci, `res_*.json`, nu sunt în folder. Cifrele cursei nu pot fi recalculate fără o rulare nouă.

## De ce e sub prag și ce ar trebui făcut

**Regula de notare cere toate cele 5 meciuri.** Agentul câștigă mai mult de jumătate din meciurile luate separat. Pe cele mai multe hărți are însă cel puțin un egal sau o înfrângere, iar atunci harta aduce 0.5 sau 0.

**Aritmetica meciului.** Sunt 4 contracte de produs a câte 12 puncte și 4 brute a câte 5. Dacă luăm 3 contracte de produs din 4, câștigăm sigur. Avem cel puțin 36, iar Profesorul are cel mult 12 + 20 = 32. Dacă ele se împart 2 la 2, decid contractele brute. Când și acelea se împart 2 la 2, iese egal, 34 la 34. Deci o victorie sigură cere ca al treilea produs al nostru să fie livrat înaintea celui de-al doilea al Profesorului.

**Cursa, în cifre.** Sunt din `RESULTS.txt`. Rundele noastre sunt măsurate pe `final2`, nu pe `main.py`. Pentru `main.py` aceste runde nu sunt notate.

| Livrare | Runda mediană |
|---|---|
| Primul nostru produs | 29 |
| Al doilea al nostru | 36 |
| Al treilea al nostru | 42, și apare doar în 51% din meciuri |
| Al doilea al Profesorului, lăsat singur | 41, iar în cele mai rapide 10% din meciuri până la runda 30 |

Al treilea produs al nostru și al doilea al Profesorului ies cam în aceeași rundă. Rezultatul e o monedă aruncată: 3 la 1 în 42% din meciuri, 2 la 2 în 42% și 4 la 0 în 7%. O monedă aruncată de 5 ori pe hartă nu trece de regula „cel mai slab din 5”. De aici multele egaluri 34 la 34 și înfrângerile la un singur contract. `README.md` spune că al treilea produs ar trebui să iasă în jurul rundei 32, nu 42. Calculul țintei nu apare în surse.

**Unde se pierd rundele.** Nu se știe. Sursele nu au o defalcare a timpului dintre start și al treilea produs. Se știu doar constrângerile: un produs cere 2 lemn și 1 fibră, iar o barcă duce 3 unități. Atelierul face un singur produs odată și jucătorul are o singură acțiune pe rundă. Dintre încercări, a treia barcă, deschiderea fixă și graba mai mare la produse nu au ajutat. Rollout-ul a ridicat nota, dar fabrică și mai încet.

**Ce ar trebui făcut.** În surse nu există un plan scris. Există ținta și câteva lucruri care decurg din măsurători.

1. Orice versiune nouă se judecă întâi după runda celui de-al treilea produs. Ținta e runda 32. `research/race_stats.py` scoate această rundă din rezultatele unei rulări.
2. Pentru lucrul la viteza de fabricare există `research/solo.py`. Joacă agentul singur pe hărți reale, fără motor, și spune la ce rundă livrează fiecare produs. E scris pentru variantele cu rollout. Rezultatul lui nu e salvat.
3. p3 are nevoie de o limită de timp înainte de orice rulare în arenă. Apoi trebuie măsurat din nou, cu limita pornită.
4. Comparațiile se fac pe 192 de hărți noi și cu buget fix. Pe 96 de hărți două câștiguri s-au dovedit zgomot.
5. Nu se pune nimic pe site până când arena nu dă peste 0.65 pe mai multe seed-uri.

## Unde e fiecare lucru

Fișierele organizatorilor sunt în `../task_mozambique/participant/`. Restul e în folderul taskului.

| Unde | Fișiere | La ce folosește |
|---|---|---|
| rădăcină | `main.py` | Agentul de acum, prima versiune. |
| rădăcină | `README.md`, `DETALII.md`, `EXPLICATIE.md` | Descrierea scurtă, șase rânduri de note vechi și acest document. |
| `work/` | `evalx.py` | Verificarea noastră: N hărți a câte 5 meciuri, cel mai slab meci pe hartă. |
| `work/` | `rules_check.py` | Compară copia de reguli cu motorul. |
| `work/` | `probe_wait.py`, `probe_vis.py`, `probe_watch.py`, `probe_react.py` | Probe despre Profesor. Arată ce livrează singur și ce se vede din joc. Arată și unde îi sunt bărcile și dacă reacționează. |
| `research/` | `RESULTS.txt` | Jurnalul notelor măsurate, cu cele mai noi jos. |
| `research/` | `c1.py`, `c2.py`, `c3.py`, `c4.py`, `c5.py`, `final.py`, `final2.py` | Variante cu căutare. Agenți întregi, fiecare făcut din cel dinainte. |
| `research/` | `mk_c2.py`, `mk_c3.py`, `mk_c4.py`, `mk_c5.py`, `mk_final.py`, `mk_final2.py` | Generatoare. Fac varianta următoare prin înlocuiri de text. |
| `research/` | `p1_body.py`, `p2_new.txt`, `p3_new.txt`, `p2_body.py`, `p3_body.py`, `p2.py`, `p3.py` | Variante cu rollout: corpul comun, părțile noi și agenții asamblați. |
| `research/` | `mk_p2.py`, `assemble.py` | Lipesc corpul, părțile noi și tabelele într-un singur fișier. |
| `research/` | `_p2_parts.txt` | Părțile comune scoase din `p1_body.py`. Niciun script din folder nu îl citește. |
| `research/` | `prof_probe.py`, `prof_collect.py`, `prof_summary.py`, `prof_model1.py`, `an_wait.py`, `react_probe.py` | Studiul Profesorului: strâng și rezumă meciuri în care joacă singur. |
| `research/` | `mk_tables.py`, `tables.txt`, `tables.info` | Tabelele de supraviețuire și rezumatul lor. |
| `research/` | `evalx2.py`, `ev.sh` | Ca `evalx.py`, plus stări încercate și timp pe decizie. |
| `research/` | `race_stats.py`, `fail_mix.py`, `decode.py`, `solo.py`, `trace.py`, `trace_full.py` | Analiză: cursa produselor, de ce pică o hartă, urma unui meci. |
| `research/` | `sweep_c2.sh`, `sweep_c3.sh`, `sweep_c4.sh`, `sweep_c5.sh`, `sweep_ebud.sh`, `sweep_ebud2.sh`, `confirm1.sh` | Rulări în serie: mai multe setări la rând. |
| `research/` | fișierele `.log` | Ieșirile rulărilor. `arena_final.log` are cele 3 rulări din arenă. |
| `research/` | `analyze.py` | Rătăcit. Ține de alt task, cel din Japonia. |

Cum decurg variantele una din alta:

```
main.py --> c1.py --> c2.py --> c3.py --> c4.py --> c5.py --> final.py --> final2.py
                 mk_c2     mk_c3     mk_c4     mk_c5     mk_final     mk_final2

p1_body.py + p2_new.txt  --mk_p2.py-->  p2_body.py  --assemble.py + tables.txt-->  p2.py
p1_body.py + p3_new.txt  --mk_p2.py-->  p3_body.py  --assemble.py + tables.txt-->  p3.py
```

`c1.py` e `main.py` plus opțiunea pentru a treia barcă. Pentru el nu există generator.

## Cum refaci totul

Logurile au fost făcute în WSL, cu PyPy 3.10 la `~/pypy/bin/pypy3`, într-un folder de lucru numit `~/moz`. Aranjarea de acolo nu e cea din repo, deci întâi trebuie refăcută.

1. Fă un folder de lucru. Scripturile `.sh` încep cu `cd ~/moz`, deci ori îi dai acest nume, ori schimbi linia. Copiază în el `arena.py` și `mozambique-engine` din `../task_mozambique/participant/`.
2. Copiază alături `main.py` și folderul `work/`.
3. Copiază din `research/` fișierele de care ai nevoie, direct în folderul de lucru. Scripturile `.sh` le caută acolo.
4. Mută `evalx2.py` în `work/`. Scripturile îl caută ca `work/evalx2.py`.

Arena oficială și verificarea noastră. La `evalx.py` argumentele sunt agentul, numărul de hărți, seed-ul și numărul de procese.

```
python arena.py main.py --maps 24 --seed 11        # la fel cu 12 și 13; nota e câmpul match_score
pypy3 work/evalx.py main.py 96 11 16
pypy3 work/evalx.py main.py 192 22 16 res_main.json
pypy3 work/evalx.py c4.py 192 22 16 res_c4.json rawv=0.5
```

Comanda exactă folosită pentru `arena_final.log` nu e salvată, deci numărul de procese paralele nu se știe. `confirm1.sh` cheamă un fișier `main_prior.py`, care nu e în folder. După `RESULTS.txt` e vechiul `main.py`, adică cel de acum. Variantele cu căutare sunt deja în `research/`, gata de rulat. Refacerea lor din `c1.py` prin generatoare se oprește la `mk_c4.py`. Acesta așteaptă în `c3.py` două setări pe care `mk_c3.py` nu le scrie, `open` 1 și `rawv` 0.5. `c3.py` din folder le are.

Tabelele Profesorului, apoi variantele cu rollout și cursa produselor:

```
python3 work/probe_wait.py wait400.jsonl 400 SEED PROCESE
python3 prof_summary.py                                   # scrie prof_sum.json
python3 prof_collect.py prof_sum2.json HARTI SEED PROCESE
pypy3 mk_tables.py prof_sum.json prof_sum2.json > tables.txt 2> tables.info

python mk_p2.py p3_new.txt p3_body.py
python3 assemble.py p3_body.py tables.txt p3.py
bash ev.sh p3.py 192 22 p3                                # scrie res_p3.json
python3 race_stats.py res_p3.json
```

Seed-urile pentru tabele nu sunt notate. Nici numărul de hărți pentru `prof_sum2.json`. Fișierele `wait400.jsonl`, `prof_sum.json` și `prof_sum2.json` nu sunt în folder. `tables.txt` este, deci primul bloc se poate sări. `ev.sh` rulează cu garda de timp oprită și tipărește nota, cursa produselor și timpii de gândire. Rulările în serie se pornesc cu `bash`, de exemplu `bash sweep_c4.sh`. Fiecare setare tipărește două rânduri: nota cu detalii și nota pe grupuri. Ca să compari două rulări, fă-le pe aceeași mașină și fără alte programe grele pornite.
