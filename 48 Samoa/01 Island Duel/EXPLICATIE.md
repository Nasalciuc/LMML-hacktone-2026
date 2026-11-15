# Samoa: Island Duel. Explicația completă

Documentul explică taskul și soluția de la zero, pentru cineva care nu a lucrat la el. Taskul are două părți. Partea 1 e un joc în care programul nostru trebuie să bată programul organizatorilor. Partea 2 e o hartă cu stele din care se scoate flagul. Căile de fișiere sunt relative la acest folder.

## Ce se pune pe site

| Ce cere platforma | Ce avem |
|---|---|
| Răspunsul | flagul `SIGMOID_OCEANICWAYFINDING` |
| Link de GitHub | adresa descrierii soluției din repo |

- Răspunsul e un flag. Nu se încarcă niciun fișier și nici agentul.
- Agentul se evaluează pe calculatorul nostru, cu programul `evaluate.py` din pachetul organizatorilor.
- Sursele din folder nu spun ce link de GitHub a fost pus.
- Nu se știe dacă platforma a acceptat flagul. Niciun fișier din folder nu confirmă asta.

## Ce cere taskul

Sunt două lucruri de făcut, unul după altul.

```
agentul nostru (my_agent.py)
   -> evaluate.py joacă 512 meciuri cu campionul
   -> pragul e trecut? atunci se salvează harta stelelor
   -> harta se decodează
   -> flagul SIGMOID_<CUVÂNT>
```

### Partea 1: jocul

Agentul e programul nostru care alege mutarea. Campionul e programul organizatorilor, adică adversarul. Regulile sunt în `../task_samoa/island-duel-participant/README.md`, în pachetul organizatorilor.

| Regulă | Cum e |
|---|---|
| Tabla | 15×15, adică 225 de pătrățele. Numărul unui pătrățel e `rând * 15 + coloană`. |
| Start | Jucătorul 0 pornește din pătrățelul 0. Jucătorul 1 pornește din 224. Jucătorul 0 mută primul. |
| Pătrățele | Fiecare are o valoare de la 1 la 9 și un cost de la 1 la 4. Unele sunt blocate. |
| Energie | Pornești cu 6. Maximul e 9. O odihnă aduce 3. |
| Ocupare | Iei un pătrățel liber vecin cu teritoriul tău și plătești costul lui. |
| Capturare | Iei un pătrățel al adversarului vecin cu teritoriul tău. Plătești costul lui plus 2. Valoarea lui trece imediat la tine. |
| Pătrățelele de start | Au valoare și cost zero. Nu pot fi capturate. |
| Durata | 256 de mutări individuale, deci 128 ale noastre. |
| Câștigător | Cine are suma valorilor mai mare la sfârșit. La scor egal e remiză. |

Vecin înseamnă sus, jos, stânga sau dreapta. Diagonala nu contează. O mutare individuală e o singură acțiune a unui singur jucător.

Jocul are ceață („fog of war”). Asta înseamnă că nu vezi toată tabla. Vezi doar pătrățelele tale și vecinii lor. Restul terenului și pozițiile adversarului rămân ascunse până ajungi lângă ele. Scorurile și energiile ambilor jucători se văd tot timpul.

Agentul e o funcție Python, `choose_action(state)`. Primește starea jocului și întoarce numărul unui pătrățel sau `None` pentru odihnă. Starea conține locul nostru (0 sau 1), energiile, scorurile și lista mutărilor permise. Pentru fiecare pătrățel mai spune patru lucruri: dacă se vede, cine îl deține, valoarea și costul. Opțional există și `reset_agent()`, care se apelează înaintea fiecărui meci.

Campionul joacă în aceeași ceață și ține minte ce a văzut. La fiecare mutare încearcă până la 2.048 de poziții și se uită cel mult 4 mutări individuale în față. Are trei profiluri, numerotate 0, 1 și 2. Profilul se vede în stare, în câmpul `opponent_profile`.

Evaluarea oficială are 512 meciuri pe hărți noi. Sunt 32 de grupe de hărți a câte 16 meciuri, cu 256 de meciuri pe fiecare loc. Te califici dacă îndeplinești ambele condiții:

1. cel puțin 308 victorii din 512;
2. victorii în cel puțin 24 din cele 32 de grupe.

Toate meciurile intră în socoteală, inclusiv înfrângerile și remizele. Nu poți relua un meci și nu poți alege hărțile.

### Partea 2: harta stelelor

După o evaluare calificată, `evaluate.py` salvează două fișiere în plus: harta ca JSON și aceeași hartă ca pagină web. Harta are 17 stele, fiecare cu o literă, și o poezie de patru versuri. Poezia spune cum se citesc literele și cum se descifrează. Cuvântul obținut se pune după `SIGMOID_`.

## Ce am aflat din date

### Despre joc și despre campion

1. **Regulile simple pierd aproape tot.** Agentul care joacă prima mutare permisă nu a câștigat niciun meci din 64. Agentul lacom din pachet a avut rata de victorie 0,031 pe 128 de meciuri. Campionul e deci puternic. `DETALII.md` mai notează că se apără la capturi.
2. **Ceața ascunde cea mai mare parte a tablei.** Starea arată doar ce se vede în acel moment. Valorile și costurile pătrățelelor ascunse vin goale. Un agent care nu ține minte uită terenul pe care l-a văzut deja.
3. **Locul contează în practică.** `DETALII.md` spune că locul 0 e constant cu aproximativ 10 puncte procentuale mai slab decât locul 1. Explicația dată acolo: cine mută primul își arată fiecare mutare unui adversar care deja vede. La evaluarea oficială diferența nu s-a văzut: 164 de victorii de pe locul 0 și 155 de pe locul 1.
4. **Profilul 2 al campionului e cel mai greu.** Așa scrie în `DETALII.md`, fără cifre. Agentul nostru nu folosește profilul deloc.
5. **Grupele nu sunt la fel de grele.** Măsurătoarea e pe 1.024 de meciuri de practică, câte 32 pe grupă. Cele mai slabe au fost grupa 16 (aproximativ 34%), grupa 29 (aproximativ 44%) și grupa 18 (aproximativ 47%). Restul au fost la 50% sau peste.
6. **Spre final rămân puține mutări posibile.** Puține pătrățele sunt și destul de ieftine, și vecine cu un teritoriu. De aceea la final se pot încerca toate variantele, ceea ce la mijlocul meciului ar dura prea mult.

### Despre harta stelelor

Harta e în `evals/attempt-1.chart.json`. Are un titlu („The navigator’s homeward chart”), prefixul `SIGMOID_`, patru versuri și 17 stele. Fiecare stea are coordonatele `x` și `y`, o mărime și o literă.

- **Stelele sunt de două feluri.** 5 sunt mari: patru au mărimea 10 și una are mărimea 7. 12 sunt mici, cu mărimea 3,5.
- **Ordinea din fișier e amestecată.** Stelele nu au numere. Ordinea de citire trebuie scoasă din poziții și din poezie.
- **În fișier `y` crește în jos, ca pe ecran.** De aceea steaua T, cu `y` = 180, e cea de jos.
- **Harta e aceeași la orice rulare calificată.** `evals/LOG.md` spune că JSON-ul ei e inclus în programul organizatorilor, cu aceleași coordonate și litere.

| Stea mare | x | y | Mărime | Unde e |
|---|---|---|---|---|
| T | 100 | 180 | 10 | jos |
| N | 40 | 88 | 10 | stânga |
| E | 96 | 28 | 10 | sus |
| C | 160 | 76 | 10 | dreapta |
| I | 132 | 130 | 7 | cea mai mică dintre cele mari |

Fiecare stea mică stă lângă o stea mare, la o distanță exactă de 18, 23 sau 28. Distanța se calculează din coordonate, cu teorema lui Pitagora.

| Stea mare | Stelele mici de lângă ea, de la cea mai apropiată | Distanțe |
|---|---|---|
| T | C, E, V | 18, 23, 28 |
| N | M, K, H | 18, 23, 28 |
| E | Y, Y | 18, 23 |
| C | N, V | 18, 23 |
| I | Z, U | 18, 23 |

## Ce am încercat, în ordine

### Partea 1: versiunile agentului

Ratele sunt din meciuri de practică împotriva campionului, din tabelul din `DETALII.md`.

| Nr. | Versiune | Fișier | Meciuri | Rată de victorie |
|---|---|---|---|---|
| 1 | prima mutare permisă | `baselines/starter_agent.py` | 64 | 0,000 |
| 2 | agent lacom | `baselines/greedy_agent.py` | 128 | 0,031 |
| 3 | prima euristică | `heur_agent.py` | 128 | 0,438 |
| 4 | Monte Carlo | `mc_agent.py` cu `sim.py` | 256 | 0,375 |
| 5 | euristică cu ponderi reglate | `agent2.py` | 512 | 0,613 |
| 6 | euristică reglată plus căutare exactă pe ultimele 6 mutări individuale | `agent4.py` | 512 | 0,621 |
| 6 | la fel, pe ultimele 10 mutări individuale | `agent4.py` | 512 | 0,627 |
| 6 | la fel ca rândul de sus, pe hărți noi | `agent4.py` | 1.024 | 0,623 |
| 7 | căutare cu numba | `agent3.py` cu `sim.py` | abandonată | nu are |

**1. Prima mutare permisă.** E agentul de pornire din pachet. Ia prima mutare din lista celor permise. Servește doar ca să verifici că totul rulează.

**2. Agentul lacom.** Vine din notebook-ul din pachet. Dă fiecărei mutări nota „câștig minus 1,5 × cost” și o joacă pe cea mai bună. Se uită doar la mutarea de acum și doar la ce se vede acum.

**3. Prima euristică.** O euristică e o regulă făcută de mână care dă repede o notă fiecărei mutări, fără să calculeze toate urmările. Aici nota e valoarea împărțită la (cost + 3). Problema rezolvată: energia e puțină, deci contează cât câștigi pe unitate de energie. Tot aici a apărut memoria hărții. Rata a urcat de la 0,031 la 0,438.

**4. Monte Carlo.** Monte Carlo înseamnă să estimezi ceva din multe încercări la întâmplare. Pentru fiecare mutare bună se inventează mai multe variante ale terenului ascuns. Pe fiecare variantă meciul e jucat până la capăt de o regulă simplă. Se alege mutarea cu media cea mai bună. A ieșit mai slab decât euristica: 0,375 față de 0,438. `DETALII.md` dă cauza: meciurile simulate erau jucate prea slab. Campionul real caută în față, regula din simulare nu.

**5. Euristică cu ponderi reglate.** Nota a primit mai multe corecții, fiecare cu o pondere, adică un număr care spune cât contează corecția. Reglarea automată a ponderilor înseamnă că un program încearcă multe seturi de numere, joacă meciuri cu fiecare și le păstrează pe cele care câștigă. Programul e `tune.py`. Pe hărțile de reglare rata a urcat de la 0,426 la 0,613 (`logs/tune_a2_r1.log`). Pe alte 512 hărți rata a fost tot 0,613.

**6. Euristică reglată plus căutare exactă a finalului.** Căutarea exactă a finalului înseamnă că pe ultimele mutări agentul nu mai dă note. Încearcă șirurile de mutări ale lui și ale adversarului până la capătul meciului și alege drumul cu cel mai bun scor final. Problema rezolvată: nota euristicii e o aproximare, iar la final câștigă cine are scorul mai mare, nu cine a jucat „eficient”. Câștigul a fost de aproximativ 1,5 puncte procentuale. Aceasta e versiunea finală.

**7. Căutare cu numba, abandonată.** Numba e o bibliotecă Python care transformă funcții Python în cod mașină, ca să ruleze mai repede. Aceeași căutare a finalului a fost scrisă întâi cu numba, în `sim.py`. Un singur meci dura peste 100 de secunde. Varianta în Python simplu durează între 0,3 și 0,9 secunde pe meci, deci a rămas ea. `agent3.py` mai avea câteva corecții în plus la notă. Reglarea lui a pornit de la 0,599 și a ajuns la 0,608 pe 768 de meciuri (`logs/tune_a3_r1.log`). Nu a fost folosit.

### Partea 2: decodarea hărții

Trei cifruri apar mai jos. Cifrul Cezar mută fiecare literă cu același număr de pași în alfabet. Cifrul Vigenère face la fel, dar numărul de pași se schimbă de la o literă la alta și e dat de literele unei chei. Beaufort e o variantă în care litera se scade din cheie, nu cheia din literă.

1. **Citire după poziție.** `decode_chart.py` tipărește literele în mai multe ordini geometrice: pe rânduri, pe coloane, după unghi sau după mărime. Nu a ieșit niciun cuvânt.
2. **Prima interpretare a poeziei.** `solve_flag.py` a presupus că stelele mari sunt cheia și că „turns the alphabet” e un cifru Cezar aplicat repetat. A încercat toate deplasările fixe și câteva reguli de deplasare după poziție. Nimic.
3. **Crucea Sudului.** `solve2.py` a pus stelele mari în ordinea literelor grecești după poziția lor în cruce. A încercat cheile SIGMOID și SIGMOIDWORD. Nimic.
4. **Căutări prin forță brută.** Forță brută înseamnă să încerci mecanic foarte multe combinații și să păstrezi ce seamănă cu un cuvânt.

| Script | Ce a încercat |
|---|---|
| `brute.py` | mai multe ordini de citire × toate cele 26 de deplasări Cezar × Vigenère cu chei tematice (SIGMOID, SAMOA, HOME, STAR, CROSS și altele) |
| `brute2.py` | toate ordinile posibile ale celor 5 stele mari; drept chei, toate aranjările literelor lor |
| `brute3.py` | o deplasare separată pentru fiecare din cele 5 grupuri de stele, în toate combinațiile |
| `brute4.py` | deplasări care cresc cu poziția literei, pentru toate perechile de pornire și pas |
| `brute5.py` | Vigenère în ambele sensuri și varianta Beaufort, cu alte chei, printre care FOUNDATION |
| `brute6.py` | deplasare calculată din litera grecească, rangul în grup și poziție, cu toți coeficienții de la 0 la 25 |
| `brute_dict.py` | aceleași familii de cifruri, verificate cu o listă de cuvinte în loc de un scor |

Primele scripturi judecau rezultatul cu un scor de „seamănă cu engleza”, făcut din perechi frecvente de litere. `brute_dict.py` căuta cel mai lung cuvânt din dicționar aflat în rezultat. Niciuna dintre căutări nu a dat un cuvânt.

5. **De ce au eșuat toate.** `DETALII.md` numește două cauze. Prima: se citeau doar cele 12 stele mici, fără literele celor 5 mari. A doua: deviza Samoei nu a fost încercată niciodată drept cheie. A fost încercat chiar cuvântul FOUNDATION, dar nu cuvântul samoan pe care îl traduce.
6. **Presupunerea greșită despre dublul Y.** Steaua mare E are lângă ea două stele mici cu aceeași literă, Y. S-a presupus că ambele trebuie să dea aceeași literă după descifrare. Asta e adevărat doar dacă tot grupul se mută cu același număr de pași. La Vigenère cheia se schimbă la fiecare literă. Cele două Y stau pe pozițiile 10 și 11, sub literele A și T ale cheii. Dau Y și F.
7. **Ideea cu deviza.** `DETALII.md` spune că a venit de la coordonator. `motto_keys.py` a luat drept chei deviza și data independenței, 1962. Le-a încercat pe toate ordinile de citire, cu și fără literele stelelor mari. A încercat scădere, adunare și Beaufort, cu cheia pornită din orice poziție. Rezultatele au fost verificate cu o listă de aproximativ 80.000 de cuvinte. Un singur rezultat a fost un text curat: OCEANICWAYFINDING.

## Soluția finală, pas cu pas

### Partea 1: agentul

Agentul e în `my_agent.py`. Acolo sunt scrise cele 16 numere ale lui. Logica e în `agent4.py`.

```
starea primită de la joc
   -> 1. memoria: notează ce se vede acum
   -> au rămas cel mult 10 mutări individuale?
         da -> 4. căutarea exactă a finalului -> mutarea
                  (dacă depășește 60.000 de poziții, mergi la 2)
         nu -> 2. euristica: o notă pentru fiecare mutare permisă
               3. regula de odihnă: joacă nota cea mai mare sau se odihnește
```

**Pasul 1. Memoria.** Agentul ține trei liste de câte 225 de numere: valoarea, costul și proprietarul fiecărui pătrățel. La început toate sunt marcate „nevăzut”. La fiecare mutare trece prin pătrățelele vizibile și le copiază în liste. Ce a ieșit din vedere rămâne cum a fost văzut ultima dată.

**Pasul 2. Euristica.** Se folosește cât timp au rămas mai mult de 10 mutări individuale. Fiecare mutare permisă primește o notă.

```
pătrățel liber:   nota = (valoare + corecții) / (cost + t_neu)
captură:          nota = (cap_g × 2 × valoare + corecții) / (cost + 2 + t_cap)
```

- **De ce se împarte la cost.** Energia e resursa care se termină. Nota măsoară cât câștigi pe unitate de energie.
- **De ce se adaugă `t_neu` și `t_cap`.** O mutare consumă și un rând, nu doar energie. Fără acest cost fix, un pătrățel cu valoare 1 și cost 1 ar avea aceeași notă ca unul cu valoare 4 și cost 4.
- **De ce „2 × valoare” la captură.** Tu câștigi valoarea și adversarul o pierde. Diferența de scor se mișcă de două ori.

Corecțiile și ponderile lor după reglare, rotunjite la două zecimale (valorile întregi sunt în `my_agent.py`):

| Pondere | Ce măsoară | Valoare |
|---|---|---|
| `t_neu` | costul fix al unei ocupări | 4,11 |
| `t_cap` | costul fix al unei capturi | 0,97 |
| `cap_g` | cât contează câștigul dublu al capturii | 0,80 |
| `look` | cea mai bună notă a unui pătrățel la care ajungi abia după această mutare | 1,56 |
| `hid` | câți vecini încă nevăzuți are pătrățelul | 1,57 |
| `dfn` | la captură: valoarea pătrățelelor noastre care scapă de singurul vecin dușman | 2,74 |
| `exp_n` | penalizare când un pătrățel ocupat rămâne lângă adversar | 0,04 |
| `exp_c` | penalizare când un pătrățel capturat rămâne lângă adversar | -1,40 |
| `cont` | bonus pentru un pătrățel liber pe care l-ar putea lua și adversarul | 0,81 |
| `opp_e` | cât depind penalizările de energia adversarului | 0,05 |
| `lam` | nota minimă ca să mute în loc să se odihnească | 3,47 |
| `save` | pragul de la care strânge energie pentru o țintă prea scumpă | -4,01 |
| `end_h` | cu câte mutări înainte de final renunță la economisire | -2,44 |

Un pătrățel nevăzut e socotit ca unul mediu: valoare 5 și cost 2,5, cu un factor de 0,8. Mai sunt două lucruri de știut despre valorile reglate. `exp_n` a ajuns aproape de zero, iar `exp_c` a ieșit negativ. Deci penalizarea pentru expunere a dispărut, iar la captură a devenit bonus. `end_h` a ieșit negativ, deci modul de „final” al euristicii nu se mai activează niciodată. Finalul e tratat de pasul 4.

**Pasul 3. Regula de odihnă.** Cu valorile reglate, regula din cod se reduce la trei cazuri:

1. Energie 7 sau mai mult: joacă mutarea cu nota cea mai mare. O odihnă ar irosi energie, fiindcă maximul e 9.
2. Energie 6 sau mai puțin și există un pătrățel la care ajunge, dar pe care nu și-l permite: se odihnește. Asta e urmarea valorii negative a lui `save`.
3. Energie 6 sau mai puțin și cea mai bună notă e sub 3,47: se odihnește. Altfel joacă mutarea cea mai bună.

Urmarea practică: agentul mută mai ales când are energie multă și se odihnește des când are puțină.

**Cum au fost reglate ponderile.** Programul e `tune.py`. Metoda se numește CMA-ES și lucrează în generații:

1. Pornește de la un set de ponderi și un „nor” de variații în jurul lui.
2. Scoate din nor mai multe seturi de ponderi apropiate.
3. Joacă cu fiecare set 512 meciuri de practică: 32 de grupe × 16 hărți.
4. Dă fiecărui set un scor: media lui `tanh(diferența de scor / 30)` pe toate meciurile.
5. Mută centrul norului spre jumătatea mai bună a seturilor și repetă.

`tanh` turtește diferențele mari. O victorie la 30 de puncte dă 0,76, una la 100 de puncte dă aproape 1. Scorul seamănă deci cu „câștig sau pierd”, dar rămâne neted. Au fost reglate 13 ponderi în 40 de generații, în aproximativ 22 de minute. La final rata era 0,613, iar diferența medie de scor crescuse de la 2,4 la 15,7 puncte. S-a folosit centrul norului de la sfârșit (`params/a2_r1_mean.json`), nu cel mai bun set găsit pe drum (`params/a2_r1_best.json`). Sursele nu spun de ce.

**Pasul 4. Căutarea exactă a finalului.** Pornește când au rămas cel mult 10 mutări individuale, adică aproximativ ultimele 5 ale noastre.

1. Ia tabla din memorie, energiile și scorurile.
2. Pentru jucătorul la rând face lista mutărilor posibile: pătrățele libere sau ale adversarului, vecine cu teritoriul lui, pe care și le permite. Păstrează cele mai bune 6 și adaugă odihna.
3. Joacă fiecare mutare „în minte”, apoi fiecare răspuns al adversarului, și tot așa până la ultima mutare a meciului.
4. La capăt notează diferența de scor dintre noi și adversar.
5. Presupune că adversarul alege mereu ce e mai rău pentru noi. Alege mutarea care duce la cea mai bună diferență garantată.

Alpha-beta e un mod de a face această căutare mai repede. Când o ramură nu mai poate schimba alegerea, o sare. Rezultatul e același ca la căutarea completă.

Căutarea are trei limite. Vede doar pătrățelele ținute minte, nu și terenul nevăzut. Ține doar 6 mutări pe poziție. Se oprește la 60.000 de poziții încercate, iar atunci agentul se întoarce la euristică pentru acea mutare.

### Partea 2: decodarea hărții

Fiecare vers din poezie e un pas.

```
5 stele mari -> ordinea grecească -> pentru fiecare: litera ei, apoi stelele mici de lângă ea
   -> 17 litere (textul cifrat)
   -> se scade cheia, literă cu literă (cheia = deviza Samoei)
   -> OCEANICWAYFINDING
```

**Pasul 1. „The five lights of home keep the Greek watch.”** Cele 5 stele mari sunt Crucea Sudului, de pe steagul Samoei. Astronomii numesc stelele unei constelații cu litere grecești. Ordinea lor în cruce e:

| Litera grecească | Poziția în cruce | Litera de pe hartă |
|---|---|---|
| alfa | jos | T |
| beta | stânga | N |
| gama | sus | E |
| delta | dreapta | C |
| epsilon | steaua mai mică | I |

**Pasul 2. „Each keeps its nearer companions before the farther.”** Pentru fiecare stea mare se citește litera ei, apoi literele stelelor mici de lângă ea, de la cea mai apropiată. Distanțele sunt în tabelul din secțiunea despre date. Iese textul cifrat `TCEVNMKHEYYCNVIZU`, 17 litere. Mai jos e împărțit pe stele mari.

```
T C E V | N M K H | E Y Y | C N V | I Z U
```

**Pasul 3. „Our foundation turns the alphabet, over and over.”** Deviza Samoei e „Fa'avae i le Atua Samoa”. „Fa'avae” înseamnă „temelie”, adică „foundation”. Deviza e cheia. Se scrie fără apostrof și fără spații: `FAAVAEILEATUASAMOA`. „Turns the alphabet” înseamnă că fiecare literă e mutată în alfabet. „Over and over” înseamnă că o cheie mai scurtă decât textul se reia de la capăt.

**Pasul 4. „For the homeward message, undo those turns.”** Mutările se desfac, deci cheia se scade. Acesta e cifrul Vigenère, în sensul de descifrare. Literele se scriu ca numere: A = 0, B = 1 și tot așa până la Z = 25. Din numărul literei cifrate se scade numărul literei din cheie aflate pe aceeași poziție. Dacă iese sub zero se adaugă 26.

Primele cinci litere, lucrate de mână:

| Poziția | Litera cifrată | Număr | Litera din cheie | Număr | Scădere | Rezultat |
|---|---|---|---|---|---|---|
| 1 | T | 19 | F | 5 | 19 - 5 = 14 | O |
| 2 | C | 2 | A | 0 | 2 - 0 = 2 | C |
| 3 | E | 4 | A | 0 | 4 - 0 = 4 | E |
| 4 | V | 21 | V | 21 | 21 - 21 = 0 | A |
| 5 | N | 13 | A | 0 | 13 - 0 = 13 | N |

Un caz cu rezultat sub zero e poziția 8: H = 7 și L = 11. 7 - 11 = -4, iar -4 + 26 = 22, adică W. Tot textul e mai jos.

```
poziția        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
text cifrat    T  C  E  V  N  M  K  H  E  Y  Y  C  N  V  I  Z  U
ca număr      19  2  4 21 13 12 10  7  4 24 24  2 13 21  8 25 20
cheia          F  A  A  V  A  E  I  L  E  A  T  U  A  S  A  M  O
ca număr       5  0  0 21  0  4  8 11  4  0 19 20  0 18  0 12 14
diferența     14  2  4  0 13  8  2 22  0 24  5  8 13  3  8 13  6
rezultat       O  C  E  A  N  I  C  W  A  Y  F  I  N  D  I  N  G
```

Cheia are 18 litere și textul are 17. Ultima literă a cheii e A și nu mai e folosită. Aici cheia nu ajunge deci să se reia.

**Pasul 5. Flagul.** Prefixul din hartă e `SIGMOID_`. Rezultatul e `SIGMOID_OCEANICWAYFINDING`. „Oceanic wayfinding” e navigația polineziană după stele.

## Cât de bine merge

### Partea 1: agentul

Evaluarea oficială a fost rulată o singură dată, cu `my_agent.py`.

| Măsură | Rezultat | Prag |
|---|---|---|
| Victorii | 319 din 512 (0,623) | cel puțin 308 |
| Grupe câștigate | 32 din 32 | cel puțin 24 |
| Rezultat | calificat | |
| Remize | 11 | |
| Victorii de pe locul 0 | 164 | |
| Victorii de pe locul 1 | 155 | |
| Durată | aproximativ 4 minute | |

Marja față de prag e de 11 victorii. Din scădere rezultă 182 de înfrângeri (512 - 319 - 11).

Mai târziu agentul a fost comparat în practică cu un al doilea agent, pe aceleași hărți. Al doilea agent are căutare alpha-beta scrisă în C++. E făcut de un coleg, nu e în acest folder și nu a fost folosit. Fiecare a jucat 1.280 de meciuri de practică.

| Eșantion | Agentul nostru | Agentul în C++ |
|---|---|---|
| 256 de meciuri | 155 | 112 |
| 512 meciuri | 314 | 229 |
| 512 meciuri | 304 | 223 |
| Total, 1.280 de meciuri | 773 (60,4%) | 564 (44,1%) |

Pe aceleași hărți, agentul nostru a câștigat 243 de meciuri pe care celălalt le-a pierdut. Celălalt a câștigat 34 de meciuri pe care al nostru le-a pierdut. `README.md` notează că în practică ambii au avut victorii în toate grupele.

Sursele dau două cifre de practică pentru agentul final. `DETALII.md`, mai vechi, are 0,623 pe 1.024 de meciuri. `README.md`, mai nou, are 60,4% pe 1.280 de meciuri. Sunt eșantioane diferite. Cifra mai nouă e 60,4%.

### Partea 2: harta

Toate cele 17 litere se descifrează și nu rămâne nimic pe dinafară. Rezultatul e o expresie englezească de 17 litere, legată de temă. `DETALII.md` notează că nicio altă ordine, cheie sau variantă de cifru nu a dat un cuvânt de 8 litere sau mai lung.

## Cum a fost verificată

**Agentul.**

1. Înainte de evaluarea oficială a fost rulat pe 1.024 de hărți de practică noi. Un seed e numărul din care se generează o hartă, iar aici s-a pornit de la altă bază de seed, 70000. Rata a fost 0,623.
2. Evaluarea oficială e dovada propriu-zisă. Fișierele ei sunt în `evals/`. `evaluate.py` refuză să scrie peste un rezultat existent, deci dovada nu poate fi înlocuită din greșeală.
3. Comparația cu agentul în C++ s-a făcut pe aceleași hărți. Așa diferența nu poate veni din noroc la hărți.

**Harta.**

1. Scăderea a fost refăcută literă cu literă, de mână. Dă același cuvânt. Tabelul e mai sus.
2. Distanțele de 18, 23 și 28 au fost recalculate din coordonatele din `evals/attempt-1.chart.json`. Fiecare stea mică are o singură stea mare clar mai apropiată decât celelalte.
3. `motto_keys.py` a încercat și celelalte ordini, chei și variante. Doar combinația de mai sus dă un text curat.
4. Cheia e exact deviza, pornită de la prima literă. Ordinea de citire e exact cea din poezie. Nu a fost nevoie de nicio ajustare.
5. Un singur lucru nu a fost verificat: răspunsul platformei la flag.

## Ce nu e sigur

- **Dacă platforma a acceptat flagul.** Nu se știe.
- **Marja de calificare a fost mică.** 319 față de 308. Evaluarea oficială s-a rulat o singură dată. Unul dintre eșantioanele de practică de 512 meciuri a dat 304 victorii, adică sub prag. O a doua evaluare oficială ar putea să nu treacă. Nu e nevoie de ea, fiindcă harta e aceeași la orice rulare calificată.
- **Ce înseamnă exact „grupă câștigată”.** Sursele spun doar „victorii în cel puțin 24 de grupe”. Nu spun câte victorii trebuie într-o grupă.
- **Efectul locului.** În practică locul 0 era cu aproximativ 10 puncte procentuale mai slab. La evaluarea oficială a avut mai multe victorii decât locul 1. Sursele nu explică diferența.
- **Dacă taskul aduce puncte.** `DETALII.md` îl descrie ca „learning/experiment only, not scored”. `README.md`, mai nou, nu mai spune asta.
- **Forma flagului.** Harta dă 17 litere legate, fără spațiu între OCEANIC și WAYFINDING. Flagul e scris la fel. Dacă platforma îl vrea altfel nu se știe.
- **Comparația cu agentul în C++.** Fișierele ei nu sunt în acest folder, deci cifrele nu pot fi reverificate de aici.
- **Câteva alegeri nu au explicație în surse.** Nu se știe de ce s-a folosit centrul norului și nu cel mai bun set de ponderi. Dintre adâncimile de căutare, doar 6 și 10 au rezultate notate. Celelalte fișiere din `params/` nu au.
- **Lucruri pe care agentul nu le face.** Nu folosește profilul campionului, deși profilul 2 e cel mai greu. Căutarea finalului nu ia în calcul terenul nevăzut.

## Unde e fiecare lucru

| Grup | Fișier | Ce e |
|---|---|---|
| Descrieri | `README.md` | descrierea scurtă, la zi |
| Descrieri | `DETALII.md` | descrierea lungă, mai veche, în engleză |
| Descrieri | `EXPLICATIE.md` | acest document |
| Agentul final | `my_agent.py` | agentul trimis la evaluare; conține cele 16 numere și apelează `agent4.py` |
| Agentul final | `agent4.py` | memoria, euristica, regula de odihnă și căutarea exactă a finalului |
| Agentul final | `params/a4_eg10.json` | aceleași 16 numere, ca fișier separat pentru meciurile de practică |
| Versiuni vechi | `baselines/starter_agent.py`, `baselines/greedy_agent.py` | cei doi agenți simpli din pachet |
| Versiuni vechi | `heur_agent.py` | prima euristică |
| Versiuni vechi | `mc_agent.py` | agentul Monte Carlo |
| Versiuni vechi | `sim.py` | simulatorul jocului scris cu numba, folosit de `mc_agent.py`; tot aici e căutarea cu numba, abandonată |
| Versiuni vechi | `agent2.py` | euristica cu ponderi reglate, fără căutare |
| Versiuni vechi | `agent3.py` | euristica plus câteva corecții și căutarea cu numba; nefolosit |
| Unelte | `practice.py` | joacă multe meciuri de practică în paralel și tipărește victoriile pe loc, pe profil și pe grupă |
| Unelte | `tune.py` | reglarea automată a ponderilor |
| Ponderi | `params/a2_r1_mean.json`, `params/a2_r1_best.json` | rezultatele reglării lui `agent2.py`; s-a folosit primul |
| Ponderi | `params/a3_init.json` | ponderile de pornire pentru reglarea lui `agent3.py` |
| Ponderi | `params/a4_eg6.json`, `a4_eg12.json`, `a4_eg14.json`, `a4_k8_p10.json`, `a4_k10_p12.json` | alte variante pentru `agent4.py` |
| Ponderi | `params/eg4.json`, `eg6.json`, `eg8.json`, `eg10.json`, `tmp_eg4.json` | variante mai vechi; sursele nu le descriu |
| Jurnale | `logs/starter.json`, `greedy.json`, `heur_v1.json`, `mc_v1.json` | după nume, meciurile primelor patru versiuni |
| Jurnale | `logs/tune_a2_r1.log`, `tune_a2_r1.json` | reglarea lui `agent2.py`, generație cu generație |
| Jurnale | `logs/tune_a3_r1.log`, `tune_a3_r1.json` | reglarea lui `agent3.py` |
| Jurnale | `logs/base_a2mean_s40k.json`, `a4_eg6_s40k.json`, `a4_eg10_s40k.json` | după nume, rulările de 512 meciuri din tabelul versiunilor |
| Jurnale | `logs/a4_eg10_valid1024.json` | după nume, verificarea pe 1.024 de hărți noi |
| Jurnale | `logs/fake.chart.json` | o hartă de probă, cu titlul „Test” |
| Jurnale | `logs/explore_maps.json`, `det_a.json`, `det_b.json`, `tune_smoke.json` | sursele nu le descriu |
| Evaluarea oficială | `evals/LOG.md` | jurnalul încercărilor; are o singură linie, `attempt-1` |
| Evaluarea oficială | `evals/attempt-1.json` | toate cele 512 meciuri și rezumatul |
| Evaluarea oficială | `evals/attempt-1.events.jsonl` | jurnalul scris meci cu meci în timpul rulării |
| Evaluarea oficială | `evals/attempt-1.html` | câteva meciuri care pot fi revăzute în browser |
| Evaluarea oficială | `evals/attempt-1.chart.json`, `attempt-1.chart.html` | harta stelelor, ca date și ca pagină |
| Decodarea hărții | `decode_chart.py` | tipărește literele hărții în mai multe ordini geometrice; nu dă singur flagul |
| Decodarea hărții | `motto_keys.py` | căutarea care a găsit soluția |
| Decodarea hărții | `solve_flag.py`, `solve2.py` | primele interpretări ale poeziei |
| Decodarea hărții | `brute.py`, `brute2.py`, `brute3.py`, `brute4.py`, `brute5.py`, `brute6.py`, `brute_dict.py` | căutările prin forță brută care nu au dat nimic |

În afara folderului sunt două lucruri de care depind scripturile:

- `../task_samoa/island-duel-participant/`: pachetul organizatorilor. Conține regulile (`README.md`), `evaluate.py` și `run_agent.py`. Tot acolo sunt `client.py`, `replay.py` și notebook-ul `baseline_solution.ipynb`. Programul jocului e `samoa-windows-x86_64.exe`.
- `../../17 Cuba/01 If you know you know/index/words.txt`: lista de cuvinte folosită de `motto_keys.py` și de `brute_dict.py`.

## Cum refaci totul

Comenzile vin din `DETALII.md` și se rulează din acest folder. `run_agent.py` și `evaluate.py` sunt în pachet, așa că le-am adăugat calea. Comenzile nu au fost rulate din nou la scrierea acestui document.

**Ce trebuie instalat.** Pachetul cere Python 3.10 sau mai nou, pe Windows sau Linux pe 64 de biți. Agentul final nu are nevoie de nicio bibliotecă. `tune.py` are nevoie de numpy. `sim.py` și `mc_agent.py` au nevoie de numba. `DETALII.md` notează versiunile folosite: numba 0.65.1 și numpy 2.4.4.

**1. Un meci de practică, cu reluare în browser.**

```
python "../task_samoa/island-duel-participant/run_agent.py" --agent my_agent.py --arena 0 --seed 144 --output practice.json
```

Iese `practice.json` și `practice.html`. În comenzi, grupa se numește `arena`.

**2. Multe meciuri de practică.**

```
python practice.py --agent agent4.py --arenas 0-31 --seeds 0-15 --workers 12 --seed_base 40000 --params params/a4_eg10.json --out logs/run.json
```

Sunt 32 de grupe × 16 hărți, deci 512 meciuri. `--seed_base` alege setul de hărți. Programul tipărește victoriile totale, apoi pe loc, pe profil și pe grupă.

**3. Reglarea ponderilor.**

```
python tune.py --agent agent2.py --params cap_g,t_neu,t_cap,look,hid,dfn,exp_n,exp_c,cont,lam,save,end_h,opp_e --seeds 0-15 --gens 40 --out logs/tune_nou.json
```

Lista de ponderi și numărul de generații sunt luate din `logs/tune_a2_r1.log`. Comanda exactă de atunci nu e păstrată. Pune alt nume la `--out`, ca să nu scrii peste jurnalul vechi. Nu se știe dacă iese exact același set de ponderi.

**4. Evaluarea oficială.** Nu e nevoie să fie refăcută. Dacă totuși o rulezi, alege un nume nou de fișier și treci rezultatul în `evals/LOG.md`. Ține cont că poate ieși sub prag.

```
python "../task_samoa/island-duel-participant/evaluate.py" --agent my_agent.py --output evals/attempt-2.json
```

`DETALII.md` dă comanda cu `agent4.py` și cu ponderile într-o variabilă de mediu. `evals/LOG.md` arată că rularea oficială s-a făcut cu `my_agent.py`, care are ponderile în el.

**5. Decodarea hărții cu scriptul.**

```
python motto_keys.py
```

Scriptul tipărește cele mai bune rezultate. Caută linia cu eticheta `cg_TNECI|vdec|FAAVAEILEATUASAMOA|off0`. Ea conține `OCEANICWAYFINDING`.

Eticheta se citește așa. `cg` înseamnă stelele mici de la cea mai apropiată, cu litera stelei mari în față. `TNECI` e ordinea stelelor mari. `vdec` e scăderea cheii. `off0` înseamnă că cheia pornește de la prima ei literă. Calea către lista de cuvinte e scrisă în script ca o cale completă de pe acest calculator. Pe alt calculator trebuie schimbată.

**6. Decodarea hărții de mână.** Deschide `evals/attempt-1.chart.json`. Urmează pașii 1 până la 5 din secțiunea „Soluția finală, pas cu pas”, partea 2. Ai nevoie doar de coordonate, de deviză și de tabelul cu A = 0.
