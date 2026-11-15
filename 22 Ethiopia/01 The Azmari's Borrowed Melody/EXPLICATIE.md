# Etiopia: The Azmari's Borrowed Melody. Explicația completă

Documentul explică taskul, ce am aflat din date, ce am încercat și ce trimitem. E scris pentru un coleg care nu a lucrat la task. Cifrele sunt copiate din fișierele din folder, la starea din 15 noiembrie 2026. Unde sursele nu spun ceva, scrie că nu se știe.

## Ce se pune pe site

Pe site se încarcă un program Python. Se adaugă și un link GitHub către descrierea soluției. Adresa linkului nu apare în surse. Flagul vine abia după ce programul trece testele platformei.

| Ordine | Fișier | sha256 |
|---|---|---|
| Prima încercare | `main.py` din vârful folderului | `55d236f83f1d9c058434f7042c0dd60dd5f4797cfca0d7a17af720f71bf176a1` |
| A doua încercare | `a_doua_incercare/main.py` | `f6e1b1b5a6a40a1c82eb15dd28271afa6b88cd829828cd677632c6209b3d62ba` |

1. Rulează `sha256sum main.py`. Comanda afișează amprenta fișierului. Ea trebuie să fie cea din tabel. Așa știi că încarci fișierul care a fost testat.
2. Încarcă `main.py` și pune linkul GitHub.
3. Dacă platforma respinge fișierul, notează numele testului picat. Apoi încarcă `a_doua_incercare/main.py`.

A doua încercare e alt detector, scris de un coleg. Pe testele noastre trece 312 din 344 de înregistrări. `main.py` trece 334 din aceleași 344.

Platforma rulează programul pe trei înregistrări ascunse. Arată doar primul test picat: test-1, test-2 sau test-3. Dacă arată test-2, prima înregistrare a trecut. Dacă arată test-3, primele două au trecut. Despre testele de după cel arătat nu se știe nimic.

Flagul are forma `SIGMOID_<WORD>`. Verificatorul îl dă doar când trec toate cele trei înregistrări. Pentru acest task nu avem încă niciun flag. Nicio sursă nu spune că `main.py` de acum a fost trimis sau acceptat. `DETALII.md` spune că la data scrierii lui nu fusese trimis.

## Ce cere taskul

Primești o înregistrare audio de 45 până la 60 de secunde. În ea cântă un instrument etiopian. Poate fi masenqo, care se cântă cu arcuș. Poate fi washint, care e un fluier.

Cineva a scos câteva secunde din înregistrare. În locul lor a lipit o bucată din altă înregistrare. La îmbinări nu e pocnet și nu e salt de volum. Trebuie să dai două numere: secunda la care începe bucata lipită și secunda la care se termină.

```
test-1-input.wav  ->  python main.py test-1-input.wav  ->  detection.csv
ce dă platforma       ce rulează platforma                 start_s,end_s
                                                           14.965,20.912
```

Fișierul CSV are un rând de titlu și un rând cu cele două numere. În CSV numerele se scriu cu punct. În textul de aici sunt scrise cu virgulă, ca în română.

| Regulă | Valoare |
|---|---|
| Când trece un test | ambele margini sunt la cel mult 1,5 s de adevăr |
| Înregistrări ascunse | 3 |
| Când se dă flagul | când trec toate trei |
| Exemple cu răspuns | unul singur, fișierul train |
| Răspunsul la train | început 14,960 s, sfârșit 20,965 s |
| Biblioteci de care e nevoie | numpy și scipy |
| Dificultate | Medium, după `arhiva/README.md` |

Fișierul train stă în `../task_ethiopia/`, cu un nivel mai sus de folderul taskului. Are circa 56 de secunde, iar bucata lipită din el are circa 6. Textul complet al enunțului nu e în folder. Regulile de mai sus vin din `README.md`, `DETALII.md` și `arhiva/README.md`.

### Cuvintele folosite mai jos

| Cuvânt | Ce înseamnă aici |
|---|---|
| bucata lipită | cele câteva secunde aduse din altă înregistrare |
| gazda | restul înregistrării, partea originală |
| margine | începutul sau sfârșitul bucății lipite |
| cadru | o felie scurtă de sunet; în `main.py` are circa 93 de milisecunde și se ia una la fiecare 50 de milisecunde |
| măsură | un număr calculat pe un cadru, de exemplu cât zgomot e sub 120 Hz |
| bloc | un șir continuu de cadre, candidat pentru bucata lipită |
| lanț de înregistrare | tot ce stă între instrument și fișier: microfon, cameră, aparat, zgomot de fond |
| detector | programul care primește fișierul și scoate cele două numere |
| sintetic | sunet făcut de un program, nu înregistrat cu microfonul |
| Hz | vibrații pe secundă; sunetele joase au puțini Hz |
| armonice | o notă are o frecvență de bază și multipli ai ei; multiplii sunt armonicele și dau timbrul instrumentului |
| dB | unitatea pentru tărie; 20 dB în plus înseamnă o undă de 10 ori mai mare |

## Ce am aflat din date

### Din fișierul train

Colegul care a scris detectorul din `arhiva/` a comparat cele 6 secunde lipite cu restul fișierului. A găsit patru diferențe. Sunt descrise în `arhiva/README.md`, cu imagini în `arhiva/figures/`.

| Semn | În gazdă | În bucata lipită |
|---|---|---|
| Instrumentul | masenqo | washint: domină armonicele impare, armonica a 2-a e la circa -15 dB |
| Registrul, adică ce note se cântă | C4 până la A4 | B4 până la E5, deci mai sus |
| Banda 20-150 Hz | mai slabă cu 20-30 dB | tonuri la 30-95 Hz sub fiecare notă |
| Deriva lentă sub 10 Hz | există și depinde de nota cântată | lipsește |

Deriva lentă înseamnă că nivelul mediu al undei urcă și coboară foarte încet. Nu se aude, dar se vede în cifre. Primele două semne țin de instrument și de note. Ultimele două țin de lanțul de înregistrare.

La îmbinări nu există pocnet sau salt de volum, exact cum spune enunțul. De aceea nu căutăm îmbinarea. Căutăm porțiunea care sună altfel pe toată lungimea ei.

### Din testele pe alte înregistrări

Avem un singur exemplu cu răspuns. De aceea am făcut 344 de înregistrări de test cu răspuns cunoscut. Primele cinci lucruri de mai jos vin din ele. Al șaselea vine din `arhiva/README.md`, iar al șaptelea de la platformă.

1. În train apar toate cele patru semne deodată. O înregistrare ascunsă poate avea doar unele dintre ele.
2. Măsurile care urmăresc notele cântate strică pe alte melodii. Acolo mutau marginile sau alegeau un pasaj obișnuit al gazdei.
3. Blocurile mai lungi de 20 de secunde au fost de fiecare dată alegeri greșite.
4. Măsurile despre lanțul de înregistrare nu ajung singure. Cu ele singure trec 317 din 344, mai puțin decât cu toate măsurile.
5. Dintre cele trei feluri de bucăți sintetice, cel mai greu e același instrument prin același lanț de înregistrare. Acolo diferă doar felul de a cânta.
6. Fără semnele din frecvențele joase, începutul găsit pe train iese cu 1,0 s mai târziu. E încă în toleranță, dar aproape de limită.
7. Detectorul nostru vechi a răspuns bine pe train, cu 14,95 și 20,95. Totuși a picat pe platformă. Un răspuns bun pe train nu garantează nimic pentru înregistrările ascunse.

## Ce am încercat, în ordine

Au fost două familii de detectoare. Prima e a noastră și stă în `varianta_veche/`. A doua pornește de la detectorul primit de la coleg și stă în `arhiva/` și în `main.py`. Ultimul adăugat e un al treilea detector, în `a_doua_incercare/`.

### 1. Primul detector al nostru (commit `ec2b674`)

- Ce s-a încercat: 37 de măsuri despre lanțul de înregistrare și despre culoarea sunetului. Cadrele aveau 0,4 s și se luau la fiecare 0,1 s. Se căuta porțiunea de 1,5 până la 20 s a cărei medie diferă cel mai mult de rest.
- Ce a ieșit: după `DETALII.md` și `varianta_veche/README.md`, a picat pe platformă la test-3. Primele două înregistrări ascunse au trecut.
- De ce a fost lăsat: nu trecea a treia înregistrare. Codul lui a rămas ca prima trecere din `varianta_veche/main.py`.

### 2. A doua versiune (commit `b061ff3`)

- Ce s-a încercat: porțiuni mai largi, de la 1 la 30 s. Măsuri noi pentru partea de sus a spectrului, pentru forma zgomotului de fond și pentru acordaj. Margini rafinate din 25 în 25 de milisecunde. O reducere de scor pentru porțiunile scurte.
- Ce a ieșit: după aceleași două surse, a picat pe platformă la test-2. Adică a stricat un test pe care prima versiune îl trecea.
- De ce a fost lăsată: bănuiala a căzut pe reducerea pentru porțiuni scurte. Era singura schimbare care împingea marginile în afară pe îmbinări asemănătoare cu train. Cea mai mare greșeală de margine era 1,17 s cu reducerea și 0,39 s fără ea.

### 3. Forma în doi pași (`varianta_veche/main.py`, commit `ba2fb1b`)

Scopul a fost să păstrăm prima versiune și să o schimbăm doar când nu e sigură.

```
trecerea 1 = versiunea ec2b674               -> răspunsul A
trecerea 2 = versiunea b061ff3 fără reducere -> răspunsul B
A și B diferă cu cel mult 1,5 s                                 -> rămâne A
A are scor de 1,5 ori peste orice porțiune care nu îl atinge    -> rămâne A
altfel                                                          -> se ia B
```

- Ce a ieșit la noi: pe train 14,95 și 20,95, cu scorul lui A de 2,6 ori peste celelalte porțiuni. Verificarea e în `varianta_veche/validate.py`. Pe setul „realistic” răspunsul final a trecut 80 din 80. Pe setul „tough” a trecut 148 din 180. Trecerea 1 singură trecea 111, iar trecerea 2 singură 149.
- Ce a ieșit pe platformă: nimic. După `DETALII.md` și `varianta_veche/README.md`, forma în doi pași nu a fost trimisă.
- De ce a fost lăsată: a picat pe platformă. În plus, pe 36 de înregistrări sintetice a trecut 32, față de 34 pentru primul detector.

### 4. Detectorul primit de la coleg (`arhiva/main.py`)

- Ce s-a încercat: detectorul din arhiva `lmml-ethiopia-azmari-borrowed-melody.zip`, neschimbat. Calculează 37 de măsuri la fiecare 50 de milisecunde, inclusiv măsuri despre notele cântate. Caută blocul de 1,5 până la 40 s care diferă cel mai mult de rest.
- Ce a ieșit: pe train 14,965 și 20,912. Pe cele 344 de înregistrări ale noastre trece 327, deci pică 17.
- De ce nu a fost trimis așa: 7 din cele 17 picate veneau din două cauze reparabile. Măsurile de melodie mutau marginile. Blocurile lungi câștigau pe nedrept.

Autorul l-a verificat și el, pe seturile lui. Cifrele sunt din `arhiva/README.md` și nu au fost refăcute la noi:

| Set al autorului | Trecute |
|---|---|
| Bucata din train lipită în alte locuri | 49 din 49 |
| Variante mai grele: scurte, lungi, la margini, cu zgomot | 96 din 100 |
| Sintetice, alt instrument | 46 din 46 |
| Sintetice, același instrument | 87 din 92 |

Următoarele patru încercări pornesc toate de la acest detector. Cifrele lor sunt scoase de `check/ablation.py` și se află în `check/results.txt`:

| Variantă | Primele 140 | Noi 2,5-10 s | Scurte 1,5-2,5 s | Lungi 10-14 s | Lungi 14-20 s | La început | La sfârșit | Toate 344 |
|---|---|---|---|---|---|---|---|---|
| Număr de înregistrări | 140 | 72 | 30 | 30 | 24 | 24 | 24 | 344 |
| Arhiva neschimbată, maxim 40 s | 134 | 70 | 22 | 29 | 24 | 24 | 24 | 327 |
| Maxim 20 s | 135 | 71 | 22 | 29 | 24 | 24 | 24 | 329 |
| Maxim 15 s | 135 | 71 | 23 | 29 | 9 | 24 | 24 | 315 |
| Maxim 12 s | 136 | 71 | 23 | 24 | 0 | 24 | 24 | 302 |
| Fără măsurile de melodie | 135 | 71 | 25 | 29 | 24 | 24 | 24 | 332 |
| Fără melodie și maxim 20 s, adică `main.py` | 136 | 72 | 25 | 29 | 24 | 24 | 24 | 334 |
| Fără melodie și maxim 15 s | 136 | 72 | 26 | 29 | 9 | 24 | 24 | 320 |
| Fără melodie și maxim 12 s | 137 | 72 | 27 | 25 | 0 | 24 | 24 | 309 |
| Doar măsurile de lanț | 132 | 67 | 21 | 29 | 22 | 24 | 22 | 317 |

### 5. Bloc mai scurt: maxim 12 s, 15 s sau 20 s

Limita de lungime a blocului a fost coborâtă de la 40 s. Primul gând a fost 12 s. Ajută câteva bucăți scurte, dar pierde toate bucățile de 14-20 s: trec 0 din 24. La 15 s trec doar 9 din 24. La 20 s nu se pierde nimic și totalul urcă de la 327 la 329. A rămas 20 s, cea mai mică limită care nu strică bucățile lungi.

### 6. Fără măsurile de melodie

Au fost scoase din căutare cele 9 măsuri care urmăresc notele. Trec 332 din 344. De aici vine cea mai mare parte din câștig. Bucățile scurte urcă de la 22 la 25 din 30. Schimbarea a fost păstrată.

### 7. Doar măsurile de lanț

Căutarea a fost făcută doar cu măsurile despre lanțul de înregistrare. Pe lângă melodie au fost scoase armonicele, acordajul, volumul în timp și tremurul notei. Trec 317 din 344, mai puțin decât arhiva neschimbată. Varianta a fost lăsată, fiindcă măsurile scoase ajută.

### 8. Fără melodie și maxim 20 s

Cele două schimbări bune au fost puse împreună. Trec 334 din 344. Față de arhivă au fost reparate 7 înregistrări și nu a fost stricată niciuna. Aceasta e versiunea din `main.py`.

### 9. Al treilea detector (`a_doua_incercare/main.py`)

E scris de un coleg și lucrează altfel. Sursele nu spun dacă e același coleg care a trimis arhiva. Textul de descriere de la începutul fișierului spune cum lucrează. Pentru fiecare cadru se caută cele mai asemănătoare cadre din restul fișierului. Cadrele gazdei se repetă, deci își găsesc perechi. Un cadru din bucata lipită nu are pereche. Zonele fără pereche devin candidați. Se alege candidatul cel mai consecvent, apoi marginile lui sunt rafinate.

Trece 312 din aceleași 344 de înregistrări. E păstrat ca a doua încercare, dacă platforma respinge `main.py`. Are scor mai mic pe testele noastre, deci nu e prima alegere.

## Soluția finală, pas cu pas

`main.py` este detectorul din `arhiva/main.py` cu două schimbări. Ele sunt la pașii 5 și 7.

```
test-1-input.wav
  -> [1] citire: un singur canal, aceeași scară de volum
  -> [2] cadre de 93 ms, unul la 50 ms
  -> [3] nota cântată în fiecare cadru
  -> [4] 37 de măsuri pe fiecare cadru
  -> [5] se scot cele 9 măsuri de melodie, rămân 28
  -> [6] netezire, loc în ordine, curbă în clopot, decorelare
  -> [7] fiecare bloc de 1,5-20 s primește un scor
  -> [8] blocul cu scorul cel mai mare dă începutul și sfârșitul
  -> detection.csv
```

### De la fișier la măsuri

1. Citirea. Intră calea către fișierul WAV. Iese unda, ca șir de numere. Fișierele pe 8 biți, pe 32 de biți și cele cu numere zecimale sunt aduse la scara de 16 biți. Un fișier stereo devine media celor două canale. Problema rezolvată: nu știm în ce format sunt înregistrările ascunse.
2. Cadrele. Unda e tăiată în cadre de circa 93 de milisecunde. Se ia unul la fiecare 50 de milisecunde, deci 20 pe secundă. Pentru fiecare cadru se calculează spectrul, adică energia de pe fiecare frecvență. La fișiere mai lungi de 75 de secunde pasul crește, ca numărul de cadre să rămână cel mult 1500.
3. Nota. Programul încearcă note de la 70 Hz în sus, cu 96 de trepte pe octavă. Pentru fiecare adună nivelul de la primele 6 armonice. Scade jumătate din nivelul de la mijlocul dintre armonice. Nota cu cel mai mare rezultat câștigă. Problema rezolvată: multe măsuri se raportează la notă, de exemplu energia de sub ea.
4. Măsurile. Pe fiecare cadru se calculează 37 de numere. Sunt în tabelul de mai jos.

| Grup | Nume în cod | Ce măsoară | Intră în căutare |
|---|---|---|---|
| Nota | `f0log` | cât de sus e nota cântată | nu |
| Culoarea sunetului | `centroid_f0`, `mfcc1` până la `mfcc6` | unde e centrul energiei și ce formă are spectrul | nu |
| Schimbarea în timp | `flux` | cât se schimbă spectrul de la un cadru la altul | nu |
| Acordajul | `tune_s`, `tune_c` | cât de departe e nota de grila de semitonuri | da |
| Armonicele | `h2_h1` până la `h5_h1`, `even_odd`, `hslope` | cât de tari sunt armonicele 2-5 față de prima, pare față de impare, panta lor | da |
| Sub notă | `subf0_rel` | partea de energie aflată sub nota cântată | da |
| Frecvențe joase | `lf_abs`, `lf_lo`, `infra`, `dc` | nivelul la 20-120 Hz, la 3-40 Hz, sub 12 Hz și deriva sub 2 Hz | da |
| Zgomot între armonice | `nz_lo`, `nz_mid`, `nz_hi` | zgomotul dintre armonice față de armonice, pe trei benzi | da |
| Zgomotul de fond | `floor1`, `floor2`, `floor3` | nivelul celor mai slabe 10% din frecvențe, pe trei benzi între 1 și 10 kHz | da |
| Frecvențe înalte | `hf_rel`, `flat2_8k` | partea de energie de la circa 5 kHz în sus și cât de plat e spectrul la 2-8 kHz | da |
| Brum de rețea | `hum50`, `hum60` | bâzâitul de la priză, la 50 Hz sau la 60 Hz | da |
| Forma undei | `skew`, `crest` | cât de asimetrică e unda și cât de ascuțite îi sunt vârfurile | da |
| Volumul în timp | `am_fast`, `env_dip` | tremurul rapid al volumului și cât de adânc coboară volumul în 0,6 s | da |
| Tremurul notei | `f0_jit` | cât se mișcă nota în 0,3 s | da |

Acordajul rămâne în căutare deși ține de notă. El măsoară abaterea față de grila de semitonuri, nu nota în sine. Un instrument acordat puțin mai sus rămâne mai sus la orice notă.

### De la măsuri la răspuns

5. Melodia iese. Primele trei rânduri din tabel au 9 măsuri. Ele se calculează, dar nu intră în căutare. Rămân 28. Aceasta e prima schimbare față de arhivă.
6. Aceeași scară. Fiecare valoare e înlocuită cu cea din mijloc dintre 5 cadre vecine, ca o valoare aberantă izolată să dispară. Apoi e înlocuită cu locul ei în ordinea crescătoare, așezat pe o curbă în formă de clopot. După asta toate măsurile au aceeași împrăștiere, fie că erau în dB sau în Hz. La sfârșit măsurile sunt decorelate, adică legăturile dintre ele sunt scoase. Altfel același lucru ar fi numărat de mai multe ori: sunt patru măsuri doar pentru frecvențele joase.
7. Căutarea. Programul ia pe rând fiecare bloc posibil și îi dă un scor. Un bloc are între 1,5 s și 20 s și cel mult jumătate din fișier. La 56 de secunde sunt câteva sute de mii de blocuri. Limita de 20 s e a doua schimbare față de arhivă, unde era 40 s.
8. Răspunsul. Câștigă blocul cu scorul cel mai mare. Începutul e centrul primului său cadru minus 25 de milisecunde. Sfârșitul e centrul ultimului cadru plus 25 de milisecunde. Cele două numere se scriu în `detection.csv`, cu trei zecimale.

```
cadre:  |---------------------- N cadre ----------------------|
bloc:                [ a ........... b )
                      n1 cadre înăuntru       n2 = N - n1 cadre afară

scor = n1 * n2 / N^2 * (distanța dintre media dinăuntru și media de afară)^2
```

Distanța se calculează pe toate cele 28 de măsuri deodată. Factorul `n1 * n2` ține cont de câte cadre susțin diferența. Un bloc foarte scurt cu o medie extremă primește scor mic. Distanța e ridicată la pătrat, deci nu contează dacă o măsură urcă sau coboară în bucata lipită. De aceea detectorul merge la fel când gazda e masenqo și când e washint.

### Plasa de siguranță

Dacă programul s-ar opri cu eroare, nu ar mai scrie niciun răspuns. De aceea există trei trepte:

| Ce se întâmplă | Ce face programul |
|---|---|
| Căutarea normală dă eroare | Rulează o căutare simplă, cu o singură măsură: partea de energie la 15-150 Hz. |
| Și căutarea simplă dă eroare | Răspunde cu 40% și 50% din durata fișierului. |
| Fișierul nu poate fi citit | Lucrează pe 10 secunde de liniște și scrie tot un CSV valid. |

### Cele două schimbări față de arhivă

`diff arhiva/main.py main.py` arată două rânduri noi de setări și locurile în care sunt folosite:

```
MELODY = {'f0log', 'centroid_f0', 'flux', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6'}
MAX_BLOCK_S = 20.0
```

| Schimbare | Care era problema | Efect singură | Efect împreună |
|---|---|---|---|
| Fără cele 9 măsuri de melodie | Un pasaj al gazdei cântat altfel arăta ca o bucată străină. Marginile se mutau. | 327 -> 332 | 327 -> 334 |
| Bloc de cel mult 20 s, nu 40 s | Blocurile lungi câștigau uneori, deși enunțul spune „câteva secunde”. | 327 -> 329 | 327 -> 334 |

Restul codului e identic, de la citirea fișierului până la scrierea CSV-ului.

## Cât de bine merge

Pe train răspunsul e 14,965 și 20,912. Adevărul e 14,960 și 20,965. Greșeala e de 0,005 s la început și de 0,053 s la sfârșit. O rulare durează circa 2 secunde.

Pe cele 344 de înregistrări de test, cu regula de 1,5 s la fiecare margine:

| Detector | Trecute din 344 |
|---|---|
| `main.py` | 334 |
| `arhiva/main.py`, neschimbat | 327 |
| `a_doua_incercare/main.py` | 312 |

Aceleași 344, împărțite o dată după felul bucății lipite și o dată după lungimea ei:

| Împărțire | Grup | Cazuri | Arhiva | `main.py` |
|---|---|---|---|---|
| După fel | Alt instrument | 104 | 103 | 104 |
| După fel | Același instrument, alt lanț de înregistrare | 104 | 98 | 103 |
| După fel | Același instrument, același lanț | 104 | 97 | 98 |
| După fel | Făcute din train | 32 | 29 | 29 |
| După lungime | Sub 2,5 s | 30 | 22 | 25 |
| După lungime | 2,5-5 s | 72 | 70 | 71 |
| După lungime | 5-10 s | 188 | 182 | 185 |
| După lungime | 10-20 s | 54 | 53 | 53 |

Bucățile aflate chiar la începutul sau la sfârșitul fișierului trec toate, 48 din 48. Pentru precizie am luat, la fiecare caz trecut, marginea mai greșită dintre cele două. Mediana, adică valoarea de la mijlocul listei, e 0,04 s. 95% din cazuri sunt sub 0,14 s. Cea mai mare greșeală e 1,17 s, iar peste 1,0 s sunt 1,8% din cazuri. Deci un caz trecut e de obicei departe de limita de 1,5 s.

Cele 10 înregistrări care pică și acum nu sunt „aproape bune”. La 9 din ele răspunsul e în alt loc din fișier:

| Grup | Fel | Lungime | Adevăr | Răspuns | Ce s-a întâmplat |
|---|---|---|---|---|---|
| F | făcută din train | 5,0 s | 6,00-11,00 | 30,01-34,01 | alt loc |
| F | făcută din train | 5,0 s | 25,00-30,00 | 44,00-48,05 | alt loc |
| F | făcută din train | 5,0 s | 40,00-45,00 | 10,07-13,92 | alt loc |
| Primele 140 | același instrument, același lanț | 2,6 s | 17,02-19,67 | 0,02-16,36 | alt loc |
| Scurte | același instrument, alt lanț | 1,6 s | 25,73-27,30 | 30,21-40,00 | alt loc |
| Scurte | același instrument, același lanț | 1,6 s | 50,24-51,87 | 3,22-11,92 | alt loc |
| Scurte | același instrument, același lanț | 1,7 s | 3,97-5,71 | 20,26-32,16 | alt loc |
| Scurte | același instrument, același lanț | 1,6 s | 49,02-50,59 | 22,06-42,05 | alt loc |
| Scurte | același instrument, același lanț | 2,4 s | 14,89-17,28 | 36,06-51,00 | alt loc |
| Lungi 10-14 s | același instrument, același lanț | 12,4 s | 16,72-29,07 | 20,01-29,06 | început greșit cu 3,29 s |

## Cum a fost verificată

Toate scripturile sunt în `check/`. Ce au scris ele e în `check/results.txt`. Au fost rulate pe 15 noiembrie 2026, cu Python 3.13 pe Windows.

### 1. Înregistrările de test (`check/build_cases.py`)

Scriptul face 345 de înregistrări. 344 au o bucată lipită cu poziție cunoscută. Una e de control și nu are nimic lipit. 32 sunt făcute din train. Folosesc gazda din train, cu bucata lipită scoasă. Arhiva și `main.py` trec exact aceleași cazuri:

| Grup | Cazuri | Cum e făcut | Trecute |
|---|---|---|---|
| A | 1 | fișierul train așa cum e | 1 |
| B | 7 | bucata lipită din train, mutată în alt loc din gazdă | 7 |
| C | 15 | notele gazdei rămân; peste ele se adaugă doar partea sub 150 Hz a bucății lipite, la trei volume | 15 |
| D | 3 | muzica bucății lipite, dar cu frecvențele joase ale gazdei | 3 |
| E | 3 | bucata lipită, plus o momeală de 4 s în alt loc, adică o a doua zonă care sună altfel | 3 |
| F | 3 | o îmbinare slabă, doar în frecvențele joase, plus un pasaj al gazdei urcat cu o octavă ca momeală | 0 |

312 sunt sintetice și vin din generatorul `arhiva/validation/synth.py`. El face melodii de masenqo și de washint de 45 până la 62 de secunde. Apoi le trece printr-un lanț de înregistrare ales la întâmplare. Lanțul poate adăuga ecou de cameră, brum de rețea, zgomot de fond sau huruit jos. Din cele 312, 108 intră în grupul „Primele 140”. Restul de 204 sunt celelalte coloane din tabelul cu variante, de la „Noi 2,5-10 s” până la „La sfârșit”.

Fiecare grup sintetic are trei feluri de bucăți, în părți egale. La „alt instrument” diferă instrumentul și lanțul. La „același instrument, alt lanț” diferă doar lanțul. La „același instrument, același lanț” diferă doar felul de a cânta.

### 2. Comparația și efectul fiecărei schimbări (`check/compare.py`, `check/ablation.py`)

`compare.py` cheamă funcția `detect()` din `arhiva/main.py` și din `main.py` pe fiecare înregistrare. Măsurile sunt calculate o singură dată. Scriptul verifică întâi că funcția care le calculează e identică în cele două fișiere. Rezultatul e 327 pentru arhivă și 334 pentru `main.py`. Pe înregistrarea de control `main.py` răspunde 13,17-17,86, iar arhiva 16,16-17,81.

`ablation.py` reface căutarea cu opțiuni diferite și scoate tabelul cu variante. Are și o verificare încrucișată. Primul lui rând trebuie să dea cifra arhivei din `compare.py`, adică 327. Rândul marcat `= ../main.py` trebuie să dea 334. Amândouă se potrivesc.

### 3. Formate de fișier (`check/formats.py`)

Scriptul rulează `main.py` la fel ca verificatorul. Copiază `main.py` într-un folder gol. Scrie acolo înregistrarea cu numele `test-1-input.wav`. Rulează `python main.py test-1-input.wav` și citește `detection.csv`. Sunt 19 variante ale fișierului train:

| Variante | Răspuns | Verdict |
|---|---|---|
| 12 variante: alt număr de biți, numere zecimale, stereo, alt volum, nivel mediu decalat, undă întoarsă, doar primele 30 s | 14.965, 20.912 | bun |
| 44,1 kHz | 14.971, 20.921 | bun |
| 16 kHz și 8 kHz | 14.939, 20.889 | bun |
| Doar secundele 12-24, cu adevărul la 2,96 și 8,965 | 2.970, 8.917 | bun |
| 3 s de liniște | 0.000, 1.500 | CSV valid |
| Un fragment de 0,2 s | 0.021, 0.200 | CSV valid |
| Un fișier care nu e audio | 0.000, 1.500 | CSV valid |

Toate cele 19 rulări ies cu cod 0. O rulare durează între 1,3 și 3,3 secunde. Programul nu scrie niciun alt fișier în afară de `detection.csv`.

## Ce nu e sigur

Lucruri care pot pica un test ascuns:

- Bucăți foarte scurte, sub 2,5 s: trec 25 din 30.
- Același instrument prin același lanț de înregistrare: trec 98 din 104. `DETALII.md` spune că acesta e cel mai probabil mod în care poate pica un test ascuns.
- Două zone ciudate în același fișier: câștigă cea mai puternică. La grupul F rezultatul e 0 din 3, înainte și după schimbări. Momeala de acolo arată ea însăși străină, deci testul e aspru.
- O bucată lipită mai lungă de 20 s nu mai poate fi găsită.
- Detectorul nu are o măsură de încredere. Răspunde mereu cu un bloc, chiar dacă nu e nimic lipit.

Limite ale verificării:

- Toate cifrele sunt pe înregistrări făcute de noi, nu pe cele trei ascunse.
- 312 din cele 344 sunt sintetice. Sunt făcute de un program, nu cântate la instrumente reale.
- Cele 32 făcute din train folosesc aceeași gazdă și aceeași bucată lipită.
- Nimic nu a fost reglat special pentru bucățile scurte sau pentru cazul cu același lanț. Orice reglaj s-ar fi potrivit pe generatorul sintetic, nu pe înregistrările organizatorilor.
- Cifra 312 pentru `a_doua_incercare/main.py` apare doar în `README.md`. `check/compare.py` compară doar arhiva cu `main.py`, iar `check/results.txt` nu are o coloană pentru acest detector.

Lucruri care nu se știu:

- De ce a picat detectorul vechi la test-2. Platforma arată doar numele testului.
- Pe ce fel de cazuri pică `a_doua_incercare/main.py`. Sursele dau doar totalul.
- Ce urmează dacă platforma respinge ambele fișiere.

Locuri în care sursele nu se potrivesc:

- Ce s-ar fi întâmplat cu forma în doi pași pe platformă. Cele două teste picate sunt ale primelor versiuni: `ec2b674` la test-3 și `b061ff3` la test-2. Forma în doi pași din `varianta_veche/` nu a fost trimisă.
- Scorul minim pe îmbinările sintetice asemănătoare cu train. `varianta_veche/README.md` spune 1,56 ori. Textul de la începutul fișierului `varianta_veche/main.py` spune 1,74 ori.
- Numărul de măsuri. `arhiva/README.md` spune „circa 40”. `DETALII.md` spune 37 calculate și 28 folosite, iar codul confirmă.
- Durata unei rulări. `arhiva/README.md` spune circa 1,3 s. `check/results.txt` arată 2,2 s pe calculatorul nostru.

## Unde e fiecare lucru

Căile sunt față de folderul taskului.

| Fișier sau folder | La ce folosește |
|---|---|
| `main.py` | detectorul de trimis |
| `README.md` | descrierea scurtă, cea mai nouă |
| `DETALII.md` | descrierea lungă mai veche, în engleză, cu toate tabelele |
| `EXPLICATIE.md` | documentul de față |
| `a_doua_incercare/main.py` | al doilea detector; se încarcă doar dacă primul e respins |
| `check/build_cases.py` | face cele 345 de înregistrări de test și le păstrează măsurile |
| `check/compare.py` | rulează arhiva și `main.py` pe fiecare înregistrare și le compară |
| `check/ablation.py` | arată ce face fiecare schimbare luată separat |
| `check/formats.py` | rulează `main.py` ca verificatorul, pe 19 formate de fișier |
| `check/common.py` | cod comun: încarcă cele două detectoare și generatorul, ține toleranța de 1,5 s |
| `check/results.txt` | ce au scris cele patru scripturi la rularea din 15 noiembrie 2026 |
| `arhiva/main.py` | detectorul colegului, baza pentru `main.py` de acum |
| `arhiva/README.md` | descrierea colegului: semnele din train, metoda, verificarea lui |
| `arhiva/train-output.csv` | răspunsul corect pentru train: 14.960, 20.965 |
| `arhiva/analysis/` | șase scripturi de explorare a fișierului train: `feat_scan.py`, `pitch.py`, `plot1.py`, `plot2.py`, `plot4.py`, `plot7.py` |
| `arhiva/validation/synth.py` | generatorul de înregistrări sintetice, folosit și de `check/` |
| `arhiva/validation/synth2.py` | a doua serie de cazuri sintetice ale autorului |
| `arhiva/validation/harness.py`, `harness2.py` | cazuri făcute din materialul train, lipit din nou; al doilea script face variantele mai grele |
| `arhiva/validation/perturb.py` | train cu zgomot, filtre, altă rată de eșantionare și alt volum |
| `arhiva/validation/control.py` | controale pe train: fără grupuri de măsuri, lungime cu număr prim de eșantioane, sunet întors |
| `arhiva/validation/evalrun.py` | rulează un detector pe o listă de cazuri și numără câte trec |
| `arhiva/figures/` | trei imagini: `train_overview.png` cu tot fișierul, `splice_zoom.png` cu zona lipită, `dc_drift_vs_pitch.png` cu deriva lentă față de notă |
| `varianta_veche/main.py` | detectorul nostru vechi, forma în doi pași |
| `varianta_veche/README.md` | descrierea lui scurtă, cu cifrele de verificare |
| `varianta_veche/validate.py` | verificarea lui pe seturile „realistic” și „tough” |
| `.venv/`, `__pycache__/` | fișiere ale mediului Python de pe calculator; nu se trimit |
| `../task_ethiopia/` | `train-input.wav` de la organizatori și `train-output.csv` cu răspunsul lui |
| `NOTA.md` din folderul `handoff_fixed` | nota scrisă la reparație; nu e în repo, e pe calculatorul de lucru |

## Cum refaci totul

Ai nevoie de Python cu numpy și scipy.

1. Răspunsul pe train. Copiază `main.py` și `../task_ethiopia/train-input.wav` într-un folder gol. Dă fișierului audio numele `test-1-input.wav`. Rulează acolo `python main.py test-1-input.wav`. În `detection.csv` trebuie să apară `14.965,20.912`.
2. Cele 344 de teste. Rulează comenzile de mai jos și compară ce iese cu `check/results.txt`.
3. Fișierul de trimis. `diff arhiva/main.py main.py` trebuie să arate doar cele două schimbări. `sha256sum main.py` trebuie să dea valoarea din prima secțiune. În PowerShell comanda este `Get-FileHash main.py -Algorithm SHA256`.

```
cd check
python build_cases.py ../../task_ethiopia/train-input.wav <folder cache> 3
python compare.py <folder cache>
python ablation.py <folder cache>
python formats.py ../../task_ethiopia/train-input.wav <folder de lucru gol>
```

| Comandă | Ce să aștepți |
|---|---|
| `build_cases.py` | circa 5 minute cu 3 procese; scrie `cases.pkl` de circa 115 MB |
| `compare.py` | rândul `ALL 344 327 334`, apoi `fixed by the change: 7` și `broken by the change: 0` |
| `ablation.py` | tabelul cu variantele; rândul `= ../main.py` se termină cu 334 |
| `formats.py` | 19 rânduri, toate cu `exit 0` |

De ținut minte la rulare:

- Ține folderul de cache în afara repo-ului, fiindcă are 115 MB.
- `formats.py` șterge tot ce găsește în folderul de lucru. Dă-i un folder gol, de care nu ai nevoie.
- `formats.py` primește opțional un al treilea argument: alt `main.py` de testat. Așa poți verifica și `a_doua_incercare/main.py`.

Ce nu se poate reface direct:

- Cifra 312. În `check/` nu există un script care o scoate.
- Verificarea autorului arhivei. Scripturile din `arhiva/analysis/` și `arhiva/validation/` au căi fixe din mediul în care au fost scrise. Trebuie schimbate înainte de rulare. Excepție face `validation/synth.py`, pe care `check/` îl folosește așa cum e.
- Verificarea detectorului vechi. `varianta_veche/README.md` dă comanda `python validate.py ../../task_ethiopia 20 3 -v`. Sursele nu spun dacă a mai fost rulată de când fișierele stau în `varianta_veche/`.
