# Kazahstan: Stamp of Disapproval. Explicația completă

Documentul explică taskul, încercările care nu au mers și soluția care a dat flag-ul. Cifrele vin din fișierele acestui folder și din verificările rulate de noi. Unde un lucru nu apare în ele, scrie că nu se știe.

## Ce se pune pe site

La LMML 2026 fiecare țară e un task. Un task se predă pe site ca fișier sau ca flag de forma `SIGMOID_<CUVÂNT>`. Pe lângă răspuns se dă un link GitHub către descriere.

Aici răspunsul e un flag: `SIGMOID_SARSA`. Nu se știe dacă platforma l-a acceptat. Linkul GitHub exact nu apare în surse. Cuvântul are cinci litere, câte una pe album: S A R S A. SARSA este numele unui algoritm de învățare prin recompensă (`State-Action-Reward-State-Action`).

## Ce cere taskul

Textul original al enunțului nu este în folder. Descrierea de mai jos vine din `README.md` și din `DETALII.md`. Datele sunt în `../task_kazakhstan/`.

| Ce primim | Câte | Unde |
|---|---|---|
| Albume de timbre | 5 | `albums/album_1` până la `album_5` |
| Pagini într-un album | 26 | fișiere `page_<număr>.jpg` de 1600×1200 pixeli |
| Timbre pe o pagină | cam 3 | puse pe un fundal închis |
| Exemple cu răspuns cunoscut | 30 | `exhibits/` |

Pe fiecare pagină exact un timbru a fost modificat, într-un singur loc mic. Sunt trei feluri de modificări:

1. O cifră schimbată. De exemplu valoarea timbrului devine 202.
2. Un element copiat în alt loc al timbrului.
3. O inscripție mică ștearsă. De obicei e anul emiterii sau numele desenatorului.

Un exemplu este un timbru modificat, singur, fără pagină. În fișiere se numește „exhibit”. Fiecare exemplu vine cu o mască: o poză alb-negru care arată exact locul modificat. Cele 30 de exemple sunt singurele răspunsuri cunoscute.

```
pagină -> cam 3 timbre -> 1 timbru modificat -> 1 loc mic -> 1 semn pe foaie
26 de pagini ale unui album -> 26 de semne pe aceeași foaie -> 1 literă
5 albume -> 5 litere -> SIGMOID_<CUVÂNT>
```

Foaia are mărimea unei pagini. `DETALII.md` pomenește trei reguli: nu există copii neatinse ale timbrelor modificate, compararea cu o copie nu e admisă, iar numele fișierelor nu poartă niciun sens.

Partea grea: locul modificat are cam 1 până la 2% din suprafața timbrului. Trebuie nimerit și timbrul, și locul din el. O literă nu se vede dacă doar jumătate din puncte sunt bune. Organizatorii au dat și un cod de pornire, `../task_kazakhstan/baseline_solution.py`. El salvează pagina din nou ca JPEG și caută unde se schimbă cel mai mult. Comentariul din fișier spune că nu rezolvă taskul.

## Ce am aflat din date

Primele fapte vin din încercările noastre vechi (`DETALII.md`). Ultimele două vin din soluția finală (`solutie/README_original_en.md`).

**Câte timbre sunt.** Tăierea simplă a paginilor dă 387 de timbre. Tăierea din `vlm/crops.py` dă 394 de decupaje, pentru că taie 3 pagini în 4 sau 5 bucăți.

**Ce fel de modificări au exemplele.** Felul se vede din mască. `DETALII.md` spune într-un loc 18 inscripții șterse și în altul 9. Am folosit 18, din partea mai nouă a fișierului.

| Fel | Câte din 30 | Care exemple |
|---|---|---|
| Cifră schimbată | 10 | 2, 7, 9, 12, 13, 17, 22, 24, 25, 26 |
| Element copiat | 2 | 15, 19 |
| Inscripție ștearsă | 18 | restul |

**Cifrele schimbate dau valori care nu există.** Cifra nouă nu e o copie pixel cu pixel a altei cifre de pe timbru. A fost redesenată.

| Unde | Valori citite |
|---|---|
| Exemple | 55, 66, 99, 133, 155, 110, anul 2211 |
| Pagini de album | 202 (de 6 ori), 110, 33, 550, 155, 66, 88, 22, 101, 55, 188, 244, 252, 303, 212, 232, 44, 99 |

**Inscripțiile șterse sunt umplute în trei feluri.** Cu o culoare constantă. Cu o culoare constantă plus zgomot alb slab, cu abatere standard între 2 și 3. Sau cu o umplere netedă. Zgomot alb înseamnă că fiecare pixel e abătut independent de vecinul lui. La tiparul adevărat pixelii vecini seamănă între ei. Măsura asemănării se numește autocorelație. E între +0.15 și +0.3 la tiparul adevărat și cam −0.13 la zgomotul alb.

**Desenele nu se repetă.** Din 417 timbre, 404 apar o singură dată. Cele 417 sunt timbrele din albume plus cele 30 de exemple. Nu ai deci cu ce compara un timbru.

**Timbrul modificat stă de obicei în mijlocul paginii.** Am numărat asta pe 42 de pagini unde știam timbrul modificat (vezi încercarea 4). E cel mai aproape de centrul paginii pe 26 de pagini, al doilea pe 14 și cel mai departe pe 2. La întâmplare ar fi ieșit câte 14. Distanța lui mediană până la centrul paginii e 276 de pixeli, față de 442 la timbrele vecine. Explicația: litera e desenată în mijlocul foii. Faptul ajută doar la ghicit timbrul. Nu arată locul din timbru.

**Fiecare timbru poartă urma unei salvări JPEG mai vechi.** Pe acest fapt stă soluția finală. Fiecare timbru a fost salvat ca JPEG înainte de a fi modificat, la o calitate între 82 și 92. Apoi a fost pus pe pagină, iar pagina a fost salvată la calitatea 97. Locul modificat nu poartă urma primei salvări. Secțiunea „Soluția finală, pas cu pas” explică de ce.

**Zgomotul din locurile umplute e altfel.** Unele umpleri sunt prea netede, de exemplu valoarea 254 constantă în hârtia albă. Altele au zgomot adăugat artificial. La el cele trei canale de culoare se mișcă independent: corelația e cam 0.1, față de cam 0.9 la zgomotul timbrului. Umplerilor din hârtia albă le lipsesc și pixelii de exact 255 pe care îi are hârtia adevărată.

## Ce am încercat, în ordine

Primele șase încercări sunt ale noastre și sunt descrise în `DETALII.md`. A șaptea e soluția care a mers.

```
1. rețea antrenată pe modificări făcute de noi       -> foi fără litere
2. indicii clasice, măsurate pe exemple              -> cel mult 8 din 30
3. căutarea unei copii a fiecărui timbru             -> doar 12 timbre au copie
4. citirea timbrelor cu un model de imagine și text  -> 42 de pagini de încredere din 130
5. potrivirea literelor pe punctele găsite           -> la fel ca puncte puse la întâmplare
6. verificări suplimentare                           -> nicio literă sigură
7. urma primei salvări JPEG plus un clasificator     -> S A R S A
```

### 1. Rețeaua antrenată pe modificări făcute de noi

Fișiere: `forgery_net.py`, `detect_pages.py` și `sheets.py`. Rezultate: `outputs_v1/` și `outputs_v2/`. Problema: 30 de exemple sunt prea puține ca să înveți o rețea neuronală. Am fabricat exemple în plus.

1. Am tăiat timbrele din paginile albumelor și am făcut pe fiecare o modificare artificială. Cifră schimbată: un petic cât o cifră, copiat alături. Element copiat: o pată cu margini moi, copiată în alt loc. Inscripție ștearsă: un dreptunghi umplut cu culoarea marginii lui.
2. Am salvat timbrul din nou ca JPEG, la o calitate între 90 și 98.
3. Am antrenat de la zero o rețea de tip U-Net. Ea primește timbrul și scoate pentru fiecare pixel un număr: cât de probabil e modificat.
4. Cele 30 de exemple adevărate au intrat și ele în antrenare, în 25 până la 40% din loturi.
5. Pe fiecare pagină am luat locul cu răspunsul cel mai puternic al rețelei.

**Ce a ieșit.** Am ținut deoparte 8 exemple și am antrenat pe celelalte 22. Rețeaua a găsit locul la 2 până la 4 din cele 8 (`train_log_val.txt`). Găsea bine inscripțiile șterse de pe hârtie simplă. Se aprindea însă și pe colțuri simple neatinse, iar cifrele schimbate le rata aproape mereu. Rețeaua antrenată pe 22 de exemple și cea antrenată pe toate 30 cădeau de acord pe 8 până la 13 pagini dintr-un album. Nicio foaie nu arăta o literă.

**De ce nu a mers.** Am măsurat mai târziu, pe cele 42 de pagini de încredere de la încercarea 4. Rețeaua alegea același timbru ca citirea valorilor pe 18 din 42, iar la întâmplare ar fi ieșit 14. Alegea deci timbrul aproape la noroc. Un fapt din date se leagă de asta: noi copiam cifra, iar cifrele adevărate sunt redesenate. Altă cauză nu e scrisă în surse.

### 2. Indicii clasice, măsurate pe exemple

Un indiciu clasic e o măsură simplă calculată direct din pixeli, fără învățare. Pentru fiecare am numărat la câte exemple punctul lui cel mai puternic cade în mască.

| Indiciu | Fișier | Nimerite din 30 |
|---|---|---|
| Zonă copiată: blocuri de 12×12 care seamănă cu altele din timbru | `cues.py` | 1, sau 0 fără contururi |
| Zonă prea netedă față de granulația timbrului | `cues.py` | 4 |
| Umplere cu zgomot alb pe zone netede | fără fișier | 8 |
| Nivel de zgomot nepotrivit cu luminozitatea | `noise_cue.py` | 7 |
| Grila veche de blocuri JPEG, decalată | abandonat | 5 |
| Exces de detaliu fin în culoare | abandonat | 2 |

**Ce a ieșit și de ce nu a mers.** Cel mai bun indiciu nimerea 8 exemple din 30. Pe paginile albumelor dădea puncte împrăștiate. Fiecare indiciu e făcut pentru un singur fel de modificare. Rândul cu grila JPEG e cel mai apropiat de soluția finală. Atunci am căutat doar decalajul grilei și am renunțat la 5 din 30. Soluția finală măsoară altceva: dacă fiecare număr dintr-un bloc stă pe treptele primei salvări.

### 3. Căutarea unei copii a fiecărui timbru

Ideea: dacă același desen apare de două ori, suprapui cele două timbre și vezi unde diferă. `copies.py` a comparat fiecare pereche dintre cele 387 de timbre. 368 de perechi s-au aliniat pe ceva comun, de obicei un text sau un chenar. Doar 6 perechi sunt același desen, cu potrivire peste 0.9. Asta înseamnă 12 timbre, pe cel mult 12 pagini din 130. `copy_residual.py` a calculat diferența pentru ele (`copy_residual_log.txt`). **De ce nu a mers.** 12 pagini sunt prea puține pentru cinci litere. Celelalte 375 de timbre nu au nicio copie în albume. În plus, regulile exclud compararea cu o copie.

### 4. Citirea timbrelor cu un model de imagine și text

Fișiere: folderul `vlm/`. Modelul e Qwen2.5-VL-7B-Instruct, rulat local. Un astfel de model primește o poză și o întrebare și răspunde în text. Durează cam 5 secunde pe timbru. Ideea: o cifră schimbată dă o valoare care nu există, de exemplu 202 sau 155. Citești valoarea de pe fiecare timbru și o cauți pe cea imposibilă.

1. `crops.py` taie fiecare pagină în timbre.
2. `vlm_fast.py` citește de pe fiecare timbru țara, valoarea și anii. `vlm_read.py` e prima variantă, folosită pe exemple. Ea mai întreabă „a fost modificat?” și „unde?”.
3. `content_rules.py` scoate indicii din ce s-a citit. Indiciu tare: o valoare cu cifră repetată sau copiată, ori un an după 2026. Indiciu slab: o valoare care nu se împarte la 5.
4. `ground.py` cere modelului dreptunghiul în care e scris numărul ciudat. Centrul lui e punctul de pe pagină.
5. `yearq.py` întreabă „e tipărit anul emiterii?” și păstrează probabilitatea răspunsului „nu”.
6. `decide.py --mode tiered` hotărăște pagina pe trei niveluri de încredere.

| Nivel | Când se aplică | Unde pune punctul | Pagini |
|---|---|---|---|
| De încredere | un singur timbru de pe pagină are indiciu tare | pe numărul ciudat | 42 |
| Mediu | anul lipsește clar la un timbru: probabilitate de cel puțin 0.9 și cu 0.3 peste următorul | pe vârful rețelei de la încercarea 1, în acel timbru | 32 |
| Slab | niciuna din cele de sus | pe vârful rețelei de la încercarea 1 | 56 |

| Verificare pe exemple | Rezultat |
|---|---|
| Indiciul tare se declanșează la cifre schimbate | 8 din 10 |
| Indiciul tare se declanșează la ștergeri și copieri | 0 din 20 |
| „Anul lipsește”, la inscripții șterse | 13 din 18 |
| Aceeași întrebare pe timbre de album presupuse neatinse | alarmă falsă la 34% |
| „A fost modificat?” | 2 din 30 |
| „Unde e modificarea?” | 4 din 30 |

**Ce a ieșit pe albume.** Indiciul tare a apărut la exact un timbru pe 42 de pagini și niciodată la două. Cam o treime din modificări sunt cifre schimbate, deci ne așteptam la vreo 43. Cele 42 sunt foarte probabil corecte. Nivelul mediu era corect în cam 70% din cazuri, după estimarea de atunci. Nivelul slab era aproape de noroc. **De ce nu a mers.** Metoda vede doar cifrele schimbate. Două treimi din pagini au ștergeri sau copieri. Acolo nu știam sigur nici timbrul, nici locul din el.

### 5. Potrivirea literelor pe punctele găsite

`vlm/fit_letters.py` ia punctele de nivel „de încredere” și „mediu” ale unui album. Încearcă toate literele de la A la Z, la mai multe mărimi și poziții. Scorul e distanța dintre puncte și liniile literei, măsurată în ambele sensuri. Mai mic înseamnă mai bine. Pentru comparație face același lucru cu tot atâtea puncte puse la întâmplare.

| Album | Cea mai bună literă | Scor | Scor cu puncte la întâmplare |
|---|---|---|---|
| 1 | C | 20.5 | 21.9 ± 1.5 |
| 2 | M | 20.6 | 23.3 ± 2.3 |
| 3 | R | 17.9 | 18.7 ± 1.9 |
| 4 | D | 24.7 | 25.6 ± 1.2 |
| 5 | A | 17.1 | 15.7 ± 1.1 |

**Ce a ieșit și de ce nu a mers.** Niciun album nu bate punctele la întâmplare. „CMRDA” nu e o presupunere, e doar cel mai bun scor dintr-un zgomot. Nici cu ochiul liber nu se citea vreo literă. Asta e valabil și pentru albumul 4, care avea 13 puncte de încredere. Jumătate din punctele unei litere nu arată litera, iar restul punctelor erau puse aproape la noroc.

### 6. Verificări suplimentare

Am căutat o greșeală de alt fel sau un cod ascuns.

- Coordonatele sunt bune. Toate paginile au 1600×1200, iar punctele sunt în pixelii paginii.
- Nu contează al câtelea e timbrul în rând. La albumul 4 timbrul modificat e pe cele trei poziții de 4, 5 și 4 ori.
- Nu contează numărul paginii. Legătura lui cu poziția punctului își schimbă semnul de la un album la altul.
- Un indiciu de zonă netedă, `vlm/patch_cue.py`, nimerește masca la 8 exemple din 30. Toate sunt ștergeri. La alegerea timbrului e slab.

Am făcut și o foaie combinată, `vlm/album_sheets_bayes.png`. Pe paginile fără indiciu tare ea alege timbrul după cât e de central și după întrebarea despre an. Punctul e pus pe zona cea mai netedă a timbrului. Locul din timbru e corect de cel mult 4 ori din 10. **Ce a ieșit.** Albumul 5 începea să semene cu un A. Albumul 2 putea fi N sau V. Albumele 1, 3 și 4 nu se citeau. De aceea `DETALII.md` scrie la început „Flag: not found”. Fișierul e mai vechi decât soluția, iar `README.md` e mai nou și dă flag-ul.

### 7. Soluția care a mers

Un coleg de echipă a scris soluția din `solutie/`. Ea nu citește timbrul și nu învață din modificări fabricate. Caută urma primei salvări JPEG a fiecărui timbru. Urma lipsește exact în locul modificat, la toate cele trei feluri de modificări. Unde urma e prea slabă hotărăște un clasificator. Foile ei se citesc: S A R S A.

## Soluția finală, pas cu pas

### Cum salvează JPEG o poză

1. Poza e împărțită în pătrate de 8×8 pixeli. Le numesc blocuri.
2. Fiecare bloc e rescris ca 64 de numere. Primul spune cât de luminos e blocul în medie. Celelalte spun cât de repede se schimbă luminozitatea în bloc.
3. Fiecare număr e împărțit la o treaptă și rotunjit la întreg. Se păstrează doar întregul.
4. La deschidere întregul e înmulțit înapoi cu treapta.

După o salvare, fiecare număr e deci un multiplu al treptei lui. Mărimea treptelor e dată de calitate, un număr ales la salvare. Calitate mare înseamnă trepte mici și pierdere mică. După formula din `solutie/code/dq.py`, treapta primului număr e 6 la calitatea 82, 3 la calitatea 92 și 1 la calitatea 97.

### Un exemplu cu numere

Numerele de aici sunt alese ca exemplu. Nu sunt din date. Treapta este 6.

| Numărul înainte de salvare | Împărțit la 6 | Rotunjit | Numărul după salvare |
|---|---|---|---|
| 13.7 | 2.28 | 2 | 12 |
| −8.2 | −1.37 | −1 | −6 |
| 20.9 | 3.48 | 3 | 18 |

După salvare toate numerele sunt multipli de 6. Acum cineva modifică un loc de pe timbru. Apoi timbrul e pus pe pagină și pagina e salvată la calitatea 97. La 97 treptele sunt mici, deci numerele se mișcă foarte puțin. Măsurăm din nou numerele în două blocuri:

| Bloc | Numere măsurate | Distanța până la cea mai apropiată treaptă |
|---|---|---|
| neatins | −5.8, 12.3, 18.4 | 0.2, 0.3, 0.4 |
| modificat | −2.9, 9.2, 15.1 | 2.9, 2.8, 2.9 |

```
trepte:      -6         0         6        12        18
neatins:      x                             x          x
modificat:         x                   x         x
```

Blocul neatins stă lângă trepte. Blocul modificat a fost desenat după prima salvare, deci numerele lui cad oriunde. Distanța cea mai mare posibilă e 3, adică o jumătate de treaptă. Un singur număr nu dovedește nimic, pentru că poate nimeri lângă o treaptă din întâmplare. Zeci de numere din mai multe blocuri vecine dovedesc.

### Ce înseamnă dublă compresie

Dublă compresie înseamnă că aceeași poză a trecut prin două salvări JPEG. Aici drumul unui timbru a fost:

```
timbru -> salvat JPEG la calitate între 82 și 92
       -> modificat într-un loc mic
       -> pus pe pagină, la o poziție oarecare
       -> pagina salvată JPEG la calitate 97
```

1. A doua salvare are trepte mult mai mici decât prima. De aceea urma primei salvări rămâne vizibilă.
2. Blocurile primei salvări pornesc din colțul timbrului, nu din colțul paginii. Grila de blocuri a timbrului e deci decalată față de a paginii. Decalajul e între 0 și 7 pixeli pe fiecare direcție, adică 64 de variante.
3. Tot ce a fost desenat după prima salvare nu stă pe treptele ei. Asta e valabil pentru o cifră lipită, pentru un element copiat și pentru o umplere. De aceea aceeași măsură prinde toate cele trei feluri de modificări.

Metoda nu are nevoie de o copie curată a timbrului. Timbrul se compară cu el însuși: partea neatinsă arată care sunt treptele. După autorul soluției, metoda tace în două situații. Prima: modificarea e în hârtie complet albă, unde aproape nu e semnal. A doua: modificarea a fost făcută înainte de prima salvare, deci stă și ea pe trepte.

### Pașii soluției, fișier cu fișier

Toate fișierele sunt în `solutie/code/`.

```
pagină
  -> segment.py                      timbrele de pe pagină
  -> dq.py                           scorul de trepte al fiecărui bloc
  -> noise_features.py, nlf.py       măsuri de zgomot
  -> allfeatures.py, build_cache.py  132 de numere pentru fiecare pixel
  -> train_detect.py                 clasificatorul și decizia pe pagină
  -> boxes.csv                       un dreptunghi pe pagină
  -> draw_sheets.py                  o foaie pe album, pe care se citește litera
```

**1. `segment.py` taie pagina în timbre.** Problema: un prag fix de culoare pierde timbrele închise și fâșiile mici. Fundalul albumului e închis și constant de-a lungul fiecărui rând de pixeli. Codul estimează culoarea fundalului pe fiecare rând. Un pixel e „timbru” dacă vecinătatea lui diferă de fundal la culoare sau la textură.

**2. `dq.py` calculează scorul de trepte.** Lucrează doar pe luminozitate. Pentru fiecare timbru face patru lucruri:

1. Găsește decalajul grilei. Încearcă toate cele 64 de variante. O păstrează pe cea la care numerele se adună cel mai clar în jurul unor multipli.
2. Găsește calitatea primei salvări. Încearcă fiecare calitate de la 50 la 99. O păstrează pe cea la care numerele timbrului stau cel mai bine pe trepte.
3. Dă un scor fiecărui număr din fiecare bloc. Compară două explicații: „numărul stă pe o treaptă, cu o mică abatere” și „numărul poate fi oriunde”. Scorul e pozitiv lângă o treaptă și negativ între trepte. E limitat între −4 și +4.
4. Adună scorurile numerelor dintr-un bloc. Folosește primul număr și următoarele 27. Suma e scorul de trepte al blocului.

Patru cazuri sunt tratate separat:

- Un număr mai mic decât 0.6 dintr-o treaptă e sărit, pentru că nu aduce informație. La fel o treaptă mai mică de 3.
- Un bloc cu pixeli alb complet sau negru complet e sărit. Varianta strictă îl sare de la 5% astfel de pixeli, iar cea îngăduitoare de la 30%.
- Banda de 12 pixeli de la marginea timbrului e ignorată. Zimții și marginea au fost desenate după prima salvare.
- Două timbre lipite pot fi tăiate ca unul singur. Atunci codul potrivește a doua grilă pe zona mare care o contrazice pe prima.

Scorul de trepte e apoi adunat într-o fereastră de 33×33 pixeli, care acoperă cam 17 blocuri. Suma o numesc scorul ferestrei (în cod `dqL33`). Un scor al ferestrei foarte negativ înseamnă că acolo multe numere nu stau pe trepte.

**3. `noise_features.py` și `nlf.py` măsoară zgomotul.** Problema: scorul de trepte tace pe hârtia albă. Acolo rămâne altă urmă, zgomotul. Codul măsoară în jurul fiecărui pixel cât de legat e zgomotul între cele trei canale de culoare. Măsoară și cât seamănă zgomotul unui pixel cu al vecinului. Mai măsoară cât din zonă e perfect netedă și cât din hârtia albă are exact valoarea 255. Fiecare măsură e luată și ca diferență față de împrejurimi și față de restul timbrului.

`nlf.py` adaugă o corecție. Într-un timbru zgomotul normal depinde de ce e desenat: culoare, luminozitate, contururi. Codul învață această legătură chiar din timbru. Poate face asta pentru că locul modificat are doar 1 până la 2% din suprafață. Păstrează apoi ce nu se explică prin desen: zone prea netede sau prea zgomotoase pentru ce e acolo.

**4. `allfeatures.py` și `build_cache.py` strâng totul.** Pentru fiecare pixel ies 132 de numere, numite trăsături. Scorul de trepte intră în ambele variante, cea strictă și cea îngăduitoare. La exemple se salvează toată poza. La pagini se salvează fiecare al doilea pixel din interiorul timbrelor.

**5. `train_detect.py` antrenează clasificatorul.** Un clasificator e un program care învață din exemple etichetate. Aici primește cele 132 de trăsături ale unui pixel. Dă o probabilitate între 0 și 1 că pixelul e modificat. Felul folosit se numește gradient boosting: mulți arbori de decizie mici, fiecare corectând greșelile celor dinainte. Runda 0 învață doar din cele 30 de exemple. Pixelii din mască sunt „modificat”, iar cei din restul timbrului sunt „neatins”.

Problema: paginile nu arată exact ca exemplele și nu au etichete. Soluția e autoantrenarea. Clasificatorul etichetează singur paginile și apoi învață din propriile etichete. Se sprijină pe singurul fapt sigur despre pagini: fiecare are exact un loc modificat. Se fac 3 runde. O rundă arată așa:

1. Clasificatorul calculează probabilitatea pe fiecare pagină și găsește cele mai înalte vârfuri.
2. Încrederea paginii e diferența dintre primul vârf și al doilea.
3. Pixelii aflați la peste 50 de pixeli de primele trei vârfuri devin „neatins”. Se iau cel mult 1500 pe pagină.
4. Pe jumătatea de pagini cu încrederea cea mai mare, pixelii de lângă primul vârf devin „modificat”. Se iau cel mult 200 pe pagină.
5. Clasificatorul e antrenat din nou pe exemple plus acești pixeli.

**6. Decizia pe pagină** e tot în `train_detect.py`. `finalize.py` aplică aceeași regulă pe o rulare terminată.

```
cel mai negativ scor al ferestrei de pe pagină
   -> sub −50:  locul acela e răspunsul              (source = dq)
   -> altfel:   vârful clasificatorului e răspunsul  (source = classifier)
```

Prima ramură o numesc regula tare. Pe exemple, regula cu pragul −50 nu s-a declanșat niciodată în afara locului modificat. În jurul punctului ales se desenează un dreptunghi cât pata de probabilitate. Lățimea e ținută între 30 și 120 de pixeli, iar înălțimea între 24 și 100. Rezultatul e `solutie/outputs/boxes.csv`, cu un rând pe pagină. Patru rânduri din `album_1`, fără coloanele dreptunghiului:

| page | source | score | classifier_top | dq_min |
|---|---|---|---|---|
| page_1759 | dq | −293.2500 | 1.0000 | −293.2 |
| page_3151 | dq | −50.4062 | 0.9922 | −50.4 |
| page_3114 | classifier | 0.9995 | 0.9995 | −5.4 |
| page_4114 | classifier | 0.9961 | 0.9961 | −43.9 |

Coloanele `x`, `y`, `w` și `h` dau dreptunghiul. `source` spune cine a hotărât pagina. `score` e scorul ferestrei la `dq` și probabilitatea la `classifier`. `dq_min` e cel mai negativ scor al ferestrei de pe pagină. `classifier_top` e cea mai mare probabilitate dată de clasificator. Pagina `page_1759` e exemplul din descrierea autorului: singura contradicție de pe toată pagina e ultimul „2” din valoarea inexistentă 202. Pagina `page_3151` trece pragul la limită. Pagina `page_4114` rămâne la clasificator.

**7. `draw_sheets.py` desenează foile.** Pentru fiecare album pune cele 26 de dreptunghiuri pe o foaie de 1600×1200 (`sheet_album_N.png`). Face și o variantă „la lampă”: media tuturor paginilor albumului, cu dreptunghiurile peste ea (`lamp_album_N.jpg`). `overlay_glyphs.py` desenează dreptunghiurile peste forma literei citite (`letters_check.png`). Acolo negru înseamnă regula tare și roșu înseamnă clasificator.

## Cât de bine merge

Paginile nu au răspunsuri cunoscute. Nu există deci un procent de pagini corecte. Există trei măsurători ale autorului soluției și faptul că literele formează un cuvânt.

**Regula tare, pe cele 30 de exemple** (`solutie/outputs/validation/dq_rule_exhibits.txt`):

| Ce se întâmplă pe exemplu | Câte | Scorul ferestrei |
|---|---|---|
| Regula se declanșează și locul e în mască | 13 | de la −55.3 până la −207.9 |
| Regula nu se declanșează, dar locul e în mască | 3 | −10.0, −38.2, −42.2 |
| Regula nu se declanșează și locul e în afara măștii | 14 | cel mai negativ e −24.1 |

Când se declanșează, regula nu greșește: 13 din 13. Cel mai negativ scor pus greșit e −24.1, departe de pragul −50. Regula se declanșează la toate felurile de modificări: la 5 din 10 cifre schimbate, la 1 din 2 elemente copiate și la 7 din 18 inscripții șterse. Această împărțire pune alături lista de feluri din `DETALII.md` și fișierul de mai sus.

**Clasificatorul, cu testul în care scoți câte un exemplu** (`solutie/outputs/validation/loo.txt`). În engleză testul se numește „leave one out”. Scoți un exemplu și antrenezi pe celelalte 29. Apoi verifici dacă vârful de probabilitate cade în masca exemplului scos. Repeți pentru fiecare din cele 30. Rezultat: 25 din 30. Testul folosește doar runda 0, fără autoantrenare.

| Exemplu ratat | Fel | Regula tare pe același exemplu |
|---|---|---|
| 7 | cifră schimbată | nu se declanșează (−11.7) |
| 9 | cifră schimbată | se declanșează, în mască (−90.9) |
| 13 | cifră schimbată | niciun scor negativ (0.0) |
| 19 | element copiat | niciun scor negativ (0.0) |
| 25 | cifră schimbată | nu se declanșează (−24.1) |

Exemplul 9 e prins de regula tare. Rămân 4 exemple din 30 pe care nu le prinde nici regula, nici clasificatorul. Autorul explică ratările prin cele două situații în care metoda tace. **Pe pagini.** Regula tare hotărăște 58 de pagini din 130. Clasificatorul hotărăște celelalte 72. Împărțirea pe albume e în tabelul din secțiunea următoare. Scorul mediu al primului vârf crește de la 0.952 în runda 0 la 0.972 în runda 3 (`solutie/outputs/run/train_log.txt`). Numărul arată cât de sigur e clasificatorul pe propriul vârf. Nu arată dacă vârful e corect.

**Literele, verificate de autor** (`solutie/outputs/validation/letter_check.txt`). Fiecare literă mare e desenată în trei fonturi groase și potrivită peste punctele albumului. Sunt două scoruri și la amândouă mai mare înseamnă mai bine. Scorul F cere ca punctele să cadă în literă și ca toată litera să aibă puncte aproape. Al doilea scor numără doar cât din puncte cad în literă peste ce ar da norocul.

| Album | Primele două litere după F | Primele două după al doilea scor |
|---|---|---|
| 1 | S 0.54, K 0.45 | L 0.14, S 0.13 |
| 2 | A 0.70, W 0.56 | A 0.25, W 0.20 |
| 3 | R 0.88, K 0.64 | R 0.23, K 0.22 |
| 4 | B 0.84, S 0.82 | S 0.28, B 0.17 |
| 5 | A 0.80, W 0.65 | A 0.32, W 0.22 |

La albumul 4, descrierea autorului spune că B e pe locul doi după F. Fișierul arată B pe primul loc, la 0.02 de S. Am folosit fișierul.

## Cum a fost verificată

Verificările din `verificare/` au fost rulate după primirea soluției, pe copia noastră de date. Scopul a fost să nu luăm de bune cifrele altcuiva.

| Nr. | Ce s-a refăcut | Cu ce | Rezultat |
|---|---|---|---|
| 1 | Regula tare pe cele 30 de exemple | `dq_repro.py` | se declanșează la 13 și nimerește locul adevărat la toate 13 |
| 2 | Cel mai negativ scor al ferestrei, pe toate cele 130 de pagini | `dq_repro.py` | egal cu cel din `solutie/outputs/boxes.csv` la 130 din 130; diferența cea mai mare e 0.16 |
| 3 | Locul găsit pe cele 58 de pagini hotărâte de regula tare | `dq_repro.py` | cade în dreptunghiul soluției la 58 din 58 |
| 4 | Celelalte timbre de pe aceleași 58 de pagini | `dq_repro.py` | nu arată nimic: mediana e cam −3, față de cam −120 la timbrul modificat |
| 5 | Calitatea primei salvări, pe fiecare timbru | `dq_repro.py` | între 82 și 92, împărțită aproape egal |
| 6 | Comparația cu cele 42 de pagini de încredere ale noastre | `compare.py` | același timbru la 34 din 42; la întâmplare ar ieși cam 14 |
| 7 | Citirea literelor cu alte fonturi și alt scor | `letters.py` | S, A, R sau Q, S, A |

- Rândurile 1 și 2 arată că tabelul soluției chiar iese din datele noastre, cu codul ei.
- Rândurile 3 și 4 arată că regula tare găsește un singur timbru cu contradicție pe pagină. Celelalte timbre sunt curate.
- Rândul 6 compară două metode care se uită la lucruri diferite. A noastră citește valoarea tipărită. A soluției măsoară treptele JPEG și zgomotul. Totuși aleg același timbru de 34 de ori din 42.

Aceleași cifre, pe albume:

| Album | Hotărâte de regula tare | Hotărâte de clasificator | Pagini de încredere ale noastre | Din ele, același timbru |
|---|---|---|---|---|
| 1 | 11 | 15 | 6 | 6 |
| 2 | 11 | 15 | 6 | 5 |
| 3 | 13 | 13 | 11 | 9 |
| 4 | 13 | 13 | 13 | 8 |
| 5 | 10 | 16 | 6 | 6 |
| Total | 58 | 72 | 42 | 34 |

Rândul 7, în detaliu. `letters.py` folosește fonturi groase din Windows și alt scor decât autorul: distanța dintre puncte și literă, în ambele sensuri. Mai mic înseamnă mai bine. Literele din tabel ies cu toate cele 26 de puncte ale fiecărui album. La albumul 3, Q are 0.078 și R are 0.079, deci e egalitate. A doua potrivire a păstrat doar paginile regulii tari plus cele 42 de puncte de încredere ale noastre. Rămân între 13 și 21 de puncte pe album. Atunci ies doar trei litere: S la albumul 1, S la albumul 4 și A la albumul 5. Albumele 2 și 3 se citesc doar cu toate cele 26 de puncte.

Poza `verificare/litere.png` arată punctele fiecărui album peste litera potrivită. Negru e pagină hotărâtă de regula tare. Gri e pagină hotărâtă de clasificator. Cercurile roșii sunt punctele noastre de încredere. Cele 72 de pagini hotărâte de clasificator nu au fost refăcute de noi, pentru că antrenarea durează cam 40 de minute.

## Ce nu e sigur

- Nu se știe dacă platforma a acceptat `SIGMOID_SARSA`.
- Pentru cele 72 de pagini ale clasificatorului, singura măsură e testul autorului pe exemple, 25 din 30. Pe pagini nu există răspunsuri cu care să-l compari.
- Literele albumelor 2 și 3 (A și R) depind de paginile clasificatorului. Doar din punctele verificate de noi ies trei litere din cinci.
- La albumul 3, R și Q sunt la egalitate în potrivirea noastră. Cu Q nu se formează niciun cuvânt. La autor R e primul după F cu 0.88, iar Q are 0.50.
- La albumul 4, scorul F al autorului pune B (0.84) puțin peste S (0.82). Al doilea scor al lui și potrivirea noastră dau S.
- La albumul 1, al doilea scor al autorului pune L (0.14) puțin peste S (0.13). Scorul F și potrivirea noastră dau S.
- Pe 8 din cele 42 de pagini de încredere, soluția alege alt timbru decât citirea noastră. Nu se știe cine are dreptate acolo. Cinci din cele 8 sunt în albumul 4.
- Pe exemple, 4 din 30 nu sunt prinse nici de regula tare, nici de clasificator. Nu se știe câte pagini de album sunt de același fel.
- Pragul −50 a fost verificat doar pe cele 30 de exemple. Cel puțin o pagină îl trece la limită: `album_1/page_3151`, cu −50.4.
- `README.md` spune că SARSA se potrivește cu celelalte flag-uri, fiind numele unui algoritm. Acesta e un argument, nu o dovadă.

## Unde e fiecare lucru

Căile sunt față de folderul taskului.

| Folder sau fișier | La ce e bun |
|---|---|
| `README.md` | descrierea scurtă, la zi |
| `DETALII.md` | descrierea veche, în engleză, a încercărilor care nu au dat litere |
| `EXPLICATIE.md` | acest document |
| `solutie/README_original_en.md` | descrierea autorului soluției, în engleză |
| `solutie/code/` | codul soluției. Pe lângă fișierele din „Pașii soluției”: `run_all.py` rulează tot lanțul, `make_figures.py` face figurile, iar `dq_rule_check.py`, `loo_eval.py` și `letter_check.py` sunt verificările autorului |
| `solutie/outputs/boxes.csv` | dreptunghiul găsit pe fiecare pagină |
| `solutie/outputs/sheets/` | foile: `sheet_album_N.png`, `lamp_album_N.jpg`, `all_albums.png`, `letters_check.png` |
| `solutie/outputs/run/` | vârfurile din fiecare rundă (`peaks_r0.json` până la `peaks_r3.json`) și `train_log.txt` |
| `solutie/outputs/validation/` | `dq_rule_exhibits.txt`, `loo.txt`, `letter_check.txt` |
| `solutie/outputs/figures/` | `dq_exhibits.jpg`, `dq_page_album1_1759.jpg` |
| `verificare/` | `dq_repro.py` reface scorul de trepte și scrie `dq_repro.json`. `compare.py` compară cu cele 42 de pagini de încredere și verifică în același timp taskul Japoniei. `letters.py` potrivește literele și desenează `litere.png` |
| `forgery_net.py`, `detect_pages.py`, `sheets.py` | încercarea 1: modificările artificiale, rețeaua, rularea pe pagini și foile |
| `forgery_net.pt`, `forgery_net_val8.pt`, `train_log_final.txt`, `train_log_val.txt` | rețeaua antrenată pe toate exemplele și cea cu 8 exemple ținute deoparte, fiecare cu jurnalul ei |
| `outputs_v1/`, `outputs_v2/` | rezultatele pe pagini: întâi ale rețelei cu 8 exemple ținute deoparte, apoi ale celei antrenate pe toate |
| `cues.py`, `noise_cue.py` | încercarea 2: indiciile clasice |
| `copies.py`, `copies.json`, `copy_residual.py`, `copy_residual.json`, `copy_residual_log.txt` | încercarea 3: căutarea de copii |
| `vlm/` | încercările 4, 5 și 6; fișierele sunt în tabelul următor |
| `.venv/` | mediul Python local |
| `../task_kazakhstan/` | datele: `albums/`, `exhibits/` și `baseline_solution.py` |

| Fișiere din `vlm/` | La ce sunt bune |
|---|---|
| `crops.py`, `crops.json`, `crops/` | decupajele timbrelor și poziția lor pe pagină |
| `vlm_read.py`, `vlm_fast.py`, `vlm_exhibits.json`, `vlm_albums.json` | citirea timbrelor și ce s-a citit |
| `content_rules.py`, `ground.py`, `ground.json`, `ground_exhibits.json` | indiciile din citiri și dreptunghiurile numerelor ciudate |
| `yearq.py`, `yearq_albums.json`, `yearq_exhibits.json`, `log_*.txt` | întrebarea despre anul lipsă; jurnalele tuturor rulărilor |
| `net_feats.py`, `net_feats.json`, `patch_cue.py`, `patch_feats.json` | vârful rețelei vechi pe fiecare timbru și indiciul de zonă netedă |
| `decide.py`, `decisions_<mod>.json`, `boxes_<mod>.csv`, `album_sheets_<mod>.png` | deciziile și foile fiecărui mod; varianta `bayes` a fost făcută fără script |
| `fit_letters.py` | potrivirea literelor de la încercarea 5 |
| `joint_fit.py`, `joint_fit_results.json`, `ground_all.py`, `elements.json` | o potrivire de litere pe toate paginile deodată; nu e descrisă în `DETALII.md` și rezultatul ei nu apare în surse |
| `candidates.json`, `central_boxes_heat.png`, `confident_album_N.png` | nu sunt descrise în surse |

## Cum refaci totul

Comenzile se dau din folderul taskului.

```
pip install -r solutie/code/requirements.txt
python solutie/code/run_all.py --data ../task_kazakhstan --work ./work --loo
```

1. `build_cache.py` taie timbrele și calculează trăsăturile în `work/cache/`. Durează cam 30 de minute.
2. `loo_eval.py` face testul cu un exemplu scos și scrie `work/loo.txt`. Pasul rulează doar cu `--loo`.
3. `train_detect.py` antrenează clasificatorul și scrie `work/result/boxes.csv`. Durează cam 10 minute.
4. `draw_sheets.py` desenează foile în `work/result/sheets/`.

Duratele sunt cele date de autor, pe 2 nuclee de procesor. Folderul `work` nu există acum și e creat de comandă. Ai reușit dacă `work/result/boxes.csv` are 130 de pagini și pe foi se citește S A R S A. Compară-l cu `solutie/outputs/boxes.csv`. Sămânța de hazard e fixată în cod. Totuși nu se știe dacă o rulare nouă dă exact aceleași dreptunghiuri, pentru că nu am refăcut-o. Modelul antrenat și hărțile de probabilitate nu sunt în `solutie/outputs/run/`. Fără o rulare nouă, `finalize.py` nu are din ce lucra.

Verificările, întâi ale autorului și apoi ale noastre:

```
python solutie/code/dq_rule_check.py ../task_kazakhstan
python solutie/code/letter_check.py solutie/outputs/boxes.csv
python verificare/dq_repro.py solutie/code ../task_kazakhstan solutie/outputs/boxes.csv dq.json exhibits album_1 album_2 album_3 album_4 album_5
python verificare/letters.py solutie/outputs/boxes.csv vlm/decisions_tiered.json vlm/crops.json litere.png
```

- `letter_check.py` și `overlay_glyphs.py` caută fonturile la căi de Linux (`/usr/share/fonts/...`). Pe Windows trebuie schimbate căile din cod. `letters.py` le ia din `C:/Windows/Fonts`.
- `dq_repro.py` durează cam 3 minute și scrie `dq.json` în folderul curent. Rezultatul păstrat de noi este `verificare/dq_repro.json`.
- `verificare/compare.py` cere cinci fișiere. Primele trei sunt cele date și lui `letters.py`. Ultimele două sunt ale taskului Japoniei, iar calea lor nu apare în surse. `README.md` nu dă o comandă pentru el.

Încercările vechi nu sunt necesare pentru flag. Comenzile pentru `vlm/` sunt în `DETALII.md`, la „Reproduce”. Ele cer pachetele torch, transformers, bitsandbytes și qwen_vl_utils. Mai trebuie cam 6 GB de memorie video. Citirea albumelor durează cam 30 de minute. Rețeaua veche se antrenează cu `python forgery_net.py train --data ../task_kazakhstan --val 0`.
