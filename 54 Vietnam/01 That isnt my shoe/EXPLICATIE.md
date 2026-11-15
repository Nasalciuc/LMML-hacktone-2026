# Vietnam: That isn't my shoe! Explicația completă

Documentul e pentru cineva din echipă care nu a lucrat la task și trebuie să-l explice unui mentor. Varianta scurtă e în `README.md`. Numerele sunt scrise cu virgulă zecimală. În comenzi și în cod rămân cu punct, cum sunt în fișiere. Căile pornesc din folderul taskului.

## Ce se pune pe site

1. Încarci fișierul `sherlock/submission/sherlock_labels.pt`.
2. Pui linkul de GitHub către descriere, adică spre `README.md` din folderul taskului.

Platforma nu spune în ce format vrea răspunsul. Dacă dă eroare de format, încerci celelalte fișiere din același folder, în ordinea din tabel. E același model în toate patru și fiecare are circa 6,95 MB. Diferă doar felul în care e scris răspunsul.

| Ordine | Fișier | Ce întoarce pentru N poze | Când se potrivește |
|---|---|---|---|
| 1 | `sherlock_labels.pt` | o listă de N cuvinte: `nike`, `adidas`, `converse` sau `unknown` | prima încercare. Dacă platforma aștepta numere, apare o eroare clară și nu un scor greșit pe tăcute |
| 2 | `sherlock_probs4.pt` | N rânduri a câte 4 numere, în ordinea nike, adidas, converse, unknown | platforma ia poziția celui mai mare număr și o traduce în nume |
| 3 | `sherlock_probs4_a-c-n.pt` | la fel, în ordine alfabetică: adidas, converse, nike, unknown | lista de clase a platformei e alfabetică |
| 4 | `sherlock_logits3.pt` | N rânduri a câte 3 numere, fără `unknown` | platforma aplică singură un prag de încredere |

De știut înainte de încărcare:

- Nu se cunoaște niciun flag pentru taskul ăsta. Se predă fișier.
- Regulile repo-ului exclud fișierele `.pt`. Ca să ajungă pe GitHub, se adaugă cu `git add -f`.
- Nicio sursă din folder nu spune că fișierul a fost încărcat sau acceptat pe platformă. Descrierea autorilor spune că la împachetare nu exista niciun rezultat de pe platformă.
- Fișierul se deschide cu `torch.jit.load(cale)` sau cu `torch.load(cale, weights_only=False)`. `torch.load` cu setările implicite refuză orice fișier de tipul ăsta. Dacă platforma încarcă așa, niciuna din cele patru variante nu merge.

## Ce cere taskul

Primim 735 de poze de antrenare cu pantofi sport. Fiecare are 256×256 pixeli și o etichetă: `nike`, `adidas` sau `converse`. Pe acestea le numim mărci cunoscute. La test apar și poze cu alte mărci: Puma, Reebok, Vans și Asics. Modelul nu a văzut niciuna la antrenare. Pentru ele răspunsul corect e `unknown`.

Problema asta se numește recunoaștere open-set. Adică modelul alege dintre clasele pe care le știe și mai are voie să spună „nu e niciuna”. Un clasificator obișnuit nu face asta. El alege mereu una din cele trei mărci, oricât de nesigur ar fi. Sunt două porți și trebuie trecute amândouă:

| Poartă | Ce se măsoară | Prag |
|---|---|---|
| acuratețe la mărci cunoscute | din pozele cu nike, adidas și converse, câte primesc marca corectă | cel puțin 0,85 |
| respingere la alte mărci | din pozele cu alte mărci, câte primesc `unknown` | cel puțin 0,70 |

Cele două porți trag în direcții opuse. Un model care respinge mai ușor urcă la a doua și coboară la prima. În măsurătorile din acest document, o poză cunoscută respinsă se numără ca greșită.

Se predă un singur fișier `.pt` de cel mult 10 MB. Platforma îl încarcă și îl rulează singură pe pozele de test, cu torch 2.11.0. Noi nu vedem pozele de test. Platforma nu spune patru lucruri:

- în ce format vrea răspunsul (de aici cele patru fișiere);
- cum deschide fișierul;
- dacă dă modelului pozele la 256×256 cu pixelii originali sau le micșorează înainte;
- câte poze de test sunt și de unde vin cele cu alte mărci.

## Ce am aflat din date

**Cum arată cele 735 de poze.** Autorii modelului Sherlock au cercetat fișierele înainte să antreneze ceva. Au găsit așa:

| Ce | Valoare |
|---|---|
| număr de poze | 735, câte 245 pe marcă |
| mărime | toate 256×256, color |
| format | JPEG, toate salvate cu aceleași setări (calitate circa 95), fără date EXIF |
| dubluri | niciuna exactă și niciuna aproape identică |
| sursa probabilă | setul Kaggle `die9origephit / nike-adidas-and-converse-imaged`, cu poze de 240×240 și 275 pe marcă |

Setul Kaggle are 275 de poze pe marcă, iar noi am primit 245. Autorii deduc că au rămas cam 30 pe marcă pentru testul ascuns. Asta nu e confirmat.

**Urma de mărire.** Pozele din setul Kaggle au 240×240 pixeli. Ale noastre au 256×256. Deci organizatorii le-au mărit de la 240 la 256, adică de 16/15 ori. La o mărire, fiecare pixel nou e un amestec al pixelilor vechi de lângă el. La raportul 16/15, la fiecare 16 pixeli noi se consumă exact 15 pixeli vechi. Așa că rețeta de amestec se repetă din 16 în 16 pixeli. Poza rămâne cu un tipar slab cu perioada 16, pe orizontală și pe verticală. Ochiul nu îl vede. O socoteală simplă pe pixeli îl vede. Tiparul ăsta îl numim în tot documentul urma de mărire.

```
nike / adidas / converse :      set Kaggle, 240x240      --mărire de 16/15-->  256x256, are urma
puma / reebok / vans / asics :  altă sursă, altă mărime  --?-->                256x256, presupunere: nu are urma
```

Rândul al doilea e presupunerea pe care stă modelul. Setul Kaggle are doar trei mărci, deci pozele cu celelalte patru vin din altă parte. Autorii au presupus că au ajuns la 256 de la alte mărimi, fără urma aliniată de la 240 la 256. Nimeni nu a putut verifica asta pe testul real.

Că urma există și are perioada 16 am verificat noi. Am deplasat pozele taskului cu 8 pixeli și modelul a respins 99,05% din ele. Le-am deplasat cu 16 pixeli și a respins 0,54%. Conținutul pozei e același în ambele cazuri. Se schimbă doar poziția tiparului.

**Cât se poate scoate din 735 de poze.** Numerele de mai jos sunt măsurate out-of-fold. Adică împarți cele 735 de poze în 5 felii și antrenezi de 5 ori, de fiecare dată fără o felie. Fiecare poză e apoi judecată de modelul care nu a văzut-o. Obții un scor cinstit pe toate cele 735.

- O rețea antrenată de la zero ajunge la 0,796. E sub pragul de 0,85 înainte să respingă ceva (`DETALII.md`).
- O rețea preantrenată pe alte imagini, cu doar ultimul strat învățat pe pozele noastre, ajunge la 0,850. Reantrenată cu totul ajunge la 0,906 (descrierea autorilor).
- Felul în care arată pantoful nu ajunge pentru respingere. Autorii au scos pe rând câte o marcă cunoscută și au tratat-o ca necunoscută. Cu 90% din cunoscute păstrate, modelul respingea doar 0,38 până la 0,51 din marca scoasă. Pragul e 0,70. De aceea a fost nevoie de urma de mărire.

**Pozele reale cu alte mărci pe care le-am avut noi.** Pentru testul nostru am folosit setul public `ipogorelov/sneakers`, cu poze de pe internet etichetate cu marca. Sunt miniaturi mici: latura scurtă are mediana 237 de pixeli și doar 4% au cel puțin 400. Asta contează la citirea rezultatelor, pentru că urma depinde de mărimea de la care pornește poza.

## Ce am încercat, în ordine

```
1. rețea antrenată de la zero                        -> 0,796 la cunoscute, pică
2. potrivire exactă cu cele 735 de poze              -> pariu pe felul testului, fișier de 84 MB
3. rețea preantrenată + clasa a patra „unknown”      -> lăsată
4. rețea preantrenată doar pe cele 735, cu prag      -> respinge 0,29 până la 0,38 din alte mărci
5. Sherlock: rețea preantrenată + două verificări    -> varianta livrată
```

Pașii 1 până la 4 sunt modelul nostru vechi. Pasul 5 e modelul făcut de un coleg de echipă.

### 1. Rețea antrenată de la zero

1. Am definit o rețea de tip ResNet-18 și am antrenat-o de la zero pe placa video, la 192 px, timp de 250 de treceri prin date.
2. La fiecare trecere pozele erau schimbate ușor: decupate, întoarse în oglindă, rotite cu până la 20 de grade.
3. Am măsurat out-of-fold pe 5 felii. Separat, am scos câte o marcă și am tratat-o ca necunoscută.

| Ce s-a măsurat | Acuratețe la cunoscute | Respingere | Porțile |
|---|---|---|---|
| rețea de la zero, 192 px, out-of-fold, nimic respins | 0,796 | nu se aplică | pică |
| aceeași, cu 10% din cunoscutele cele mai nesigure respinse | 0,743 | nu se aplică | pică |
| rețea la 128 px cu outlier exposure, out-of-fold, nimic respins | 0,680 | nu se aplică | pică |
| o marcă scoasă și tratată ca necunoscută, 10% din cunoscute respinse | 0,63 până la 0,86 | 0,14 până la 0,28 | pică |

„Outlier exposure” din rândul al treilea înseamnă așa: arăți rețelei la antrenare și poze stricate intenționat (bucăți amestecate, blur puternic) și o înveți să fie nesigură pe ele. A ieșit mai slab. Pe mărci, cea mai bună rețea a dat: nike 0,80; adidas 0,74; converse 0,85. Problema: 735 de poze sunt prea puține ca o rețea să învețe singură cum arată o marcă.

### 2. Potrivire cu cele 735 de poze

Am făcut atunci un pariu pe felul în care e construit testul. Textul taskului vorbește despre pantoful original și despre o copie care îi seamănă și diferă doar la detalii mici. Am citit asta așa: poate pozele cunoscute din test sunt chiar poze din cele 735, iar celelalte sunt copii modificate. `main.py` face următorul lucru:

1. Calculează o sumă de control a fișierului (SHA-1) și o caută printre cele 735. Dacă o găsește, dă eticheta acelei poze.
2. Dacă nu, micșorează poza la 32×32 gri și caută cele mai apropiate 5 poze de antrenare, inclusiv în oglindă.
3. Le compară la 128×128 color. Acceptă dacă diferența medie e cel mult 2 trepte de gri și niciun pătrat de 8×8 nu diferă cu mai mult de 6.
4. Orice altceva primește `unknown`.

Pragurile vin dintr-o calibrare pe 60 de poze (`match_sim.py`). O copie resalvată, redimensionată sau întoarsă în oglindă ajunge la cel mult 1,4 diferență medie și 4,7 pe cel mai rău pătrat. O copie cu o modificare locală de 20 până la 40 de pixeli are mediana 24,6 pe cel mai rău pătrat. Cea mai apropiată poză diferită e la cel puțin 26,7.

Pe un test simulat cu 40 de poze originale și 20 de copii modificate local a ieșit 1,00 la cunoscute și 0,95 la respingere. Pe 20 de copii decupate și recolorate ale pozelor de antrenare a ieșit 0,00 la cunoscute. Asta e slăbiciunea: dacă pozele cunoscute din test sunt poze noi, metoda nu recunoaște niciuna. Rețeaua de la pasul 1 era și ea în `main.py`, dar oprită. Pornită, cu un prag strict, eticheta 12 din 20 de copii modificate și respingerea cădea la 0,40.

Forma asta nu se potrivește nici cu ce cere platforma. Se preda `main.py` împreună cu `model.pt`, iar platforma vrea un singur fișier `.pt`. `model.pt` are 84.111.335 de octeți. `DETALII.md` scrie 84 MB, iar în unități de 1024 sunt 80,2 MB. Limita e 10 MB. `DETALII.md` pomenește mărimea, dar nu spune că fișierul a fost încărcat și respins. Din surse nu se știe dacă a existat o astfel de încercare.

### 3. Rețea preantrenată cu a patra clasă

Fișierele `oe_*.py` sunt a doua încercare. Rețeaua e EfficientNet-B0, cu greutăți învățate dinainte pe ImageNet. Are patru ieșiri: adidas, converse, nike și `unknown`. Pentru clasa `unknown` a fost antrenată pe poze de pe internet cu alte mărci, din setul `ipogorelov/sneakers`. La validare, o marcă era scoasă complet din antrenare și folosită ca marcă nevăzută. Sursele nu spun de ce a fost lăsată varianta asta. Logurile ei sunt în `results/` (fișierele `oe_*`), iar unul e marcat ca întrerupt.

### 4. Rețea preantrenată doar pe cele 735 de poze, cu prag

Fișierele `ko_*.py` sunt ultima formă a modelului nostru vechi. E aceeași rețea EfficientNet-B0, reantrenată doar pe cele 735 de poze. Antrenarea a fost la 384 px, timp de 40 de treceri. Pozele de pe internet nu intră în antrenare și nu sunt folosite la alegerea vreunui prag. Regula de respingere e simplă. Rețeaua dă trei scoruri brute, câte unul pe marcă. Dacă cel mai mare e sub 1,072, răspunsul e `unknown`. Altfel e marca cu scorul cel mai mare. Pragul a fost ales ca acuratețea out-of-fold la cunoscute să rămână cel puțin 0,88.

Modelul a fost împachetat într-un singur fișier de 8,16 MB, `results/vietnam_model_r384.pt`. `README.md` îl numește `vietnam_model.pt`. Un fișier cu numele ăsta nu există în folder, iar cel măsurat e cel din `results/`. Rezultatele lui (`results/ko_eval_pt_r384.txt`):

| Pe ce poze | Rezultat |
|---|---|
| cele 735 de antrenare (văzute de rețea, deci optimist) | acuratețe 1,000; niciuna respinsă |
| 1500 de poze de pe internet cu adidas, converse și nike | acuratețe 0,813; 0,125 din ele etichetate `unknown` |
| aceleași, pe mărci | adidas 0,796; converse 0,886; nike 0,758 |
| poze de pe internet cu cele patru mărci de la test: câte sunt respinse | puma 0,293; reebok 0,378; vans 0,371; asics 0,321 |
| poze cu alte mărci (new balance, saucony și altele): câte sunt respinse | 0,426 |

Deci respinge între 29% și 38% din cele patru mărci de la test, față de pragul de 70%. Problema e aceeași ca la autorii Sherlock. Un Puma seamănă prea mult cu un Nike ca rețeaua să fie nesigură pe el.

### 5. Modelul Sherlock

Colegul nostru a pornit din alt capăt. Întâi a cercetat fișierele și a găsit urma de mărire. Apoi a construit modelul în jurul ei:

1. A comparat șapte rețele preantrenate, cu doar ultimul strat învățat pe pozele noastre. Cea mai bună a fost MobileNetV3 (0,850). Celelalte au dat între 0,766 și 0,837.
2. A reantrenat MobileNetV3 cu totul. Out-of-fold pe 5 felii: 0,887 dacă poza e micșorată la 224 px și 0,906 dacă rămâne la 256 px.
3. A măsurat cât poate respinge doar după cum arată pantoful: 0,38 până la 0,51, deci sub 0,70.
4. A construit un detector pentru urma de mărire și l-a măsurat: AUROC 0,989. AUROC e un număr între 0,5 și 1 care spune cât de bine desparte un scor două grupuri. 0,5 înseamnă la întâmplare și 1 înseamnă perfect.
5. A ales regula care combină cele două verificări și pragul ei.
6. A împachetat totul într-un fișier de 6,95 MB, în patru variante de răspuns.

Ce au încercat autorii Sherlock și au lăsat:

| Încercare | Rezultat | De ce a fost lăsată |
|---|---|---|
| v2: a doua rețea, antrenată direct la 256 px, și media celor două | pe 294 de poze: prima 0,915; a doua 0,908; împreună 0,918 | câștig de doar 0,3 puncte. Cu două rețele fișierul ajunge la 10 până la 11 MB |
| aceeași poză privită la mai multe mărimi | 0,9061 cu una; 0,9075 cu două; 0,9088 cu trei | câștig neglijabil |
| alt semn al măririi: lipsa detaliilor foarte fine din poză | AUROC 0,65 | prea slab |
| detector de urmă mai mic, cu 32 de numere în loc de 128 | AUROC 0,95 până la 0,97 | cel cu 128 e mai bun (0,989) |
| straturile cele mai sensibile ținute la precizie mai mare | amprenta se apropie de cea originală (0,953 devine circa 0,985), cu 0,1 până la 0,2 MB în plus | neaplicat. Etichetele erau oricum identice pe 60 de poze de probă |

## Soluția finală, pas cu pas

```
tensorul primit de la platformă
        |
        v
[0] aducere la formă standard: N poze, 3 culori, valori între 0 și 1
        |
        +--> [1] rețeaua MobileNetV3, pe poză și pe poza în oglindă
        |         |--> 3 scoruri brute  ------------------------------->  marca ghicită
        |         '--> amprenta pozei (1280 de numere)
        |                    |
        |                    v
        |         [2] verificarea 1: asemănarea cu cea mai apropiată din 735 amprente  --.
        |                                                                                |
        '--> [3] verificarea 2 (doar dacă poza e 256x256): scorul de urmă  -------------+
                                                                                         v
                                          [4] fiecare scor devine o valoare p: p_sem și p_trace
                                                                                         |
                                                                                         v
                                          [5] log(p_sem) + 2 * log(p_trace) >= -9.043 ?
                                                  da --> marca ghicită      nu --> unknown
```

### Pasul 0. Poza e adusă la o formă standard

Problema: nu știm cum pregătește platforma pozele înainte să le dea modelului.

- Intră: un tensor în orice format obișnuit. Numere întregi de la 0 la 255 sau numere cu virgulă. Canalele de culoare pe prima sau pe ultima poziție. O poză sau mai multe. Valori între 0 și 1, între 0 și 255 sau deja normalizate.
- Se face: modelul se uită la tip și la valorile minimă și maximă. Din ele deduce formatul și îl desface. O poză gri e copiată pe trei canale. Un al patrulea canal e aruncat.
- Iese: N poze color cu valori între 0 și 1. Ele sunt lucrate apoi în grupuri de câte 32, ca memoria să nu crească la loturi mari.

### Pasul 1. Ghicirea mărcii

Rețeaua e MobileNetV3-Large, o rețea mică pentru imagini. Vine preantrenată, adică a învățat deja forme și texturi generale pe ImageNet. Noi doar am reantrenat-o pe pozele noastre:

1. Toate cele 735 de poze, 10 treceri prin date, la 224 px.
2. La fiecare trecere poza e decupată la întâmplare (rămâne între 55% și 100% din ea) și uneori întoarsă în oglindă. Culorile sunt mișcate ușor, iar 5% din poze devin gri.
3. Pasul de învățare e mic pentru corpul rețelei (3e-4) și de zece ori mai mare pentru ultimul strat (3e-3).

La rulare:

- Intră: poza la 256 px.
- Se face: rețeaua o transformă în 1280 de numere. Face la fel cu poza în oglindă. Media celor două liste e amprenta pozei, adică un rezumat al ei în 1280 de numere. Două poze care arată asemănător au amprente apropiate.
- Iese: amprenta și trei scoruri brute calculate din ea, câte unul pe marcă. Marca cu scorul cel mai mare e marca ghicită.

Rularea e la 256 px, deși antrenarea a fost la 224, pentru că așa a ieșit mai bine: 0,906 față de 0,887. Singură, ghicirea nimerește marca la 0,9061 din poze, out-of-fold. Aici nu se respinge nimic.

### Pasul 2. Verificarea 1: seamănă cu o poză de antrenare?

Problema: ghicirea dă mereu o marcă. Ne trebuie un semn că poza e altfel decât tot ce a văzut modelul. Fișierul ține minte amprentele celor 735 de poze de antrenare. Pentru o poză nouă:

1. Se ia amprenta ei de la pasul 1.
2. Se compară cu fiecare din cele 735. Măsura e cosinusul dintre două amprente. 1 înseamnă că arată în aceeași direcție. Cu cât e mai mic, cu atât pozele seamănă mai puțin.
3. Se păstrează doar cea mai mare valoare. Asta e asemănarea cu cel mai apropiat vecin, adică cu poza de antrenare cea mai asemănătoare.

Iese un singur număr, scorul de asemănare. Mic înseamnă „poză ciudată”. Verificarea asta e slabă pe pantofi de alte mărci, pentru că un pantof alb de la Reebok seamănă cu unul alb de la Nike. În testul nostru pe poze reale, ea singură ar fi respins doar 0,02 până la 0,03 din alte mărci.

### Pasul 3. Verificarea 2: a fost mărită la fel?

Problema: verificarea 1 nu ajunge. Ne trebuie un semn care nu ține de cum arată pantoful. Semnul e urma de mărire. Detectorul o caută așa:

1. Poza color devine gri, cu valori între 0 și 255.
2. Pentru fiecare pixel se calculează a doua diferență pe orizontală: `stânga - 2 × mijloc + dreapta`. Pe o zonă netedă sau pe o pantă uniformă iese zero. Unde valoarea se frânge iese un număr mare. Se face la fel pe verticală.
3. Se face media pe fiecare coloană. Rămân 254 de numere, câte unul pentru fiecare poziție pe orizontală.
4. Cele 254 de numere sunt împăturite la 16. Pozițiile 1, 17, 33 și tot așa ajung în aceeași căsuță. Pozițiile 2, 18, 34 ajung în următoarea. Rămân 16 medii.
5. Cele 16 medii sunt aduse la aceeași scară (medie 0, abatere 1). Așa contează forma profilului și nu cât de contrastată e poza.
6. Pașii 2 până la 5 se fac pentru patru mărimi: a doua diferență fără semn, pătratul ei, prima diferență fără semn și a doua diferență împărțită la media rândului. Cu două direcții și 16 căsuțe ies 4 × 2 × 16 = 128 de numere.

La o poză fără urmă, cele 16 căsuțe ies cam la fel. La o poză mărită de la 240 la 256, unele căsuțe ies sistematic mai netede decât altele. Tiparul e același la toate pozele taskului.

Cele 128 de numere intră într-o regresie logistică. E un clasificator simplu: înmulțește fiecare număr cu o greutate, adună tot și mai adună o constantă. Iese scorul de urmă. Mare înseamnă „are urma pozelor de antrenare”. La pozele taskului are mediana 4,87. La aceleași poze deplasate cu 8 pixeli are mediana -12,37. Greutățile au fost învățate din două grămezi:

- cu urmă: cele 735 de poze de antrenare;
- fără urmă: 760 de decupaje din 19 imagini naturale care nu sunt pantofi. Fiecare a fost adus la 256 px pe alt drum, uneori cu o salvare JPEG și cu o mărime intermediară între 150 și 640 px.

Detectorul le desparte aproape perfect (AUROC 0,989). Cu 98% din pozele cu urmă păstrate, respinge 0,882 din cele fără. De reținut: grămada fără urmă nu conține pantofi și nici miniaturi de pe internet. De aceea numerele autorilor și ale noastre diferă mai jos.

### Pasul 4. Cele două scoruri devin valori p

Problema: scorul de asemănare e un cosinus, deci cel mult 1. Scorul de urmă e o sumă fără limite fixe. Nu le poți aduna așa. Soluția e valoarea p. Ea răspunde la întrebarea: ce parte din pozele cunoscute au un scor cel mult atât de mare ca al pozei ăsteia?

1. Fișierul conține scorurile out-of-fold ale celor 735 de poze cunoscute, sortate. Sunt două liste, câte una pentru fiecare verificare.
2. Pentru poza nouă se numără câte scoruri din listă sunt mai mici sau egale cu al ei.
3. Valoarea p e `(numărul + 1) / (735 + 2)`.

Iese un număr între 0 și 1 pentru fiecare verificare: `p_sem` pentru asemănare și `p_trace` pentru urmă. 0,5 înseamnă „o poză cunoscută obișnuită”. 0,01 înseamnă „doar 1% din pozele cunoscute stau atât de prost”. Listele sunt out-of-fold cu un motiv. O poză de antrenare comparată cu banca din care face parte își găsește propria amprentă și iese cu asemănare perfectă. Pragul ar fi atunci prea optimist pentru poze noi.

Un exemplu din `sherlock/reports/oof_predictions_v1.csv`. Poza `train_00001.jpg` are asemănarea 0,45552, deci `p_sem` e 0,08955. Are scorul de urmă 2,5583, deci `p_trace` e 0,14925. Suma de la pasul următor iese -6,2171. E peste -9,043, deci poza e acceptată.

### Pasul 5. O singură decizie din două verificări

```
log(p_sem) + 2 * log(p_trace) >= -9.043   ->  răspunsul e marca ghicită
altfel                                    ->  răspunsul e unknown
```

- Logaritmul unui număr sub 1 e negativ. Cu cât valoarea p e mai mică, cu atât logaritmul e mai jos. Deci fiecare verificare scade din sumă pe măsură ce poza arată mai neobișnuit.
- Urma e numărată de două ori, pentru că e verificarea mai puternică.
- Pragul -9,043 a fost ales ca 98% din pozele cunoscute să treacă, out-of-fold.

Autorii au încercat mai multe greutăți și praguri (`sherlock/reports/keep_curve.txt`). Respingerea de acolo e simulată pe grămada fără urmă, presupunând că asemănarea nu ajută deloc. Cu 98% din cunoscute păstrate, a fost 0,608 la greutatea 1; 0,798 la 1,5; 0,861 la 2; 0,867 la 3. Acuratețea la cunoscute a rămas între 0,8912 și 0,8925. Deci până la 2 respingerea crește mult, iar peste 2 aproape deloc. Cu 97% păstrate și greutatea 2, respingerea era 0,913 și acuratețea 0,8816. Autorii au rămas la 98%. Motivul nu e scris, dar varianta asta are acuratețea mai mare la cunoscute.

Două socoteli făcute direct din formulă arată cât cântărește fiecare verificare. O poză cu asemănare perfectă e totuși respinsă dacă `p_trace` e sub circa 0,011. O poză cu urmă perfectă nu poate fi respinsă de asemănare: cea mai mică valoare p posibilă e 1/737, iar logaritmul ei e -6,6, peste prag. Deci verificarea 2 hotărăște aproape tot. Verificarea 1 doar înclină balanța la pozele cu urmă slabă.

### Pasul 6. Ce se întâmplă dacă poza nu vine la 256×256

Urma există doar în pixelii originali, la 256×256. Modelul se uită la mărimea tensorului primit. Dacă e exact 256×256, face ambele verificări. La orice altă mărime sare peste verificarea 2 și regula devine `log(p_sem) >= -3.789`. Poza e adusă atunci la 256 px în interior, doar pentru rețea. Pragul -3,789 păstrează tot 98% din cunoscute. Respingerea rămâne însă doar pe seama asemănării, care e slabă. Autorii estimează cam o treime din alte mărci respinse, sau mai puțin. Poarta a doua ar pica.

Mai e un caz. Platforma poate să dea 256×256, dar după ce a trecut poza prin altă mărime. Am încercat cu pozele taskului micșorate la 224 și aduse înapoi la 256. Modelul a respins 69,25% din ele. Atunci ar pica prima poartă.

### Pasul 7. Cum e scris răspunsul și cum încape în 6,95 MB

Decizia e aceeași în toate cele patru fișiere. În `sherlock_probs4.pt`, la o poză respinsă coloana `unknown` primește 0,9. La una acceptată primește 0,1, iar restul de 0,9 se împarte între mărci. În `sherlock_logits3.pt`, la o poză respinsă cele trei numere sunt egale. La una acceptată marca ghicită are un număr mare și celelalte au zero. Ca totul să intre sub limita de 10 MB:

- Cuantizare int8. Greutățile straturilor de convoluție ale rețelei sunt de obicei numere cu virgulă pe 4 octeți. Aici fiecare e rotunjită la un întreg între -127 și 127, pe 1 octet. Fiecare canal de ieșire are un factor de scară cu care întregii sunt înmulțiți înapoi la rulare. Greutățile ocupă de patru ori mai puțin.
- Costul: amprentele se mișcă puțin față de rețeaua originală (asemănare medie 0,953). Banca de amprente a fost calculată cu rețeaua deja cuantizată, deci se compară lucruri de același fel.
- Banca de 735 de amprente e ținută pe 2 octeți pe număr. 735 × 1280 × 2 octeți înseamnă circa 1,9 MB.
- TorchScript. E un fel de a salva un model PyTorch împreună cu codul lui într-un singur fișier. Platforma îl rulează fără sursele noastre Python. Tot ce e descris la pașii 0 până la 6 e înăuntru.

## Cât de bine merge

Numerele autorilor:

| Ce | Rezultat | Prag |
|---|---|---|
| ghicirea mărcii, out-of-fold, fără respingere | 0,9061 | |
| poze cunoscute acceptate, out-of-fold | 0,980 | |
| acuratețe la cunoscute după respingere, out-of-fold | 0,8912 | 0,85 |
| model de pe felia 0, împachetat ca fișierul final, pe pozele lui ținute deoparte | 0,905; 0,7% respinse | 0,85 |
| la fel, felia 2 | 0,871; 1,4% respinse | 0,85 |
| respingere simulată a altor mărci | 0,86 până la 0,92 | 0,70 |
| imagini naturale care nu sunt pantofi, respinse | 96% până la 98% | |

Rândul cu 0,86 până la 0,92 nu e măsurat pe pantofi. E o simulare în care scorurile de urmă sunt luate de la grămada fără urmă. 0,86 presupune că asemănarea nu ajută deloc. 0,92 presupune că ajută cât ajuta la marca scoasă.

Numerele noastre sunt pe 1750 de poze reale cu Puma, Reebok, Vans și Asics din setul `ipogorelov/sneakers`. Sunt mai noi decât ale autorilor și sunt pe pantofi adevărați. Unde diferă, le luăm pe ale noastre.

| Ce s-a măsurat | Rezultatul nostru | Ce spuneau autorii |
|---|---|---|
| alte mărci, aduse direct la 256 px | respinse 0,57 până la 0,66 | 0,86 până la 0,92 |
| alte mărci, trecute prin aceeași mărire de la 240 la 256 | respinse 0,02 până la 0,04 | estimare de circa 0,14 |
| cele 48 de poze cu originalul de cel puțin 400 px | respinse 0,71 până la 0,79 | |
| mărci cunoscute dintr-o sursă nouă, trecute prin 240 la 256 | acceptate 0,984; acuratețe 0,851 | |
| pozele taskului așa cum sunt | 99,3% acceptate | 0,980 out-of-fold |

Primul rând e cel care contează: 0,57 până la 0,66 e sub pragul de 0,70. Pe mărci, la 1200 de poze și cu o singură metodă de redimensionare: puma 0,603; reebok 0,640; vans 0,667; asics 0,617. Niciuna nu trece de 0,70.

La rândul al patrulea, pe mărci a ieșit: nike 0,88; adidas 0,727; converse 0,947. Fără salvarea JPEG de la mijlocul drumului, acuratețea a fost 0,844, adică sub prag.

Rezultatul depinde de mărimea de la care pornește poza. Am împărțit aceleași 1200 de poze după latura scurtă a originalului. Sub 300 px, grupele au dat între 0,565 și 0,667. Cele 20 de poze între 300 și 400 px au dat 0,800. Cele 48 de cel puțin 400 px au dat 0,729.

Am măsurat și ce s-ar întâmpla cu un prag care păstrează mai puține cunoscute. Niciuna din variante nu e construită ca fișier. Fișierul livrat e primul rând.

| Cunoscute păstrate | Acuratețe la cunoscute | Alte mărci respinse | Peste ambele praguri în aceste măsurători |
|---|---|---|---|
| 0,98 (livrat) | 0,891 | 0,632 până la 0,659 | nu |
| 0,97 | 0,882 | 0,733 până la 0,758 | da |
| 0,965 | 0,880 | 0,749 până la 0,775 | da |
| 0,96 | 0,875 | 0,762 până la 0,790 | da |
| 0,95 | 0,865 | 0,825 până la 0,848 | da, cu marjă de 0,015 la prima poartă |

Cele două coloane vin din locuri diferite. Acuratețea e din tabelul out-of-fold al autorilor, pe pozele taskului. Respingerea e pe pozele noastre aduse direct la 256 px.

Fișierul are 6,95 MB și îi trebuie 17 până la 40 de milisecunde pe poză. Dă aceleași etichete pentru toate formatele de intrare încercate. Modelul nostru vechi respinge 0,29 până la 0,38 din aceleași patru mărci, deci Sherlock e cel mai bun pe care îl avem.

## Cum a fost verificată

**Ce au verificat autorii.** Aceleași poze deplasate cu 8 px sunt respinse în proporție de 99%, deci detectorul se uită la urmă și nu la conținut. Modelele de pe feliile 0 și 2, împachetate ca fișierul final, au fost rulate pe pozele lor ținute deoparte. Formatul de intrare nu schimbă răspunsul (`sherlock/src/verify.py` și `sherlock/src/eval_like_platform.py`). Cuantizarea și împachetarea au verificări separate în `sherlock/src/tests/`.

**Ce am refăcut din numerele lor.** Comenzile de reproducere din descrierea autorilor au fost rulate. Dau numerele lor: 0,9061 la ghicire pe 5 felii, 0,8912 după respingere și AUROC 0,9890 pentru detectorul de urmă.

**Testul nostru independent.** Din pachetul testat am folosit doar fișierul `.pt`. Pozele și codul de test sunt ale noastre. Scripturile sunt în `sherlock/verificare/`. Primul e `soles_test2.py`, cu rezultatul în `log2.txt` și `result2.json`:

1. Rulează modelul pe cele 735 de poze ale taskului, neschimbate: 99,32% acceptate; 99,05% cu eticheta corectă; 22,5 ms pe poză.
2. Le deplasează. Respinse: 25,85% la 3 pixeli; 99,05% la 8 pixeli; 0,54% la 16 pixeli.
3. Le resalvează ca JPEG. Doar 0,68% sunt respinse, deci o resalvare nu strică urma.
4. Le micșorează la 224 și le aduce înapoi la 256. 69,25% sunt respinse.
5. Ia poze din setul `ipogorelov/sneakers`: 550 cu alte mărci și 450 cu mărcile cunoscute. Două poze aproape identice cu cele ale taskului au fost scoase.
6. Caută ce metodă de mărire de la 240 la 256 lasă aceeași urmă ca pozele taskului. Nouă din zece metode o lasă. Cea mai apropiată a fost interpolarea cubică din OpenCV.
7. Scenariul A: alte mărci aduse direct la 256 px, cu cinci metode. Respinse 0,567 până la 0,598.
8. Scenariul B: totul trecut prin 240 și apoi mărit la 256. Alte mărci respinse 0,024 până la 0,035.
9. Scenariul C: alte mărci aduse întâi la o mărime fixă și apoi la 256. Patru mărimi de pornire au dat respingere mare: 224 px (0,931); 241 px (0,916); 239 px (0,876); 288 px (0,716). Celelalte 14, de la 128 la 512 px, au dat 0,518 până la 0,698.
10. O poză pe rând, în cinci formate de intrare, pe 184 de poze. Același răspuns ca la rularea în lot, de fiecare dată.

La pasul 2, autorii raportează 14% la 3 pixeli, iar noi 25,85%. Deplasarea e făcută diferit în cele două teste. La 8 pixeli numerele coincid.

Al doilea e `soles_test3.py`, cu rezultatul în `log3.txt`:

1. Ia alte 1200 de poze cu alte mărci, câte 300 pe marcă.
2. Le aduce direct la 256 px cu trei metode. Respinse între 0,632 și 0,659.
3. Împarte rezultatul după mărimea originalului. La cele 48 de poze de cel puțin 400 px, cele trei metode au dat 0,729; 0,792; 0,708.
4. Refolosește scorurile ca să încerce alte praguri. De aici vine tabelul cu praguri mai stricte.

Cele 1750 de poze sunt 550 din primul script plus 1200 din al doilea. Sunt trase separat și logurile nu spun dacă au poze comune. `soles_test.py` e prima formă a aceluiași test. Rezultatul lui nu e păstrat în folder.

## Ce nu e sigur

**Poarta de respingere e în pericol.** Pe pantofi reali de alte mărci am măsurat 0,57 până la 0,66, iar pragul e 0,70. Cu fișierul livrat, pe pozele noastre, poarta pică. Pe platformă depinde de unde au luat organizatorii pozele cu alte mărci. Dacă sunt poze mari aduse direct la 256 px, poarta e la limită, judecând după doar 48 de poze. Dacă sunt miniaturi ca ale noastre, pică. Dacă au trecut prin aceeași mărire de la 240 la 256, pică sigur. Nu știm care caz e cel real.

Celelalte lucruri nesigure:

- **Verificarea 2 ține de felul în care a fost făcut setul de date**, nu de cum arată pantoful. Autorii spun asta deschis. Într-o folosire reală, unde toate pozele vin pe același drum, nu ar ajuta.
- **Platforma trebuie să dea pixelii originali la 256×256.** Dacă micșorează poza, rămâne doar verificarea 1 și poarta a doua pică. Dacă o trece prin altă mărime și o aduce înapoi, pică prima poartă.
- **Marja la mărcile cunoscute e mică.** 0,891 out-of-fold față de 0,85. Pe felii a ieșit între 0,871 și 0,905. Pe poze dintr-o sursă nouă am măsurat 0,851. Autorii socotesc că la circa 90 de poze cunoscute în test, norocul singur mișcă rezultatul cu vreo 0,03.
- **Numărul de poze de test e dedus.** Cele circa 30 pe marcă vin dintr-o scădere, nu de la organizatori.
- **Formatul răspunsului nu e cunoscut**, de aceea sunt patru fișiere. Nici felul în care platforma deschide fișierul nu e cunoscut.
- **Pragurile mai stricte sunt doar măsurători.** Nu există un fișier construit cu ele. Acuratețea și respingerea din tabel vin din seturi diferite de poze. Dacă pozele cu alte mărci au urma, pragul nu mai ajută: în simularea autorilor (`sherlock/reports/joint_sim_loco.txt`) respingerea rămâne cel mult 0,341.
- **Grămada fără urmă a autorilor e blândă.** Pe poze de antrenare rotite cu 1 până la 4 grade sau mărite puțin, detectorul ajunge doar la AUROC 0,85, pentru că o parte din urmă rămâne în ele.
- **Testul nostru are și el limite.** Pozele sunt miniaturi de pe internet. Nu știm cât seamănă cu cele ale organizatorilor.

## Unde e fiecare lucru

**Folderul taskului, primul nivel**

- `README.md`: descrierea scurtă. `EXPLICATIE.md`: documentul ăsta. `DETALII.md`: descrierea în engleză a modelului nostru vechi, prima formă.
- `.gitignore`: exclude `results/ckpt/`, `cache/`, `__pycache__/` și fișierele `.pth`.
- `main.py`, `model.pt`, `match_sim.py`: potrivirea cu cele 735 de poze și calibrarea ei. `model.pt` are 84 MB.
- `shoelib.py`, `validate.py`, `train_all.py`, `train_final.py`, `build_model.py`: rețeaua antrenată de la zero. Definiție, validare, antrenare și împachetare.
- `oe_prepare.py`, `oe_lib.py`, `oe_cv.py`, `oe_sweep.py`, `oe_final.py`, `oe_export.py`: rețeaua preantrenată cu a patra clasă. Date, model, validare, praguri, antrenare finală și împachetare.
- `oe_bank.py`, `oe_featspace.py`, `oe2_report.py`: încercări în jurul ei. Potrivire cu miniaturi gri de 64×64, scoruri pe amprente și raportul unei variante care nu vede la antrenare cele patru mărci de test.
- `oe_verify_export.py`, `oe_check_pt.py`: verifică un fișier împachetat.
- `ko_cv.py`, `ko_scores.py`, `ko_final.py`, `ko_export.py`, `ko_eval_pt.py`: rețeaua preantrenată doar pe cele 735 de poze. Validare, alegerea pragului, antrenare finală, împachetare și măsurare pe poze de pe internet.

**`results/` și `cache/`** (ale modelului nostru vechi; nimic de aici nu se predă)

- `results/ko_eval_pt_r384.txt`, `results/ko_eval_pt.json`: rezultatele modelului vechi pe poze de pe internet. `results/check_pt_r384.txt`: verificarea fișierului împachetat.
- `results/vietnam_model_r384.pt`: modelul vechi împachetat, cel măsurat. `results/vietnam_model_r256.pt`: o împachetare mai veche, la 256 px.
- `results/cand_fp16_nobank.pt`, `results/cand_int8_bank.pt`: două împachetări de probă. După nume și dată țin de varianta cu a patra clasă.
- Restul din `results/`: loguri, scoruri și modele. `ko_*` și `oe_*` sunt ale celor două variante cu rețea preantrenată. `base.*`, `oe.json`, `oe.log`, `r128.*`, `r192.*` și `ckpt/` sunt ale rețelei antrenate de la zero. `README_old_attempt.md.bak` e o copie identică a `DETALII.md`.
- `cache/`: poze pregătite și amprente salvate, în fișiere `.npz`. E exclus din git. Îl scriu `oe_prepare.py` și `ko_cv.py`.

**`sherlock/`**

- `README_original_en.md`: descrierea autorilor, în engleză.
- `submission/`: cele patru fișiere `.pt` și câte un `.json` cu pragurile și ordinea claselor.
- `verificare/`: testul nostru independent. `soles_test.py`, `soles_test2.py`, `soles_test3.py`, `log2.txt`, `log3.txt`, `result2.json`.
- `scripts/`: `download_weights.sh` aduce rețeaua preantrenată. `run_pipeline.sh` reface totul de la zero.
- `src/finetune.py`: reantrenează rețeaua. `src/oof_infer.py`: recalculează scorurile out-of-fold la 256 px. `src/extract_feats.py`: scoate amprente din rețelele comparate la început.
- `src/fit_trace2.py`: construiește grămada fără urmă și calculează cele 128 de numere. `src/fit_trace_final.py`: antrenează regresia logistică a detectorului. `src/trace_torch.py`: același calcul, în forma care intră în fișier.
- `src/make_calib.py`: calculează valorile p și pragurile. Aici se alege cât la sută din cunoscute se păstrează. `src/export_model.py`: împachetează totul în fișierul `.pt`.
- `src/make_oof_csv.py`: scrie tabelul cu scorurile fiecărei poze. `src/verify.py`, `src/eval_like_platform.py`: rulează fișierul cum l-ar rula un evaluator. `src/paths.py`: căile către date și rezultate.
- `src/analysis/`: experimentele (compararea rețelelor, cercetarea fișierelor, variante de detector, simulări, v2). `src/tests/`: verificări pentru cuantizare și împachetare.
- `reports/oof_predictions_v1.csv`: un rând pentru fiecare din cele 735 de poze, cu marca ghicită, cele două scoruri, valorile p și decizia.
- `reports/keep_curve.txt`: acuratețea și respingerea pentru mai multe reguli și praguri. `reports/fit_trace_final.txt`: AUROC-ul detectorului de urmă. `reports/joint_sim_loco.txt`: simularea cu o marcă scoasă, cu alte mărci fără urmă și cu urmă.
- `reports/frozen_probes.txt`, `reports/tta_eval.txt`, `reports/v2_ensemble_eval.txt`: compararea celor șapte rețele și două din încercările lăsate. `reports/holdout_errors.png`, `reports/train_samples.png`: imagini cu exemple.

Descrierea autorilor pomenește și un folder `runs/`, cu modelele antrenate și scorurile intermediare. El nu e în copia din repo. Nu sunt nici rețeaua preantrenată și nici datele. Datele taskului sunt lângă folderul taskului, în `../task_vietnam/`: folderul `train-input/` și fișierul `train-output.csv`.

## Cum refaci totul

**Deschizi fișierul.** `m = torch.jit.load("sherlock/submission/sherlock_labels.pt")`, apoi `m(x)`. `x` e un tensor cu N poze de 256×256. Iese o listă de N etichete.

**Verificările autorilor și numerele lor.** Se rulează din `sherlock/`. `verify.py` și `eval_like_platform.py` au nevoie doar de datele taskului. Celelalte patru comenzi au nevoie și de folderul `runs/`, care nu e în copia din repo.

```bash
export SHERLOCK_DATA=<folderul cu train-input/ și train-output.csv>
python -I src/verify.py submission/sherlock_labels.pt
python -I src/eval_like_platform.py
python -I src/make_oof_csv.py runs "$SHERLOCK_DATA" /tmp/oof.csv
python -I src/make_calib.py --runs runs/mnv3_cv{0..4} --emb emb256.npz --keep 0.98 --w_trace 2.0 --out /tmp/c.npz
python -I src/fit_trace_final.py 0.3 /tmp/t.npz
python -I src/export_model.py --ckpt runs/mnv3_full/model.pt --calib runs/calib_v1.npz --infer_size 256 \
       --outdir /tmp/re --variants "labels:nike,adidas,converse"
```

Un fișier împachetat din nou nu e identic octet cu octet cu cel din `submission/`. Autorii au verificat că dă aceleași etichete pe 147 din 147 de poze.

**Testul nostru independent.** Se rulează din folderul taskului. Îți trebuie o copie locală a setului `ipogorelov/sneakers`, în fișiere `.parquet`. Rezultatele de comparat sunt în `sherlock/verificare/log2.txt` și `log3.txt`.

```bash
python sherlock/verificare/soles_test2.py sherlock/submission/sherlock_labels.pt ../task_vietnam <folderul cu setul ipogorelov/sneakers> out.json
python sherlock/verificare/soles_test3.py sherlock/submission/sherlock_labels.pt <folderul cu setul ipogorelov/sneakers> sherlock/reports/oof_predictions_v1.csv out.npz
```

**Totul de la zero.** Durează cam 1,5 până la 2 ore pe procesor, după autori.

1. Pui datele în `sherlock/data/task_vietnam/`.
2. Rulezi `scripts/download_weights.sh`. Aduce rețeaua MobileNetV3 preantrenată.
3. Rulezi `scripts/run_pipeline.sh`. Întâi face detectorul de urmă (`fit_trace2.py`, `fit_trace_final.py`). Apoi antrenează pe 5 felii și scoate scorurile out-of-fold la 256 px (`finetune.py`, `oof_infer.py`). Apoi calculează valorile p și pragul, cu greutatea 2 și 98% păstrate (`make_calib.py`). La urmă antrenează pe toate pozele, împachetează cele patru variante și le verifică (`finetune.py`, `export_model.py`, `verify.py`).

Mediul folosit de autori: Python 3.13; torch 2.11.0; torchvision 0.26.0; numpy 2.4.6; Pillow 12.3.0; timm 1.0.31; scikit-learn 1.9.1; scikit-image.

**Un prag mai strict.** Nu a fost făcut. Pragul se alege în `make_calib.py`, prin `--keep`. Un fișier cu alt prag ar însemna aceeași comandă cu altă valoare, de exemplu `--keep 0.97`, urmată de `export_model.py`. Amândouă au nevoie de folderul `runs/`.

**Modelul nostru vechi.** Comenzile sunt în primele rânduri ale fiecărui script. `oe_prepare.py` pregătește pozele în `cache/`, iar calea către setul `ipogorelov/sneakers` e scrisă direct în el. `ko_cv.py` antrenează pe 5 felii și `ko_scores.py` alege pragul. `ko_final.py` antrenează pe toate pozele și `ko_export.py` împachetează. `ko_eval_pt.py` măsoară fișierul pe poze de pe internet și scrie `results/ko_eval_pt.json`.

**Punerea pe GitHub.** Fișierele `.pt` sunt excluse de regulile repo-ului. Cele patru din `sherlock/submission/` se adaugă cu `git add -f`.
