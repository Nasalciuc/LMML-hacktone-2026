# Cuba: If you know you know. Explicația completă

Documentul explică taskul Cuba din LMML 2026 pentru cineva care nu a lucrat la el. Cifrele vin din `README.md` și `DETALII.md`. Restul vin din `local_check_output.txt`, din `_exp/results.md` și din comentariile din cod. Zecimalele sunt scrise cu virgulă (0,940). În comenzi și în formule rămân cu punct, ca în fișiere.

## Ce se pune pe site

Taskul se predă ca fișier. Fișierul este o arhivă zip care are la nivelul de sus `main.py` și folderul `index/`.

| Ce | Valoare |
|---|---|
| Arhiva de trimis | `cuba_v4.zip` |
| sha256 | `45a8499d35c2b74b33a45ac7239345fc1ce0da79c1fa5fdd0ee7a15c4a0e41e8` |
| Mărime arhivată | 38,7 MB |
| Mărime despachetată | 51,26 MB (limita e 53 MB) |
| Unde stă arhiva | `C:\Users\user\Downloads\LMML_upload\Cuba\`, în afara repo-ului, pentru că regulile lui exclud fișierele zip |
| Rezerva | `cuba_v3.zip`, în același folder |
| Flag | nu se cunoaște niciunul pentru acest task |

Ce trebuie știut înainte de trimitere:

- Versiunea 4 are nevoie de `onnxruntime` pe platformă. Este biblioteca Python care rulează modelul din `index/encoder.onnx`. Dacă platforma spune că nu o poate importa, se trimite `cuba_v3.zip`. Este versiunea 3 și are nevoie doar de numpy.
- Site-ul cere și un link de GitHub către descrierea soluției. Linkul exact nu apare în surse. Descrierea scurtă este `README.md` din acest folder.
- Versiunea 1 a picat pe platformă cu 0,793. Sursele nu spun că versiunea 4 a fost încărcată sau acceptată. `DETALII.md` scrie că estimarea pentru platformă nu e încă verificată acolo.

## Ce cere taskul

Există o bibliotecă de 1552 de documente despre Cuba: articole Wikipedia, capitole de cărți, rapoarte. Primim o întrebare scrisă ca un bilet care descrie un capitol, de felul „partea în care...”. Trebuie să întoarcem cele 3 documente cele mai probabile.

Codul trebuie să aibă în `main.py` clasa `Retriever(index_dir)` cu metoda `search(question, k)`. Metoda primește textul întrebării și întoarce o listă de identificatori de documente. Folderul `index/` ține datele pregătite dinainte din bibliotecă.

Soluția trece doar dacă respectă toate cele patru limite:

| Măsură | Limită | Ce înseamnă |
|---|---|---|
| hit@3 pe întrebările ascunse | cel puțin 0,80 | documentul corect e printre cele 3 întoarse la cel puțin 80% din întrebări |
| mărimea `main.py` + `index/` | cel mult 53 MB | aici intră și modelul |
| p95 al unei căutări | cel mult 75 ms | măsurat pe un singur fir de procesor |
| timpul de încărcare | cel mult 10 s | pornirea, înainte de prima întrebare |

Trei termeni care apar peste tot:

- **hit@3** este partea din întrebări la care documentul corect se află printre cele 3 întoarse. 0,940 înseamnă 94 din 100.
- **p95** este timpul sub care se termină 95% din căutări. Câteva căutări lente sunt iertate, dar nu multe.
- **Întrebările ascunse** sunt cele pe care platforma le pune soluției. Noi nu le vedem.

Ce avem la dispoziție local:

- Biblioteca este în `..\task_cuba\data\corpus\`, câte un fișier text pe document. Lista documentelor este în `..\task_cuba\data\corpus_meta.jsonl`, cu identificator, titlu și fel.
- 268 de întrebări publice cu răspunsul corect, în `..\task_cuba\data\sample-input.jsonl` și `..\task_cuba\data\sample-output.jsonl`.
- Scriptul de verificare al organizatorilor, `..\task_cuba\local_check.py`. Măsoară cele patru limite pe întrebările publice.

Limitele decid ce fel de soluție e posibilă. Un model mare nu încape în 53 MB. Un model lent nu încape în 75 ms pe un singur fir.

## Ce am aflat din date

### Biblioteca

| Fel de document | Număr |
|---|---|
| Articole Wikipedia | 955 |
| Capitole de cărți | 470 |
| Secțiuni de rapoarte | 105 |
| Capitole scanate, cu erori de recunoaștere a literelor (Humboldt) | 14 |
| Tabele | 8 |
| Total | 1552 |

- 28 de documente sunt copii exacte ale altora. Rămân 1524 de documente unice. Copiile se țin o singură dată și se întorc împreună.
- Biblioteca are 3,87 milioane de cuvinte. 1,80 milioane sunt în Wikipedia și 2,07 milioane în rest.
- Pentru căutare, fiecare document e tăiat în pasaje de 200 de cuvinte. Un pasaj nou începe la fiecare 100 de cuvinte, deci pasajele vecine se suprapun pe jumătate. Ies 37944 de pasaje.

### Întrebările publice

Din cele 268 de întrebări publice, 108 cer un articol Wikipedia. Celelalte 160 cer un capitol de carte, o secțiune de raport sau un capitol scanat. Pe acestea le numim mai departe **întrebări non-Wikipedia**. 147 dintre ele cer un capitol de carte.

Întrebările sunt de mai multe feluri:

- întrebări despre un detaliu din text;
- cereri pentru un capitol întreg („capitolul care spune toată povestea...”);
- întrebări cu greșeli de tastare („bigin”, „thier”);
- întrebări cu cuvinte spaniole;
- întrebări cu cuvinte lipite între ele;
- ghicitori care ocolesc cuvintele evidente: „captive Africans” în loc de sclavi, „painted hen” în loc de gallo pinto.

### Unde sunt greșelile

hit@3 pe întrebările publice, împărțit după felul documentului corect:

| Versiune | Toate (268) | Wikipedia (108) | Non-Wikipedia (160) | Capitole de cărți găsite (din 147) |
|---|---|---|---|---|
| 1 | 0,873 | 0,991 | 0,794 | nu apare în surse |
| 2 | 0,918 | 1,000 | 0,863 | 125 |
| 3 | 0,914 | 1,000 | 0,856 | nu apare în surse |
| 4 | 0,940 | 1,000 | 0,900 | 131 |

Ce spun cifrele:

1. De la versiunea 2 încoace, toate întrebările despre Wikipedia sunt rezolvate. La versiunile 2 și 4, toate ratările sunt la capitole de cărți.
2. Versiunea 1 a luat 0,794 pe întrebările non-Wikipedia și 0,793 pe platformă. De aici presupunerea că întrebările ascunse sunt mai ales despre cărți. De aceea coloana non-Wikipedia e luată drept estimare pentru platformă.
3. Mai multe cărți povestesc aceleași evenimente. La cele 22 de întrebări ratate de versiunea 2 s-au întors 66 de documente. 48 erau capitole din altă carte, 13 capitole din cartea corectă și 5 articole Wikipedia. Subiectul e găsit, dar câștigă cartea greșită.
4. Documentul corect e rar departe. Versiunea 2 îl are în primele 10 la 95% din întrebările non-Wikipedia.
5. La întrebările ratate, potrivirea pe cuvinte pune documentul corect cam pe locul 20. Potrivirea pe sens îl are des în primele 8. Prima numără cuvintele comune, a doua compară înțelesul. Sunt întrebări care spun același lucru cu alte cuvinte. Doar o potrivire pe sens mai bună le poate muta.

`_exp/results.md` numără 21 de ratări la versiunea 2, nu 22. Scriptul de experimente tratează puțin altfel documentele duplicate. Aici e folosită cifra din `DETALII.md`, care e mai nou.

## Ce am încercat, în ordine

Toate versiunile au aceeași schemă: două feluri de căutare, apoi scorurile se adună. De la o versiune la alta s-a schimbat partea care înțelege sensul.

```
v1  BM25 + potion (vectori statici)            -> non-Wikipedia 0,794   picată pe platformă cu 0,793
v2  BM25 + arctic-embed-xs (6 straturi)        -> non-Wikipedia 0,863
v3  ca v2, dar modelul rulează doar cu numpy   -> non-Wikipedia 0,856   rezervă
v4  BM25 + gte-small (12 straturi) + PCA 256   -> non-Wikipedia 0,900   pregătită pentru site
```

Cuvinte folosite mai jos:

- **BM25** este potrivirea pe cuvinte. Numără cuvintele comune dintre întrebare și text. Un cuvânt rar în bibliotecă contează mai mult decât unul des.
- **Vector de sens** este un șir de numere pe care un model de limbă îl scoate dintr-un text. Două texte cu înțeles apropiat primesc șiruri apropiate, chiar fără cuvinte comune. Potrivirea pe sens compară vectorul întrebării cu vectorii pasajelor. Măsura apropierii se numește **cosinus**, iar 1 înseamnă aceeași direcție.
- **Straturi** sunt treptele prin care modelul trece textul. Mai multe straturi înseamnă un model mai priceput, dar mai mare și mai lent.
- **ONNX** este un format de fișier pentru un model antrenat. Biblioteca `onnxruntime` îl rulează pe procesor.
- **int8** înseamnă că numerele sunt păstrate ca întregi între -128 și 127. Un număr ocupă 8 biți în loc de 32, cu o mică pierdere de precizie. Operația se numește cuantizare.
- **Semnal** este un scor dat fiecărui document de o singură metodă. **Ponderea** spune cât contează semnalul în scorul final.
- **CV** (validare încrucișată): ponderile se aleg pe patru cincimi din întrebări și se măsoară pe cincimea rămasă. Se repetă cu alte împărțiri. Arată cât din scor ține de potrivirea ponderilor pe întrebările publice.
- **Întrebări grele** sunt cele 43 de întrebări publice la care BM25 singur nu pune documentul corect în primele 3.

### Versiunea 1: BM25 plus vectori statici

- Ce s-a încercat: BM25 adunat cu vectori de sens de la modelul potion-retrieval-32M. Este un model static: fiecare cuvânt are vectorul lui fix, oricare ar fi propoziția din jur.
- Ce a ieșit: public 0,873, Wikipedia 0,991, non-Wikipedia 0,794. Întrebări grele 0,293. Mărime 34,8 MB.
- De ce a fost abandonată: pe platformă a luat 0,793, sub pragul de 0,80. La întrebările grele găsea documentul corect în mai puțin de o treime din cazuri.
- Arhiva ei este `submission.zip` din acest folder.

### Versiunea 2: un model cu 6 straturi

- Ce s-a încercat: vectorii statici au fost înlocuiți cu modelul arctic-embed-xs. Are 6 straturi, e cuantizat int8 și rulează cu `onnxruntime`. Tot aici, modelul a început să primească întrebarea curățată de greșeli, nu pe cea brută.
- Ce a ieșit: public 0,918, non-Wikipedia 0,863, CV pe non-Wikipedia 0,865. Întrebări grele 0,535. Mărime 45,84 MB, p95 16,7 ms.
- Ce a rămas nerezolvat: 22 de întrebări ratate, toate despre capitole de cărți. De aici a pornit lucrul la versiunea 4.

### Alegerile de model din jurul versiunilor 2 și 3

| Ce s-a încercat | Ce a ieșit | Hotărâre |
|---|---|---|
| Modelul MiniLM-L6, rulat cu codul numpy rămas din versiunea 1 | 721 ms pe întrebare, pentru că întrebarea era umplută până la 250 de tokeni | abandonat |
| Întrebarea brută sau cea curățată la intrarea modelului | CV: MiniLM 0,888 -> 0,914, arctic 0,899 -> 0,903 | se dă întrebarea curățată |
| int8 cu o scară pe tot tabloul de numere sau cu o scară pe canal | cosinus față de modelul necuantizat: MiniLM 0,968 -> 0,995, arctic 0,989 -> 0,998 | o scară pe canal |
| MiniLM și arctic împreună | CV 0,925 | două modele nu încap |
| arctic fără prefixul lui de întrebare | public 0,862, întrebări grele 0,326 | prefixul e obligatoriu la arctic |
| În varianta numpy, calculul prefixului făcut o dată și refolosit | cu 15% mai rapid, dar întrebările grele scad de la 0,512 la 0,465 | respins |

- Un token este o bucată de cuvânt. Modelul împarte textul în tokeni înainte să îl citească.
- La cuantizare, fiecare număr devine un întreg înmulțit cu o scară. Cu o singură scară pe tot tabloul, rândurile cu numere mici pierd precizie. Cu o scară pe canal, fiecare rând are scara lui.
- Prefixul este o propoziție fixă pe care arctic o așteaptă în fața întrebării: `Represent this sentence for searching relevant passages: `. Modelul gte-small din versiunea 4 primește întrebarea fără prefix.

### Versiunea 3: același model, fără onnxruntime

- Problema: nu e sigur că platforma poate importa `onnxruntime`.
- Ce s-a încercat: același arctic-embed-xs, dar calculele modelului sunt scrise direct în numpy.
- Ce a ieșit: public 0,914, non-Wikipedia 0,856. Mărime 45,74 MB, încărcare 0,90 s, p95 între 40 și 51 ms. Când calculatorul e ocupat la maximum, p95 urcă la 95-116 ms, peste limita de 75 ms.
- Hotărâre: rămâne doar rezervă. Fișierele ei sunt în `_exp/v3/`, iar arhiva este `cuba_v3.zip`.

### Semnale în plus pentru a deosebi cărțile

Problema: subiectul e găsit, dar câștigă capitolul din altă carte. Prima idee a fost adăugarea unui semnal nou la versiunea 2. Pentru fiecare semnal s-a căutat ponderea care crește hit@3 pe non-Wikipedia. Punctul de plecare era CV pe non-Wikipedia 0,862.

| Semnal încercat | CV pe non-Wikipedia | Pondere aleasă |
|---|---|---|
| Scor comun pe același pasaj: BM25 și sens adunate sau înmulțite | 0,860-0,862 | 0 |
| Media primelor 3 pasaje; netezire cu pasajele vecine | 0,858-0,862 | 0 |
| Scăderea scorului mediu al cărții din scorul fiecărui capitol | 0,862 | 0 |
| BM25 pe pasaje mai mici (100 de cuvinte cu pas 50, 60 cu pas 30) | 0,862 | 0 |
| Acoperirea cuvintelor rare, pe pasaje de 100 de cuvinte | 0,852 | 0, mai slab |
| Cuvinte exacte, neaduse la rădăcină (ortografie de epocă, nume) | 0,862 | 0 |
| Perechi de cuvinte vecine | 0,856 | 0, mai slab |
| Numere și ani, potrivire exactă | 0,862 | 0 |

Niciun semnal nu a primit pondere peste 0. Întrebările ratate nu au în comun cu capitolul corect nici cuvinte rare, nici nume, nici ani. Potrivirea exactă nu are ce găsi.

### Modele cu 12 straturi și pasaje mai mici

A doua idee: un semnal de sens mai bun. Modelele au fost măsurate întâi în forma exactă, necuantizată, adunate cu BM25. „200/100” înseamnă pasaje de 200 de cuvinte cu pas de 100.

| Semnal de sens | CV pe non-Wikipedia | Non-Wikipedia la 5 seturi fixe de ponderi | Întrebări grele (43) |
|---|---|---|---|
| arctic-xs, pasaje 200/100 (versiunea 2) | 0,855 | 0,838-0,863 | 0,42-0,54 |
| arctic-xs, pasaje 100/50 | 0,871 | 0,869-0,887 | 0,51-0,61 |
| arctic-s (12 straturi) | 0,863 | 0,863-0,881 | 0,61-0,63 |
| bge-small | 0,894 | 0,881-0,906 | 0,63-0,67 |
| gte-small | 0,879 | 0,894-0,906 | 0,61-0,67 |
| e5-small-v2 | 0,884 | 0,881-0,900 | 0,58-0,70 |

Trei modele diferite cu 12 straturi (gte-small, bge-small, e5-small) câștigă fiecare 2-4 puncte pe non-Wikipedia. Pasajele mai mici ajută modelul cu 6 straturi cu 1,6 puncte.

### Aceleași modele, în forma care se livrează

Problema: modelul exact nu poate fi livrat. Pe platformă rulează varianta int8, iar vectorii pasajelor sunt scurtați și ei. Scorurile au fost măsurate din nou în această formă.

| Variantă | Public | Wikipedia | Non-Wikipedia | CV pe non-Wikipedia | Întrebări grele |
|---|---|---|---|---|---|
| versiunea 2, arctic-xs | 0,922 | 1,000 | 0,869 | 0,862 ± 0,040 | 0,535 |
| bge-small, PCA 256 | 0,933 | 1,000 | 0,887 | 0,879 ± 0,034 | 0,651 |
| bge-small, fără PCA (prea mare) | 0,937 | 1,000 | 0,894 | 0,879 ± 0,028 | 0,651 |
| gte-small, PCA 256 | 0,940 | 1,000 | 0,900 | 0,888 ± 0,024 | 0,628 |
| gte-small, fără PCA (prea mare) | 0,929 | 1,000 | 0,881 | 0,893 ± 0,036 | 0,581 |

- PCA 256 înseamnă că fiecare vector de pasaj e scurtat de la 384 de numere la 256. Cum se face e explicat la soluția finală.
- Scriptul acesta de experimente numără puțin altfel documentele duplicate. De aceea versiunea 2 are aici 0,869, nu 0,863.
- gte-small a fost ales din două motive. Păstrează câștigul după cuantizare: cosinus mediu 0,9989 față de modelul exact, cel mai mic 0,9969. La bge-small media e 0,9973. Și merge cu ponderile versiunii 2 neschimbate.
- La bge-small, ponderi potrivite pe toate întrebările urcau non-Wikipedia la 0,900, dar pierdeau o întrebare Wikipedia (0,991).
- Primul export al lui gte-small a eșuat. Modelul e salvat cu numere pe 16 biți, iar unealta de cuantizare le refuză. Rezolvarea: modelul e trecut pe 32 de biți înainte de export.

### Pasaje mai mici doar pentru cărți

Ideea: dacă pasajele mai mici au ajutat modelul cu 6 straturi, poate ajută și aici. Cărțile au fost tăiate mai mărunt, iar Wikipedia a rămas la 200/100. Mai multe pasaje înseamnă mai mulți vectori, așa că fiecare vector a fost scurtat mai mult.

| Variantă | Pasaje | Vectori (MB) | Non-Wikipedia | CV pe non-Wikipedia | Wikipedia |
|---|---|---|---|---|---|
| bge, cărți 100/50, PCA 176 | 58638 | 10,8 | 0,869 | 0,863 ± 0,046 | 1,000 |
| bge, cărți 100/50, fără PCA (nu încape) | 58638 | 23,3 | 0,875 | 0,887 ± 0,040 | 1,000 |
| bge, cărți 150/75, PCA 240 | 44846 | 11,3 | 0,887 | 0,877 ± 0,051 | 1,000 |
| gte, cărți 100/50, PCA 176 | 58638 | 10,8 | 0,875 | 0,857 ± 0,034 | 1,000 |
| gte, cărți 150/75, PCA 240 | 44846 | 11,3 | 0,906 | 0,900 ± 0,032 | 0,991 |
| gte, cărți 150/75, fără PCA (nu încape) | 44846 | 18,0 | 0,894 | 0,870 ± 0,037 | 0,991 |

Nu iese un câștig constant față de 200/100 simplu (0,900 și CV 0,888). Singurul rând mai bun pierde o întrebare Wikipedia. Varianta lui fără PCA e mai slabă (0,870), deci câștigul a fost socotit întâmplare. Nu a fost livrat.

### Versiunea 4: gte-small cu 12 straturi

- Ce s-a încercat: gte-small cuantizat int8, vectori de pasaj scurtați la 256 de numere, pasaje 200/100. Ponderile sunt cele din versiunea 2.
- Ce a ieșit: public 0,940, non-Wikipedia 0,900, CV pe non-Wikipedia 0,893. Întrebări grele 0,628. Mărime 51,26 MB.
- Hotărâre: este versiunea pregătită pentru site.

### Ce nu s-a făcut

- **Reordonarea primelor 10 cu un cross-encoder.** Un cross-encoder este un model care citește întrebarea și pasajul împreună. E mai precis, dar are nevoie de textul pasajelor la căutare. Textul are 3,87 milioane de cuvinte, adică vreo 8 MB comprimat. Modelul ar mai cere 4,5 MB. Liber e 1,2 MB sub ținta de 52,5 MB folosită în experimente. Nu încape.
- **Îmbogățirea pasajelor cu un model „sparse”** (opensearch doc-v2-distill). Un astfel de model adaugă dinainte, la fiecare pasaj, cuvinte înrudite. Potrivirea pe cuvinte ar prinde atunci și sinonimele, fără model la căutare. Două rulări au murit: una la o buclă lentă peste vocabular, alta la o repornire a calculatorului. Scriptul reparat este `_exp/sparse4.py` și nu a fost rulat până la capăt. Indexul lui ar cere câțiva MB care nu sunt liberi.

## Soluția finală, pas cu pas

Ce se întâmplă la fiecare întrebare, pe un singur fir:

```
întrebare -> curățare -> BM25 pe pasaje ................. p --+
                      -> BM25 pe documente întregi ...... d --+-> aceeași scară -> sumă cu ponderi -> primele 3
                      -> gte-small -> vector -> pasaje .. e --+
                                             -> document  ed -+
```

### Partea 1. Ce se pregătește dinainte

Se face o singură dată, pe calculatorul nostru. Scripturile sunt arătate la „Cum refaci totul”.

1. **Citirea bibliotecii.** Se citește fiecare document. Documentele cu text identic se grupează. Rămân 1524 de grupuri, salvate în `index/docs.json`.
2. **Titlul scurt.** La un articol Wikipedia se păstrează titlul. La celelalte se păstrează doar ce urmează după primul punct din titlu. Așa dispare numerotarea de tip „CHAPTER IV.”. Un titlu fără punct nu se păstrează deloc.
3. **Tăierea în pasaje.** 200 de cuvinte, cu pas de 100. Un document de cel mult 200 de cuvinte rămâne un singur pasaj. Ies 37944 de pasaje.
4. **Listele pentru BM25.** Pentru fiecare pasaj se ia titlul scurt plus textul. Textul trece la litere mici, fără accente și fără semne. Cuvintele de legătură se scot. Restul se aduc la rădăcină: „slaves” și „slave” ajung aceeași formă. Pentru fiecare rădăcină se ține lista pasajelor în care apare și de câte ori. La fel se face și pe documentele întregi.
5. **Lista pentru corectarea greșelilor.** Fiecare cuvânt din bibliotecă merge în `index/words.txt`, cu numărul lui de apariții.
6. **Vectorii de sens ai pasajelor.** Modelul gte-small citește titlul scurt plus pasajul, cel mult 256 de tokeni. Dă câte un vector pentru fiecare token. Media lor este vectorul de sens al pasajului, cu 384 de numere.
7. **Scurtarea la 256 de numere (PCA).** PCA caută direcțiile pe care vectorii pasajelor diferă cel mai mult între ei. Se păstrează primele 256 de direcții și fiecare vector se rescrie doar cu ele. Direcțiile se calculează din fiecare al treilea pasaj. Tabelul de conversie se salvează, ca să fie aplicat și întrebării. Motivul scurtării e mărimea. Modelul int8 are 34,0 MB, cu 11 MB peste cel din versiunea 2. Cu vectori de 384 de numere totul ar avea 55,7 MB. Cu 256 are 51,26 MB.
8. **Vectorii pasajelor ca int8.** Fiecare vector primește o scară proprie, aleasă astfel încât cel mai mare număr al lui să devină 127. Numerele se rotunjesc la întregi. Întregii, scările și tabelul de conversie merg în `index/dense.npz`.
9. **Modelul pentru întrebări.** gte-small e exportat în format ONNX și cuantizat int8, cu o scară pe canal. Devine `index/encoder.onnx`. Lângă el stă `index/qtokenizer.json`, care împarte textul în tokeni.
10. **Setările.** `index/config.json` ține ponderile, mărimea pasajelor și setările modelului.

### Partea 2. Încărcarea

Se face o dată, la pornire. În verificarea oficială a durat 0,86 s și 0,88 s.

1. Se citesc setările, lista de documente și listele BM25. Se calculează dinainte cât de rar e fiecare cuvânt și cât de lung e fiecare pasaj.
2. Vectorii pasajelor se readuc la numere cu virgulă: întregii se înmulțesc cu scara lor.
3. Pentru fiecare document se face un vector al documentului: suma vectorilor pasajelor lui, adusă la lungimea 1. Nu e salvat în index, pentru că se reface ieftin.
4. Se pornește `onnxruntime` cu un singur fir. Se încarcă modelul și lista pentru corectarea greșelilor.

### Partea 3. O căutare

1. **Normalizare.** Întrebarea trece la litere mici, fără accente. Rămân doar litere și cifre. „azúcar” devine „azucar”.
2. **Cuvinte spaniole.** Dacă un cuvânt e în lista spaniolă din `main.py`, e înlocuit cu traducerea: „azucar” devine „sugar”. Lista are câteva zeci de cuvinte.
3. **Corectarea greșelilor.** Un cuvânt rămâne neschimbat dacă are sub 4 litere, dacă are cifre sau dacă e cunoscut. Cunoscut înseamnă că apare în bibliotecă sau în vocabularul modelului. Pentru un cuvânt necunoscut se încearcă pe rând:
   1. Cel mai apropiat cuvânt din bibliotecă, la o singură schimbare. O schimbare este o literă adăugată, ștearsă, înlocuită sau două litere vecine inversate. De exemplu „thier” e la o schimbare de „their”. Se iau doar cuvinte care apar de cel puțin 3 ori. La egalitate câștigă cel mai des.
   2. Despărțirea în două cuvinte. Fiecare jumătate trebuie să aibă cel puțin 3 litere și cel puțin 20 de apariții în bibliotecă. Așa se repară cuvintele lipite.
   3. Pentru cuvinte de cel puțin 7 litere, cel mai apropiat cuvânt la două schimbări.
   4. Dacă nimic nu merge, cuvântul rămâne cum e.
4. **Cuvintele pentru BM25.** Se scot cuvintele de legătură și cuvintele care țin de formularea biletului („part”, „chapter”, „story”). Restul se aduc la rădăcină.
5. **Semnalul p: BM25 pe pasaje.** Fiecare pasaj primește puncte pentru fiecare cuvânt din întrebare pe care îl conține. Un cuvânt rar aduce mai multe puncte. Repetarea aduce puncte tot mai puține, iar pasajele lungi sunt ușor penalizate. Documentul ia scorul celui mai bun pasaj al lui.
6. **Semnalul d: BM25 pe documentul întreg.** Același calcul, făcut pe tot documentul.
7. **Vectorul de sens al întrebării.** Întrebarea curățată intră în model, cu cel mult 128 de tokeni. Modelul dă câte un vector pe token și se face media lor. Vectorul de 384 de numere trece prin tabelul de conversie și rămâne cu 256. Pasul durează 10-17 ms. Curățarea ajută și aici: între 1,5 și 3 puncte în plus la CV.
8. **Semnalul e: sens pe pasaje.** Se calculează cosinusul dintre vectorul întrebării și vectorul fiecărui pasaj. Documentul ia scorul celui mai bun pasaj al lui.
9. **Semnalul ed: sens pe documentul întreg.** Cosinusul dintre vectorul întrebării și vectorul documentului.
10. **Aducerea la aceeași scară.** Scorurile BM25 și cosinusurile au mărimi diferite și nu se pot aduna direct. Pentru fiecare semnal se calculează `(scor - mediană) / (maxim - mediană)`. Cel mai bun document primește 1, iar documentul de la mijlocul clasamentului primește 0.
11. **Suma cu ponderi.** Scorul unui document este `0.4 * p + 0.1 * d + 0.5 * e + 0.2 * ed`.
12. **Răspunsul.** Documentele se ordonează după scor. Se iau primele, până se strâng 3 identificatori. Un document cu copii exacte își aduce toate copiile.

Semnalele p și d prind întrebările care au cuvinte comune cu textul. Semnalele e și ed sunt cele care pot prinde ghicitorile, de felul „captive Africans” pentru un capitol despre sclavi.

## Cât de bine merge

Versiunea 4 ia 0,940 pe toate întrebările publice, 1,000 pe Wikipedia și 0,900 pe non-Wikipedia. Împărțirea pe versiuni este în tabelul de la „Unde sunt greșelile”. Mai jos sunt celelalte măsuri date de `evaluate.py`, pe același calculator și pe un singur fir:

| Măsură | Versiunea 1 | Versiunea 2 | Versiunea 4 |
|---|---|---|---|
| Pe cele două jumătăți ale întrebărilor publice | 0,896 / 0,851 | 0,940 / 0,896 | 0,940 / 0,940 |
| CV pe non-Wikipedia | nu apare în surse | 0,865 ± 0,042 | 0,893 ± 0,028 |
| CV pe toate | 0,868 | 0,920 | 0,937 ± 0,016 |
| Întrebări grele (43) | 0,293 | 0,535 | 0,628 |
| Cu greșeli de tastare puse intenționat | 0,866 | 0,899 | 0,937 |
| 59 de întrebări proprii | 0,932 | 0,949 | 0,949 |
| Non-Wikipedia, documentul corect în primele 10 | nu apare în surse | 0,950 | 0,944 |
| Mărime (`main.py` + `index/`) | 34,8 MB | 45,8 MB | 51,26 MB |

Ce s-a schimbat de la versiunea 2 la versiunea 4:

- Versiunea 4 repară 7 întrebări ratate de versiunea 2: q0048, q0058, q0078, q0090, q0182, q0210, q0253. Strică una: q0079.
- Ratările scad de la 22 la 16. Toate cele 16 sunt capitole de cărți. Sunt găsite 131 din 147 de capitole.
- Cele 16 ratări rămase sunt tot de felul „cartea greșită”. Din cele 48 de documente întoarse la ele, 38 sunt capitole din altă carte.
- Ponderile potrivite pe toate întrebările publice ies egale cu cele livrate. La fel ies și cele potrivite doar pentru non-Wikipedia.
- Două măsuri nu au crescut. Cele 59 de întrebări proprii rămân la 0,949. Măsura cu primele 10 scade ușor, de la 0,950 la 0,944.

Viteză și mărime pentru versiunea 4, în verificarea oficială:

| Măsură | Rularea 1 | Rularea 2 | Limită |
|---|---|---|---|
| Mărime | 51,26 MB | 51,26 MB | cel mult 53,0 MB |
| Încărcare | 0,86 s | 0,88 s | cel mult 10,0 s |
| p95 al unei căutări | 17,1 ms | 21,0 ms | cel mult 75,0 ms |
| hit@3 pe întrebările publice | 0,940 | 0,940 | cel puțin 0,8 |

`README.md`, care e mai nou, scrie că o căutare durează 10-17 ms. `DETALII.md` dă același interval doar pentru trecerea prin model. Cifrele măsurate de verificator sunt cele din tabel.

Estimarea pentru platformă, din `DETALII.md`, neverificată acolo:

- hit@3 în jur de 0,89-0,90, dacă întrebările ascunse seamănă cu cele non-Wikipedia. Versiunea 2 ar lua cam 0,86.
- p95 în jur de 25-35 ms. Versiunea 1 a rulat pe platformă de 1,3-1,6 ori mai lent decât local.

## Cum a fost verificată

1. **Scorul local, cu `evaluate.py`.** Scriptul încarcă `main.py` și `index/`, face 10 căutări de încălzire și apoi pune toate întrebările publice. Ieșirea pentru versiunea 4 este în `_exp/evaluate_v4.txt`. Acolo p95 e 16,7 ms, iar cea mai lentă căutare 34,7 ms. Scriptul scrie aceste rânduri:
   - `public` și `halves`: toate cele 268 de întrebări, apoi împărțite în două jumătăți. Jumătățile arată cât se mișcă scorul de la un set de întrebări la altul.
   - `wiki` și `nonwiki`: împărțirea după felul documentului corect.
   - `extra`: 59 de întrebări scrise de noi, în `extra_questions.jsonl`. Sunt despre documente pe care întrebările publice nu le ating. 23 sunt despre Wikipedia și 36 despre restul.
   - `hard`: întrebările grele, la care BM25 singur ratează.
   - `typos`: aceleași întrebări publice, stricate intenționat. Cam unul din cinci cuvinte mai lungi de 4 litere primește o greșeală: două litere inversate, o literă lipsă sau o literă dublată.
   - `cv`: validarea încrucișată, cu 5 părți și 3 repetări. Ultima linie arată ponderile găsite pe toate întrebările publice. Sunt cele livrate.
2. **Verificarea oficială, pe arhivă.** `..\task_cuba\local_check.py` a fost rulat de două ori pe `cuba_v4.zip`. Ambele rulări au trecut toate cele patru limite. Ieșirea este în `local_check_output.txt`, cu versiunile 2 și 3 alături pentru comparație.
3. **Verificarea cuantizării.** Vectorii dați de modelul int8 au fost comparați cu cei ai modelului exact. Cosinusul mediu e 0,9989, iar cel mai mic 0,9969.
4. **Experimentele.** Tabelele din `_exp/results.md` vin din scripturi separate, cum e `_exp/eval5.py`. Acolo CV e făcut cu 4 repetări, iar ponderile sunt potrivite după scorul pe non-Wikipedia. În `evaluate.py` sunt 3 repetări. Cifrele diferă puțin: CV pe non-Wikipedia pentru gte-small e 0,888 acolo și 0,893 în `evaluate.py`.
5. **Platforma.** Singurul rezultat de pe platformă din surse este 0,793 pentru versiunea 1.

## Ce nu e sigur

- **`onnxruntime` pe platformă.** `local_check.py` îl are pe lista modulelor pe care evaluatorul le poate importa. Fișierul `..\task_cuba\baseline\requirements.txt` nu îl conține. De aceea există rezerva.
- **Rezerva e mai slabă și mai lentă.** Versiunea 3 are 0,856 pe non-Wikipedia. p95 e 40-51 ms și ajunge la 95-116 ms când calculatorul e ocupat la maximum. Dacă platforma e încărcată, poate pica limita de 75 ms.
- **Cum arată întrebările ascunse.** Presupunerea că sunt mai ales despre cărți vine dintr-o singură potrivire: 0,794 local și 0,793 pe platformă la versiunea 1. Dacă sunt mai grele decât cele publice, scorul scade.
- **Rezerva față de prag.** Estimarea e 0,89-0,90 și pragul e 0,80. CV pe non-Wikipedia dă 0,893 ± 0,028. Sunt doar 160 de întrebări non-Wikipedia, deci o întrebare mută scorul cu 0,006.
- **Viteza pe platformă.** 25-35 ms este o estimare făcută din încetinirea văzută la versiunea 1. Nu e o măsurătoare pentru versiunea 4.
- **Mărimea.** Este limita cea mai strânsă. Rămân 1,74 MB până la 53 MB, deci orice fișier adăugat în `index/` trebuie socotit.
- **Cele 16 ratări rămase.** Nicio metodă încercată nu le rezolvă. Cross-encoderul nu încape, iar îmbogățirea pasajelor nu a fost rulată până la capăt.
- **Câștigul de la pasaje mai mici.** O variantă a dat 0,906 pe non-Wikipedia, dar a fost socotită întâmplare. Nu se știe dacă pe întrebările ascunse ar fi fost mai bună.
- **Refacerea.** Sursele nu spun dacă o reconstruire de la zero dă o arhivă cu același sha256. Suma de control e a arhivei existente.
- **Diferențe între surse.** 21 sau 22 de ratări la versiunea 2 și 10-17 ms sau 17-21 ms pe căutare. Ambele sunt explicate mai sus. p95 pentru versiunea 3 apare și ca 40-50 ms, și ca 40-51 ms.

## Unde e fiecare lucru

Ce se trimite:

| Fișier | Mărime | La ce folosește |
|---|---|---|
| `main.py` | 0,01 MB | codul: `Retriever(index_dir)` și `search(question, k)` |
| `index/encoder.onnx` | 34,04 MB | modelul gte-small, cuantizat int8 |
| `index/dense.npz` | 10,26 MB | vectorii pasajelor ca int8, scările lor, tabelul de conversie PCA |
| `index/lex.npz` | 4,74 MB | listele BM25 pentru pasaje și pentru documente întregi |
| `index/words.txt` | 0,95 MB | cuvintele bibliotecii cu numărul de apariții, pentru corectarea greșelilor |
| `index/qtokenizer.json` | 0,71 MB | împărțirea textului în tokeni, pentru model |
| `index/terms.txt` | 0,50 MB | lista rădăcinilor de cuvinte din BM25 |
| `index/docs.json` | 0,04 MB | identificatorii documentelor, cu copiile grupate |
| `index/config.json` | sub 1 KB | ponderile și setările |

Mărimile pe fișier sunt citite din folder și rotunjite. Totalul măsurat de verificator este 51,26 MB.

Scripturi și documente:

- `build_index.py`: construiește `index/` din bibliotecă, din vectorii pasajelor și din modelul exportat.
- `encode_passages.py`: citește biblioteca, o taie în pasaje și calculează vectorii pasajelor pe placa video. Citirea și tăierea sunt folosite și de `build_index.py`.
- `export_onnx.py`: exportă modelul în format ONNX și îl cuantizează int8, cu o scară pe canal.
- `evaluate.py`: măsoară scorul local pentru orice folder cu `main.py` și `index/`.
- `extra_questions.jsonl`: cele 59 de întrebări proprii.
- `local_check_output.txt`: ieșirea verificării oficiale pentru versiunea 4, cu versiunile 2 și 3 alături.
- `README.md` și `DETALII.md`: descrierea scurtă în română și cea lungă în engleză, cu toate versiunile. Titlul celei lungi numește soluția „Zemí”. Sursele nu explică numele.
- `EXPLICATIE.md`: documentul de față.
- `submission.zip`: versiunea 1, cea care a luat 0,793 pe platformă.
- `.gitignore`: scoate din repo folderele `.models/`, `_onnx/` și `_exp/`.

Foldere de lucru:

- `.models/`: modelele descărcate. Sunt folosite doar la construire. `.models/gte-small` este cel din versiunea 4.
- `.venv/`: mediul Python al folderului. Are numpy, tokenizers, PyStemmer și onnxruntime.
- `_onnx/`: fișierele intermediare ale construirii. `_onnx/gte_int8_pc.onnx` este modelul cuantizat. `_onnx/pv_gte.npy` ține vectorii pasajelor înainte de scurtare. `_onnx/arctic_int8_pc.onnx` și `_onnx/pv_arctic.npy` sunt aceleași lucruri pentru modelul arctic.
- `_exp/results.md`: toate tabelele de experimente.
- `_exp/evaluate_v4.txt`, `_exp/evaluate_v2.txt`: ieșirea lui `evaluate.py` pentru versiunile 4 și 2.
- `_exp/eval5.py` cu `_exp/eval5.log` și `_exp/eval6.py`: măsurarea modelelor în forma care se livrează, apoi încercarea cu pasaje mai mici doar pentru cărți.
- `_exp/sparse4.py`: scriptul de îmbogățire a pasajelor, reparat dar nerulat până la capăt.
- `_exp/v3/`: versiunea 3. Are `main.py` cu modelul scris în numpy, `encoder.npz` și `export_weights.py`.
- `_exp/v2_sub/`: versiunea 2 despachetată, cu `main.py` și `index/`. `_exp/evaluate_v2.txt` a fost rulat pe ea.
- `_exp/v1_sub/` și `_exp/cand_gte/`: tot câte un `main.py` și un `index/`. Rolul lor reiese doar din nume: versiunea 1 și o variantă de probă cu gte.
- Restul din `_exp/`: scripturi de experiment, jurnale și fișiere mari de lucru (`.npy`, `.npz`).

Arhivele `cuba_v4.zip`, `cuba_v3.zip` și `cuba_v2.zip` sunt în `C:\Users\user\Downloads\LMML_upload\Cuba\`.

## Cum refaci totul

Ce îți trebuie:

- Mediul miniconda. Are torch, transformers, onnx și onnxruntime. E folosit la exportul modelului și la vectorii pasajelor. Pentru vectori e nevoie de placă video.
- Mediul `.venv` din folder, descris mai sus. E folosit la tot restul.
- Biblioteca în `..\task_cuba\data`.

```
.models/gte-small  --export_onnx.py------> _onnx/gte_int8_pc.onnx --+
..\task_cuba\data  --encode_passages.py--> _onnx/pv_gte.npy --------+--build_index.py--> index/
..\task_cuba\data  -------------------------------------------------+
main.py + index/   --zip--> cuba_v4.zip --local_check.py--> PASS sau FAIL
```

Comenzile, rulate în ordine din folderul taskului:

```
# 1. descarci modelul, o singură dată
huggingface-cli download thenlper/gte-small --local-dir .models/gte-small
# 2. exporți modelul pentru întrebări -> _onnx/gte_int8_pc.onnx (34,0 MB)
C:\Users\user\miniconda3\python.exe export_onnx.py
# 3. calculezi vectorii pasajelor pe placa video, cam un minut -> _onnx/pv_gte.npy
C:\Users\user\miniconda3\python.exe encode_passages.py
# 4. construiești index/
.venv\Scripts\python.exe build_index.py
# 5. măsori scorul local; pentru comparație, ieșirea versiunii 4 este în _exp/evaluate_v4.txt
.venv\Scripts\python.exe evaluate.py . --cv
# 6. faci arhiva, cu main.py și index/ la nivelul de sus
.venv\Scripts\python.exe -c "import zipfile,os; z=zipfile.ZipFile(r'C:\Users\user\Downloads\LMML_upload\Cuba\cuba_v4.zip','w',zipfile.ZIP_DEFLATED); z.write('main.py'); [z.write(os.path.join('index',f),'index/'+f) for f in os.listdir('index')]; z.close()"
# 7. rulezi verificarea oficială pe arhivă
.venv\Scripts\python.exe ..\task_cuba\local_check.py C:\Users\user\Downloads\LMML_upload\Cuba\cuba_v4.zip
```

De ținut minte:

- La pasul 4, `build_index.py` șterge `index/` și îl face din nou. Afișează numărul de documente (1552, din care 1524 unice), numărul de pasaje (37944) și mărimea indexului.
- La pasul 7, verificarea se rulează pe arhivă, nu pe folder. Altfel `.models/` și `.venv/` ar intra la mărime.
- Regulile repo-ului exclud `index/dense.npz`, `index/lex.npz` și `index/encoder.onnx`. Ca să ajungă în repo se adaugă cu `git add -f`. Altfel se refac cu pașii 1-4.
- Vectorii pasajelor și indexul trebuie făcuți cu aceeași tăiere în pasaje. `build_index.py` se oprește dacă numărul de vectori diferă de numărul de pasaje.
- Pentru alt model se dau opțiunile `--model`, `--pool`, `--prefix` și `--tokenizer` la cele trei scripturi. bge-small și arctic cer `--pool cls` și prefixul de întrebare în `build_index.py`.
- Sursele nu descriu pașii de refacere pentru `cuba_v3.zip`. Codul versiunii 3 este în `_exp/v3/`, iar arhiva gata făcută este în folderul de încărcare.
