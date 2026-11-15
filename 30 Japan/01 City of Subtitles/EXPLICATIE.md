# Japonia: City of Subtitles. Explicația completă

Documentul explică taskul și tot ce s-a lucrat la el. E scris pentru un coleg care nu a făcut munca și trebuie să o poată explica unui mentor. Taskul **nu e rezolvat**. Prima jumătate e făcută și e de încredere: știm cine a făcut fiecare dintre cele 145 de poze nesemnate. A doua jumătate lipsește: cuvântul ascuns care dă flag-ul nu a fost găsit.

## Ce se pune pe site

Nimic. Taskul se predă cu un flag de forma `SIGMOID_<CUVÂNT>`, plus un link GitHub către descrierea soluției. Flag-ul se scrie cu cuvântul ascuns. Cuvântul nu a fost găsit. Fără el nu avem ce introduce.

Avem doar tabelul `attribution_table.csv`. El spune cine a făcut fiecare poză nesemnată. E un rezultat intermediar. Nicio sursă nu spune că site-ul îl primește ca răspuns. Sursele nu pomenesc nicio trimitere pe platformă. Nu se știe dacă s-a încercat vreun cuvânt acolo.

## Ce cere taskul

Șase fotografi postează poze din oraș. Fiecare poză are o subtitrare în japoneză, cu text deschis pe o bandă neagră. Unul dintre ei se numește toorisugari. README traduce numele prin „trecătorul”.

| Săptămâni | Câte poze | Semnate |
|---|---|---|
| 1-7 | 108 | da, câte 18 de fotograf |
| 8-12 | 145 | nu, câte 29 pe săptămână |

Taskul cere trei lucruri, în ordinea asta:

1. Afli cine a făcut fiecare poză nesemnată. DETALII.md spune că atribuirea se face după zgomotul senzorului camerei. Pozele semnate sunt materialul de învățare.
2. Alegi pozele lui Toorisugari.
3. Scoți din ele un cuvânt ascuns. Flag-ul este `SIGMOID_<CUVÂNT>`.

**Zgomotul senzorului** înseamnă că fiecare pixel al unui senzor răspunde puțin diferit la aceeași lumină. Diferențele vin din fabricație și rămân aceleași în orice poză. În engleză se numește PRNU. Harta acestor diferențe, cu un număr pentru fiecare pixel, este **amprenta camerei**.

Indiciul primit:

> The photos of a simple passer-by might know more than they show. Who knows, maybe an unknown photographer identity hides there.

Pe românește: pozele unui simplu trecător ar putea ști mai mult decât arată. Poate acolo se ascunde identitatea unui fotograf necunoscut. DETALII.md îl numește „indiciul din săptămâna 13”. Sursele nu explică de unde vine numele.

Enunțul complet nu e copiat în folderul taskului. Ce știm despre el vine din `README.md` și `DETALII.md`. Datele stau în afara folderului. Numele fișierului spune săptămâna și numărul de ordine: `w08_001.jpg` e prima poză din săptămâna 8.

```
../task_japan/dataset/
   credited/<fotograf>/w01_004.jpg ...    poze semnate
   anonymous/w08_001.jpg ...              poze nesemnate
```

## Ce am aflat din date

- **Fișierele sunt toate la fel.** Sunt 253 de poze de 1280 x 960 pixeli. Toate fișierele JPEG au aceeași alcătuire și aceleași setări de compresie. Niciunul nu are metadate. La nivel de fișier nimic nu deosebește fotografii. Rămâne doar zgomotul din pixeli.
- **Zgomotul senzorului e slab.** O poză pusă lângă o altă poză de la aceeași cameră nu arată nimic. Trebuie adunate multe poze ca să vezi camera.
- **Amayadori are cel mai slab senzor.** Toate cele trei greșeli pe pozele semnate sunt la el.
- **Multe poze sunt rotite cu 180 de grade.** Au fost făcute cu telefonul ținut invers și postate drept. La ele zgomotul senzorului apare rotit. Se întâmplă la toți șase. Din cele 45 de poze nesemnate ale lui Toorisugari, 21 sunt rotite. Nicio poză nu e oglindită.
- **Toorisugari postează mai mult.** Are exact 9 poze nesemnate pe săptămână, 45 în total. Fiecare dintre ceilalți cinci are exact 4 pe săptămână, 20 în total. Suma pe săptămână e 29 = 9 + 5 x 4. E singurul lucru din date care îl scoate în evidență.
- **Subtitrările par puse la întâmplare.** Au între 1 și 6 rânduri, deși povestea vorbea de un singur rând. Sunt combinații de tipul „fragment, fragment”. După citirea automată ies cam 490 de prime jumătăți și 310 de a doua jumătăți.
- **Yakin și Toorisugari nu se confundă.** DETALII.md pomenește că ar avea „același model de telefon”. Amprentele camerelor lor se corelează la 0.0018, ca orice altă pereche.

Câte dintre cele 18 poze semnate ale fiecăruia sunt rotite, adică între 22% și 56%:

| amayadori | jihanki | rojiura | shuuden | toorisugari | yakin |
|---|---|---|---|---|---|
| 9 | 10 | 4 | 7 | 4 | 8 |

## Ce am încercat, în ordine

Două noțiuni, înainte de pași:

- **Reziduul.** Poza minus o variantă netezită a ei. Netezirea păstrează scena și aruncă zgomotul. Scăderea lasă zgomotul și puține urme de contururi.
- **Testul cu poza scoasă.** Iei o poză semnată și o scoți din amprenta fotografului ei. Apoi întrebi metoda cine a făcut-o. Repeți pentru toate cele 108. Fără scoatere poza s-ar recunoaște pe ea însăși. În engleză: leave-one-out.

Metoda a crescut în cinci pași. Cifra e procentul de poze semnate nimerite la testul cu poza scoasă. La ghicit ar ieși cam 17%, adică una din șase.

| Pas | Ce s-a schimbat | Nimerite |
|---|---|---|
| 1 | netezire Wiener din scipy și corelație simplă | 56% |
| 2 | Wiener local, egalizarea reziduului, scoaterea părții comune | 72% |
| 3 | netezire cu wavelet | 79% |
| 4 | rafinare cu pozele nesemnate | 86% |
| 5 | ambele orientări ale pozei | 97% |

**Pasul 1, 56%.** Poza se netezește cu filtrul Wiener din biblioteca scipy și se scade. Din reziduurile pozelor semnate iese amprenta fiecărui fotograf. Scorul e corelația simplă dintre reziduul pozei și amprentă, un număr între -1 și 1.

**Pasul 2, 72%.** Trei schimbări:

1. Netezirea devine „Wiener local”. DETALII.md nu dă alte amănunte.
2. Reziduul e egalizat. În scriptul final asta înseamnă împărțirea la tăria lui locală. Problema era că zonele cu multă textură aveau reziduu mare și dominau corelația.
3. Din fiecare amprentă se scade media celor șase amprente. Problema era că toate pozele trec prin aceeași prelucrare. Ea lasă aceeași urmă în toate amprentele și le face să semene.

**Pasul 3, 79%.** Netezirea se face cu wavelet. Un wavelet desparte poza în detalii fine și detalii mari, pe mai multe niveluri. Fiecare nivel e curățat separat. Aici sunt 4 niveluri, tipul db4 și sigma 3. E rețeta lui Mihcak, folosită de obicei pentru zgomotul senzorului.

**Pasul 4, 86%.** Problema era că 18 poze de fotograf dau o amprentă zgomotoasă. Soluția: pozele nesemnate atribuite cu siguranță intră în amprenta fotografului lor. Amprenta se curăță, apoi totul se reia. `attribute2.py` face această rafinare. Folosește deja scorul z, explicat în secțiunea următoare.

**Ocolul cu „a șaptea cameră”.** Prima atribuire a rupt pozele lui Toorisugari în două grupuri. Amprentele celor două grupuri aveau corelația 0.0005, adică nimic în comun. DETALII.md descrie momentul așa: ca și cum o a șaptea cameră, necunoscută, s-ar ascunde în pozele lui. `gram.py` și `groups.py` au despărțit grupurile, numite T și B. `attribute7.py` a refăcut atribuirea cu șapte camere. `seventh.py` și `split_check.py` au căutat mai departe a șaptea cameră.

Apoi `orient_check.py` a comparat amprenta B cu toate celelalte, așa cum sunt, rotite și oglindite. Amprenta B s-a potrivit cu amprenta T rotită cu 180 de grade. Potrivirea a fost PCE 1542, fără nicio deplasare. PCE spune cât de sus iese vârful potrivirii față de restul hărții de corelație. Ca scară: zgomotul curat dă cam 28, chiar când alegi cea mai bună dintre toate deplasările. Deci nu există a șaptea cameră. E același telefon, ținut invers.

**Pasul 5, 97%.** Toți fotografii fac asta, nu doar Toorisugari. De aceea fiecare poză e testată cu fiecare amprentă în ambele orientări. Amprentele se construiesc cu fiecare poză întoarsă la orientarea senzorului. Oglindirea a fost testată separat cu `orient4.py`. Nicio poză nu are nevoie de ea. Un exemplu este poza `w08_002`. Cu șase camere avea z 6.14 și margine 2.07, în `attribution_6cam.csv`. Cu șapte camere era la toorisugari_B cu z 10.73, în `attribution_7cam.csv`. În tabelul final e la toorisugari, rotită, cu z 15.46 și margine 13.85.

**Cota săptămânală.** După pasul 5 rezultatul liber dădea exact 9 poze Toorisugari în fiecare săptămână. Ceilalți aveau 4 pe săptămână, cu două excepții. `finalize_table.py` impune numerele 9 și 4 și mută cele două poze.

## Partea rezolvată, pas cu pas: cine a făcut fiecare poză

Trei scripturi fac tot drumul:

```
253 de poze
   |   feats.py  (folosește prnu.py și wavelet.py)
   v
cache/feats.pt           reziduu, luminozitate și mască pentru fiecare poză
   |   attribute_final.py
   v
attribution.csv          145 de poze nesemnate, 12 scoruri z pe poză
credited_check.csv       108 poze semnate, aceleași scoruri, pentru control
   |   finalize_table.py
   v
attribution_table.csv    tabelul final
```

**1. Masca.** Subtitrarea e desenată peste poză, deci nu are zgomot de senzor. Un rând de pixeli e socotit bandă dacă peste 60% din el e aproape negru. Banda se lățește cu 12 rânduri, de siguranță. Se scot și pixelii arși sau complet negri. Nici ei nu poartă zgomot.

**2. Reziduul.** Se face pe fiecare canal de culoare. Mediile pe rânduri și coloane se scot pentru că țin de prelucrare, nu de senzor.

```
reziduu = poza - poza netezită cu wavelet
reziduu --> împărțit la tăria lui locală (fereastră 15 x 15)
        --> scoasă media fiecărui rând și a fiecărei coloane
        --> înmulțit cu masca
```

**3. Amprenta.** Notăm reziduul cu W și luminozitatea pozei cu I. Calculul se face pixel cu pixel, peste toate pozele unui fotograf. Luminozitatea apare pentru că zgomotul senzorului crește cu lumina. Un pixel luminos poartă mai multă amprentă decât unul întunecat.

```
K = suma(W x I) / suma(I x I)
K = K - media celor șase amprente
```

**4. Scorul z.** Pentru o poză și o amprentă:

1. Se calculează zgomotul așteptat, K x I, cu luminozitatea acestei poze.
2. Se corelează cu reziduul pozei. Iese un număr mic, greu de citit singur.
3. Zgomotul așteptat se mută cu 12 deplasări mari, de exemplu 97 de rânduri și 131 de coloane. La poziții greșite nu poate exista potrivire reală. Corelațiile de acolo arată cât de mare iese o potrivire din întâmplare.
4. Scorul z e corelația reală împărțită la împrăștierea acestor corelații întâmplătoare.

Un z aproape de 0 înseamnă nicio potrivire. Un z de 16 înseamnă o potrivire de 16 ori peste nivelul întâmplării. **Marginea** e diferența dintre cel mai bun z și următorul. Arată cât de departe e al doilea candidat.

**5. Cele 12 variante.** Fiecare poză primește 12 scoruri z: 6 fotografi x 2 orientări. Câștigă varianta cu z cel mai mare. Ea dă și fotograful, și dacă poza e rotită.

**6. Rafinarea.** Se repetă de cel mult 7 ori:

1. La început amprentele vin doar din pozele semnate.
2. Se calculează scorurile pentru toate cele 253 de poze. O poză aflată deja într-o amprentă e scoasă din ea înainte de calcul.
3. Pozele semnate își păstrează fotograful. Li se alege doar orientarea cu z mai mare.
4. Pozele nesemnate cu margine peste 3 intră în amprenta câștigătoare, întoarse la orientarea senzorului.
5. Amprentele se refac și se reia de la 2.

Ce arată `attribute_final.log`:

| Trecerea | Semnate nimerite | Nesemnate intrate în amprente | Poze date lui Toorisugari |
|---|---|---|---|
| 0 | 82.4% | 75 | 44 |
| 1 | 94.4% | 113 | 46 |
| 2 | 95.4% | 128 | 46 |
| 3 | 96.3% | 132 | 45 |
| 4 | 97.2% | 133 | 45 |
| 5 | 97.2% | 138 | 45 |
| 6 | 97.2% | 137 | 45 |

**7. Cota săptămânală.** Rezultatul liber dă 18 poze la amayadori și câte 21 la shuuden și yakin. Lipsesc două la amayadori. `finalize_table.py` ia fiecare săptămână separat. Are 29 de poze și 29 de locuri: 9 pentru Toorisugari și câte 4 pentru ceilalți. Alege așezarea cu suma scorurilor z cea mai mare. Se schimbă două poze:

| Poza | Alegerea liberă | După cotă |
|---|---|---|
| `w08_007` | shuuden, z 3.19 | amayadori, z 3.09 |
| `w11_008` | yakin, z 5.12 | amayadori, z 4.85 |

**8. Tabelul final.** `attribution_table.csv` are un rând pe poză. Coloanele `z` și `z_margin_vs_next` sunt scorul și marginea fotografului ales. `rotated180` e 1 la pozele rotite. `note` are `quota` la cele două poze mutate. Pozele lui Toorisugari, după numărul de ordine din săptămână:

| Săptămâna | Poze |
|---|---|
| 8 | 002 004 006 010 011 013 017 019 023 |
| 9 | 001 004 008 011 013 014 015 021 022 |
| 10 | 010 011 012 018 019 020 022 027 028 |
| 11 | 003 011 012 014 016 025 027 028 029 |
| 12 | 001 002 005 008 010 020 025 027 029 |

## Partea nerezolvată: cuvântul ascuns

Nu se știe unde e cuvântul. S-a presupus steganografie, adică un mesaj ascuns într-un fișier care arată normal. S-a căutat în toate locurile de mai jos. Toate au ieșit negative.

Două expresii revin des. **Cadrul postat** e poza așa cum a fost publicată. **Cadrul senzorului** e poza întoarsă la loc, așa cum a văzut-o senzorul. La o poză rotită cele două diferă. Un semn adăugat de cont după rotire ar sta pe loc în cadrul postat. Amprenta camerei stă pe loc în cadrul senzorului.

### 1. În zgomotul camerei lui Toorisugari

- **Un al doilea tipar, în cadrul postat.**
  - `posted_pattern.py`, `posted_additive.py`, `posted_cross.py`: pozele nerotite și cele rotite sunt mediate separat. În cadrul postat amprentele lor nu se suprapun. Dacă mediile seamănă, există ceva adăugat după rotire. La început a apărut un efect de circa 3 abateri standard. Nu a rezistat la testul fără eroare sistematică din secțiunea 2.
  - `hidden_pattern.py`: scoate amprenta proprie din fiecare poză și mediază restul. Fișierul `out/hidden_pattern_z.csv` nu poate fi folosit. Toate scorurile lui Toorisugari sunt în jur de -100, pentru că poza testată intră în construcția tiparului.
  - `posted_render.py`, `posted_fullres.py`: desenează media reziduurilor celor 45 de poze, cu amprenta proprie scoasă. Al doilea lucrează la rezoluție plină, pentru că mediile pe blocuri șterg marginile literelor. Se văd doar urme de scene și mai mult zgomot jos. Niciun text. Imaginile sunt `out/pf_*.png`.
  - `signature_scan.py`, `secret_scan.py`: caută un semn slab la un loc fix și o textură comună între perechi de poze. Rezultatul lor nu e notat separat în surse.
- **Ceva doar în pozele nesemnate.**
  - `extra_component.py`: compară jumătăți de poze nesemnate între ele și cu pozele semnate. Rezultat negativ.
  - `k_split.py`: face două amprente separate, una din cele 18 poze semnate și una din cele 45 nesemnate. Le scade și desenează diferența, pe total și pe fiecare săptămână. Yakin e martorul. Corelația dintre cele două amprente e 0.039 la Toorisugari și 0.028 la yakin. Hărțile sunt pete mărunte uniforme. Variantele netezite arată doar umbre ale scenelor: fațade, crengi, o formă rotundă mare în săptămâna 10. Nicio literă. Toorisugari arată ca martorul.
- **O a doua cameră într-o parte din poze.**
  - `pairs_after_removal.py`: compară perechi de poze după scoaterea amprentei proprii. Rezultat negativ.
  - `photo_presence.py`, `presence_map.py`, `fresh_look/presence_pool.py`: hărți pe blocuri cu locurile unde amprenta e prezentă. Un cuvânt scris prin lipsa amprentei s-ar vedea. Ies pete uniforme.
  - Al doilea cel mai bun z al celor 45 de poze e 2.75 în medie. La ceilalți e 2.6-3.4. Maximul e 6.8.
  - `fresh_look/gram2.py`, `fresh_look/gram3.py`: tabelul de corelații între toate pozele curățate. Se uită la cea mai mare valoare proprie, un număr care crește când un grup de poze împarte un tipar. Cifrele sunt în tabelul de mai jos.
- **Un desen în amprentă.** Amprenta a fost desenată netezită și ca energie locală, cu `render_sensor.py` și `render_fp.py`. A fost desenată pe benzi de textură cu `texture_maps.py` și pe diferențe de culoare cu `chroma_scan.py`. `line_patterns.py` a scos profilele pe rânduri și coloane. `sensor_xcorr.py` și `local_xcorr.py` au comparat-o local cu amprentele celorlalți. `flatfield.py` a căutat o variație lentă de luminozitate. Nu apare niciun desen.
- **Tăria și poziția amprentei.**
  - `prnu_gain.py`: cât de tare e amprenta în fiecare poză. Pe săptămâna 8 valorile merg de la 0.073 la 0.112. Pe săptămâna 10 merg de la 0.053 la 0.156. E o împrăștiere lină, fără două trepte.
  - `shift_scan.py`: caută poze decupate sau deplasate față de senzor. Nu găsește.
  - `fresh_look/lowmean.py`: caută un semn neted, pe care reziduul l-ar șterge. Media celor 45 de poze arată doar margini de scene suprapuse.

Cea mai mare valoare proprie, pe trei geometrii:

| Grup | Cadrul postat | Cadrul senzorului | Anti-senzor | Limita zgomotului |
|---|---|---|---|---|
| 45 de poze nesemnate | 12.4 | 13.3 | 12.0 | 13.4 |
| toate cele 63 | 14.3 | 16.0 | 14.4 | 15.9 |

Un grup de 25 de poze cu un tipar comun ar fi dat cam 25. DETALII.md nu explică a treia geometrie, numită anti-senzor.

### 2. Un senzor ascuns în pozele oricui

Întrebarea aici: împart niște poze, ale oricui, un tipar de senzor care nu e al fotografului lor? Dacă apărea un grup, apartenența la grup se citea ca biți. Nu a apărut niciun grup.

1. `clean_residuals.py` scoate din fiecare poză amprenta fotografului ei. Rămâne `cache/clean_res.pt`.
2. `pce_pairs.py` compară toate cele 31878 de perechi, la orice deplasare. Încearcă șase poziții relative: la fel, rotit cu 180, două oglindiri și rotirile cu 90 și 270 de grade.
3. Prima rulare a dat 1097 de perechi peste PCE 60. PCE-ul median era însă 40, iar zgomotul curat dă cam 28. Reziduurile nu erau zgomot curat, deci pragul nu însemna nimic.
4. Varianta `pce_pairs.py phat` păstrează doar alinierea, nu și tăria fiecărei frecvențe. Rămân 47 de perechi, cu median 31.
5. Perechile din vârf au toate în ele una dintre cinci poze semnate, făcute noaptea și foarte întunecate. Lista lor e sub pași. Ele se potrivesc între ele în orice poziție. 12 din cele 34 de potriviri ale lor stau la deplasări multiple de 8 pe ambele direcții. Din întâmplare ar fi 1 din 64. Opt e latura blocului JPEG. Un senzor comun s-ar potrivi într-o singură poziție.
6. Fără cele cinci poze rămân 13 perechi peste 60. Zgomotul ar da 11.6. Grupurile cele mai mari sunt lanțuri, cu fotografi amestecați.
7. `pce_scales.py` compară cele 63 de poze ale lui Toorisugari cu toate cele 253, la patru scări între 0.90 și 1.10. Peste 60 apar doar aceleași poze întunecate.
8. `pce_power.py` măsoară cât de sensibil e testul pe perechi. Perechi care chiar împart un senzor dau PCE 25-43, la fel ca perechile de la senzori diferiți. Testul pe perechi ar prinde doar un senzor ascuns mult mai tare decât cele reale.

```
amayadori w06_009     jihanki w06_006     toorisugari w02_013 w02_018 w05_013
```

De aceea s-au făcut teste pe grupuri, care au putere. Fiecare poză e comparată cu media multor altora.

| Script | Ce a ieșit | De ce |
|---|---|---|
| `loo_shared.py` | toate scorurile puternic negative | poza a fost curățată cu o amprentă care le conține pe celelalte |
| `loo_shared2.py` | toate scorurile pozitive | o amprentă din circa 19 poze e prea zgomotoasă și lasă rest |
| `loo_shared3.py` | scoruri în jur de zero | lucrează pe reziduuri necurățate, fără eroare sistematică |

- Testul T1 din `loo_shared3.py` compară fiecare poză cu media pozelor aceluiași fotograf care au orientarea opusă. Media pe fotograf e între -0.4 și 1.2. Toorisugari are 0.61, yakin 0.59.
- Testul T2 compară fiecare poză cu media tuturor celorlalți fotografi. Mediile sunt între -0.5 și 1.0. Maximul e 5.1, la o poză rojiura.
- Scorurile T1 ale lui Toorisugari, în ordinea postării, arată a zgomot. Nu e nimic de citit. Niciun al doilea senzor nu se vede, la nimeni.
- O singură poză e ciudată, găsită de `fresh_look/dualscan.py`. Poza semnată `w02_015` a lui amayadori poartă amprenta lui, cu z 11.2. Poartă și amprenta lui Toorisugari rotită, cu z 6.8 și 9.1 în două amprente independente. Asta înseamnă cam 2.8 și 3.7 abateri standard. E singura de felul ăsta, deci nu e un cod. Nu are explicație.

### 3. În pixeli și în fișier

- Metadate și date după sfârșitul fișierului: nu există.
- Rânduri ascunse sub poză: `hidden_rows.py` mărește înălțimea din antet și decodează din nou. Nu apar rânduri cu conținut.
- Structura fișierului: `fresh_look/markers.py` arată că toate cele 253 de fișiere au aceeași alcătuire. Tabelele de cuantizare sunt două, cu sumele 1109 și 1666.
- Benzile negre sunt exact 0. Nicio scenă nu apare de două ori, nici rotită, nici oglindită. Asta a verificat `fresh_look/dup.py`.

**Steganografie JPEG.** Un JPEG ține poza ca numere întregi, numite coeficienți. Programele clasice ascund biți în ultimul bit al acestor numere. Asta strică anumite rapoarte dintre câte valori de fiecare fel există. `dct_stats.py` a verificat pe coeficienți refăcuți din pixeli. `fresh_look/dct1.py` și `fresh_look/lsb.py` au citit coeficienții adevărați, pe toate pozele.

| Măsura | Toorisugari | Ceilalți | Ce ar arăta un mesaj |
|---|---|---|---|
| câte valori 2 față de câte valori 3 | 1.83 | 1.83-1.86 | raport tras spre 1 (JSteg, OutGuess) |
| câte valori 2 față de câte valori 1 | 0.350 | 0.348-0.351 | raport schimbat (F5) |
| ponderea ultimului bit, pe coeficienții nenuli | 0.687 | 0.687-0.692 | abatere față de ceilalți |
| șiruri citibile de cel puțin 10 caractere | 12.5 pe poză | 12-14 | cuvinte cu sens |

Șirurile citibile sunt toate fără sens. Nu apare nici `SIGMOID`, nici `FLAG`.

### 4. În subtitrări

- `caption_ocr.py` a citit automat subtitrările tuturor celor 253 de poze, cu manga-ocr. Rezultatul e `out/captions_ocr.json`. `caption_report.py` listează subtitrările lui Toorisugari în ordinea postării, în `out/captions_toorisugari.txt`.
- Numărul de rânduri e împrăștiat la el ca la ceilalți. Tabelul e `out/caption_linecounts.csv`.
- Primele caractere, în ordinea postării, încep cu 秋遠最雨雨開冬雨空深誰夜夜誰息鏡遠答. Nu formează un acrostih.
- Există 87 de fragmente citite la fel în cel puțin 4 poze. Niciunul nu apare doar în cele 63 de poze ale lui. La grupuri de 63 de poze alese la întâmplare iese 0.13 în medie.
- `caption_layout.py` și `caption_lines.py` măsoară poziția benzii și numărul de rânduri. `caption_sheet.py` pune subtitrările una lângă alta, pentru citit cu ochiul. DETALII.md trage linie: subtitrările sunt decor.

### 5. În ordinea și orientarea pozelor

Cele 45 de poze dau 45 de biți: 1 dacă poza e rotită, 0 dacă nu. Pe săptămâni, în ordinea postării:

| Săptămâna | Biții celor 9 poze |
|---|---|
| 8 | 1 0 1 1 0 1 0 0 1 |
| 9 | 0 0 0 1 1 1 0 1 0 |
| 10 | 1 1 1 1 1 0 1 1 1 |
| 11 | 1 0 0 0 0 0 0 0 0 |
| 12 | 1 1 0 0 1 0 0 0 0 |

- `decode.py` citește biții ca litere pe 5, 7 și 8 biți, direct și inversat. Niciun text.
- `rotation_columns.py` citește tabelul pe coloane, câte 5 biți. Cea mai bună citire e `WET?MXDLT`. Coloana 4 dă 28, care nu e literă. Trei litere care formează un cuvânt apar ușor din întâmplare.
- Ceilalți fotografi rotesc pozele la fel de neregulat.
- Pozițiile pozelor lui în săptămână, ca tabel de 5 x 29 biți: niciun text. Tabelul îl afișează `grid.py`.
- Șirul fotografilor din fiecare săptămână, ca inițiale sau ca cifre 0-5: nimic citibil. La fel distanțele dintre pozele lui.

### 6. Cele 25 de poze în plus

Toorisugari are 9 poze pe săptămână față de 4. S-a căutat ceva care să despartă cele 9 în 4 obișnuite și 5 speciale.

- `fresh_look/feat1.py`, `fresh_look/feat2.py` și `fresh_look/warm.py` măsoară fiecare poză. Iau mărimea fișierului, culoarea și luminozitatea. Iau și zgomotul, granulația și banda de subtitrare.
- Pentru fiecare măsură s-a căutat o ruptură în două grupuri. La mărimea fișierului scorul lui e 3.3, al celorlalți 2.8-5.0. La diferența roșu minus albastru e 3.7 față de 2.6-4.4.
- Iese în evidență doar la nivelul de zgomot: 6.2 față de 2.6-4.2. Cauza sunt trei poze de noapte, `w10_027`, `w12_002` și `w12_005`. Nu sunt cinci pe săptămână. Cele 9 poze pe săptămână sunt un singur set omogen.

### 7. În scenă, în granulație, în culoare

- **Scena.** Cele 45 de poze au fost privite cu ochiul la 640 x 480, pe planșe făcute de `fresh_look/tw.py`. S-a întins și contrastul în zonele întunecate și luminoase. Sunt poze obișnuite de stradă, din țări diferite, fără temă pe săptămână. Textul din ele e reclamă obișnuită, de exemplu POKKA, KOREAN BBQ și Coca-Cola. Ceilalți au același fel de text.
- **Granulația.** E aproape aceeași pe roșu, verde și albastru, cu corelația 0.996 în zonele netede. Tăria ei e de 7-9 niveluri de gri, la toți șase. `fresh_look/grain.py` și `grain_maps.py` desenează tăria ei locală. Se văd doar margini de scene. Mediana lui e 8.99, la ceilalți 7.4-8.7.
- **Culoarea.** E prelucrată la fel la toți. În zonele luminoase diferența roșu minus albastru e împinsă cu circa +20. În umbre albastrul e tăiat la 0. Pe fiecare poză urmează scena și nu are două trepte. Textul subtitrării e gri neutru, între 231 și 250. Deci subtitrarea a fost desenată după prelucrarea culorii.

### Piste rămase

DETALII.md spune că niciuna nu e puternică. Netestate au rămas:

1. Citirea semnelor mici din poze cu un model care citește imagini. Qwen2.5-VL nu a fost rulat, pentru că placa video comună avea liberi doar 3.5-5.6 GB.
2. Un tipar ascuns pus la altă deplasare în fiecare poză. Testul pe perechi nu are putere pentru așa ceva.
3. Un tipar care nu seamănă cu zgomotul de senzor și stă în alt cadru decât cel postat sau al senzorului.
4. Scări în afara intervalului 0.90-1.10.

README adaugă că împărțirea 9 față de 4 e singurul lucru pus clar în date și încă nefolosit.

## Cât de bine merge

| Măsura | Valoare |
|---|---|
| poze semnate nimerite la testul cu poza scoasă | 105 din 108 |
| toorisugari, jihanki, rojiura, shuuden, yakin | 18 din 18 fiecare |
| amayadori | 15 din 18 |
| z median al camerei alese | 16.7 |
| poze nesemnate cu margine peste 5 | 133 din 145 |
| poze Toorisugari cu margine peste 5 | 43 din 45 |
| cuvântul ascuns | negăsit, deci fără punctaj |

Din cele 12 poze nesemnate cu margine sub 5, 8 sunt la amayadori. Tot ale lui sunt cele trei poze semnate greșite:

| Poza | Unde a ajuns | Amănunt |
|---|---|---|
| `w01_015` | rojiura | z 7.80 față de 5.84 pentru amayadori |
| `w02_015` | toorisugari, rotită | z 12.07 față de 10.44. E poza ciudată din secțiunea 2 |
| `w06_009` | rojiura | scorurile ies de ordinul miilor. E una dintre cele cinci poze întunecate |

## Cum a fost verificată

1. **Testul cu poza scoasă**, pe cele 108 poze semnate: 105 corecte.
2. **Numerele rotunde.** Metoda nu știa câte poze are fiecare. Totuși Toorisugari a ieșit cu exact 9 pe săptămână înainte de cotă.
3. **A doua atribuire, independentă.** Un coleg a refăcut atribuirea separat. A ieșit același fotograf la toate cele 145 de poze. Diferă doar semnul de rotire la `w08_007` și `w10_013`.
4. **Orientarea.** `orient_check.py` dă PCE 1542 între grupul „necunoscut” și amprenta lui Toorisugari rotită. `orient4.py` arată că oglindirea nu e necesară.
5. **Confuzia dintre yakin și Toorisugari.** La 97% nu mai rămâne niciuna.
6. **Căutarea cuvântului.** Testele au avut martori: yakin, ceilalți cinci fotografi sau grupuri la întâmplare. Două teste cu defecte au fost găsite și înlocuite. S-a măsurat și cât de sensibil e testul pe perechi.
7. **Platforma.** Acolo nu s-a verificat nimic. Nu a existat ce trimite.

## Ce nu e sigur

Patru poze slabe, toate la amayadori:

| Poza | z | Margine | Cum a ajuns la amayadori |
|---|---|---|---|
| `w12_016` | 2.68 | 0.89 | alegere liberă |
| `w10_013` | 2.97 | 0.83 | alegere liberă |
| `w08_007` | 3.09 | -0.10 | cotă |
| `w11_008` | 4.85 | -0.27 | cotă |

1. **Cele patru poze.** Niciuna nu e dată lui Toorisugari, deci lista lui de 45 nu depinde de ele. README spune că `w12_016` și `w10_013` au fost puse la amayadori ca să iasă 4 pe săptămână. Tabelul arată că pozele mutate de cotă sunt celelalte două. Aici am urmat tabelul.
2. **Semnul de rotire la două poze.** La `w08_007` scorul lui amayadori e 3.09 nerotit și 3.03 rotit. La `w10_013` e 2.79 nerotit și 2.97 rotit. Diferențele sunt prea mici ca să decidă. Exact aici diferă și a doua atribuire.
3. **Două poze mai slabe ale lui Toorisugari.** `w10_011` are z 7.65 și margine 2.27. `w12_005` are z 6.39 și margine 4.02. Restul de 43 au margine peste 5.
4. **Cota nu vine din enunț.** Numerele 9 și 4 au fost văzute în rezultat și apoi impuse. Dacă regula reală e alta, cele două poze mutate pot fi greșite.
5. **Rafinarea s-a oprit la limită.** Bucla a făcut toate cele 7 treceri. Între ultimele două numărătoarea încă se schimba cu o poză, între jihanki și shuuden. Numărul lui Toorisugari a rămas 45 de la trecerea 3.
6. **Cifrele pașilor 1-4.** Procentele 56, 72, 79 și 86 apar doar în DETALII.md. Jurnal de rulare există numai pentru pasul final.
7. **Unde e cuvântul.** Nu se știe dacă e ascuns în zgomot, în pixeli sau în altceva. Nu se știe nici ce înseamnă „identitatea unui fotograf necunoscut” din indiciu. Testul pe perechi nu are putere, deci un tipar slab pus diferit în fiecare poză ar fi scăpat. Semnele mici din poze au fost citite doar cu ochiul.
8. **Citirea automată a subtitrărilor are greșeli.** De exemplu `out/captions_toorisugari.txt` dă 0 rânduri la `w10_019`, deși DETALII.md spune 1-6 rânduri peste tot.

## Unde e fiecare lucru

Căile sunt față de folderul taskului. Sunt 62 de scripturi în folderul principal și 14 în `fresh_look/`.

| Fișier | Ce e |
|---|---|
| `attribution_table.csv` | tabelul final, 145 de rânduri |
| `attribution.csv` | rezultatul liber, cu cele 12 scoruri z pe poză |
| `credited_check.csv` | aceleași scoruri pentru cele 108 poze semnate |
| `attribute_final.log` | ce a afișat `attribute_final.py` la fiecare trecere |
| `attribution_6cam.csv`, `attribution_7cam.csv`, `attribution_credited_toorisugari.csv` | tabele din etapele vechi |
| `README.md` | descrierea scurtă, la zi |
| `DETALII.md` | descrierea lungă, mai veche, în engleză |

Cinci scripturi contează cel mai mult. `feats.py` scoate reziduul din fiecare poză. `attribute_final.py` face atribuirea cu 12 variante și rafinare. `finalize_table.py` aplică cota și scrie tabelul final. `orient_check.py` a arătat că „a șaptea cameră” e același telefon rotit. `loo_shared3.py` e singurul test pe grupuri fără eroare sistematică.

Toate scripturile din folderul principal, pe grupe:

| Grupa | La ce folosește | Scripturi |
|---|---|---|
| atribuirea finală | fac tabelul | `prnu.py`, `wavelet.py`, `feats.py`, `attribute_final.py`, `finalize_table.py` |
| explorare | compară filtre și caută unde stă zgomotul | `explore_bands.py`, `explore_gpu.py`, `explore_shift.py`, `explore_shift2.py`, `explore_spec.py` |
| etape vechi | drumul până la descoperirea rotirii | `attribute2.py`, `attribute7.py`, `gram.py`, `groups.py`, `seventh.py`, `split_check.py`, `orient_check.py`, `orient4.py` |
| amprenta desenată | caută un desen în amprentă | `render_sensor.py`, `render_fp.py`, `texture_maps.py`, `chroma_scan.py`, `line_patterns.py`, `sensor_xcorr.py`, `local_xcorr.py`, `flatfield.py` |
| tipar în cadrul postat | caută ceva adăugat după rotire | `posted_pattern.py`, `posted_additive.py`, `posted_cross.py`, `posted_render.py`, `posted_fullres.py`, `hidden_pattern.py`, `signature_scan.py`, `secret_scan.py` |
| semnate față de nesemnate | caută ceva doar în pozele nesemnate | `extra_component.py`, `k_split.py` |
| amprenta pe fiecare poză | prezență, tărie, deplasare, granulație | `pairs_after_removal.py`, `photo_presence.py`, `presence_map.py`, `prnu_gain.py`, `shift_scan.py`, `grain_maps.py` |
| senzor ascuns | perechi și grupuri, toate pozele | `clean_residuals.py`, `pce_pairs.py`, `pce_pairs_lib.py`, `pce_report.py`, `pce_scales.py`, `pce_power.py`, `loo_shared.py`, `loo_shared2.py`, `loo_shared3.py` |
| fișier și coeficienți | steganografie JPEG, rânduri ascunse | `dct_coeffs.py`, `dct_stats.py`, `hidden_rows.py` |
| subtitrări | citire automată și așezare | `caption_ocr.py`, `caption_report.py`, `caption_layout.py`, `caption_lines.py`, `caption_sheet.py` |
| ordine și orientare | citirea biților | `decode.py`, `rotation_columns.py`, `grid.py` |

Folderele:

```
fresh_look/     a doua trecere, din 14 noiembrie
   dct1.py  lsb.py  markers.py  feat1.py  feat2.py  warm.py  grain.py     măsuri pe fișier și pe poză
   gram2.py  gram3.py  presence_pool.py  lowmean.py  dualscan.py          tipare comune
   dup.py  tw.py                                                          scene
   dct1.csv  dualscan.csv  feat1.csv  feat2.csv  warm.csv                 tabelele lor
out/            181 de fișiere: imagini PNG, jurnale, tabele
cache/          fișiere intermediare mari, nu intră în repo
explore/        gol
```

În afara folderului taskului mai există `../task_japan/baseline_solution.py`. Antetul lui spune că e un punct de pornire, nu o soluție.

## Cum refaci totul

Îți trebuie datele la `../task_japan/dataset` și Python cu torch, numpy, scipy și Pillow. Îți trebuie și o placă video. Atribuirea durează cam 15 minute. Căutarea senzorului ascuns cere cam 1 GB de memorie video. Pe Windows se adaugă `-X utf8` după `python`.

```
python feats.py                               # face cache/feats.pt
python attribute_final.py                     # face attribution.csv și credited_check.csv
python finalize_table.py                      # face attribution_table.csv
python clean_residuals.py                     # de aici: căutarea senzorului ascuns
python pce_pairs.py phat
python pce_report.py 60 pce_pairs_phat.npz
python pce_scales.py
python loo_shared3.py
pip install manga-ocr                         # de aici: subtitrările
python -X utf8 caption_ocr.py
python -X utf8 caption_report.py
python decode.py                              # de aici: biții de orientare
python rotation_columns.py
```

Ce trebuie să vezi după primele trei comenzi:

1. `attribute_final.py` afișează 0.972 la ultimele treceri și tabelul de confuzie cu 15 pe primul rând.
2. `finalize_table.py` afișează cele două mutări, `w08_007` și `w11_008`.
3. Numărătoarea finală e toorisugari 45 și câte 20 la ceilalți.

De știut înainte să rulezi restul:

- `clean_residuals.py` are nevoie de `attribution_table.csv` și `credited_check.csv`.
- `caption_ocr.py` și `caption_report.py` au căi absolute scrise pentru calculatorul pe care s-a lucrat. La fel scripturile din `fresh_look/`.
- `decode.py` citește `attribution.csv`. `rotation_columns.py` citește `attribution_table.csv` din folderul curent, deci se rulează din folderul taskului.
- `fresh_look/dct1.py` și `fresh_look/lsb.py` încarcă biblioteca jpeglib dintr-un folder temporar. Pe alt calculator biblioteca trebuie instalată și calea schimbată.
- `fresh_look/gram3.py` citește `gram2.npz`, scris de `fresh_look/gram2.py`. Fișierul nu e păstrat în folder.
- Cele mai multe scripturi de zgomot cer `cache/feats.pt`, făcut de `feats.py`. `k_split.py` și scripturile `pce_*.py` și `loo_shared*.py` cer `cache/clean_res.pt`, făcut de `clean_residuals.py`. `posted_fullres.py` cere `cache/sensor_toorisugari.pt`, făcut de `render_sensor.py`.
