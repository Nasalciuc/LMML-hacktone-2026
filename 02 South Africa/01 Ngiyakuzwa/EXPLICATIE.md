# Africa de Sud: Ngiyakuzwa? Explicația completă

Documentul explică taskul și rezolvarea lui pentru cineva care nu a lucrat la el. Căile de fișiere sunt relative la folderul taskului, `02 South Africa/01 Ngiyakuzwa`. Numerele au punct zecimal, la fel ca în fișiere.

## Ce se pune pe site

- Răspunsul: flag-ul `SIGMOID_ULWAZI`.
- Linkul: adresa GitHub a descrierii soluției. Adresa exactă nu apare în surse.
- *Ulwazi* înseamnă „cunoaștere, informație” în isiZulu (limba zulu).
- Forma cerută e `SIGMOID_<CUVÂNT>`. Cuvântul se scrie în ortografia standard isiZulu, cu majuscule. Regula e notată în `DETALII.md`.
- Primul răspuns la care s-a ajuns a fost ISISHO. Era greșit. Povestea lui e în secțiunea „Ce am încercat, în ordine”.
- Nu se știe dacă platforma a acceptat flag-ul. Nicio sursă nu spune asta.
- `DETALII.md` notează că taskul a fost făcut pentru exersare, nu pentru punctaj.

## Ce cere taskul

Primim un mesaj vocal: `../task_south_african_republic/voice_note.wav`. Are 10 minute și jumătate. E înregistrat la 16 kHz, pe un singur canal. O femeie citește propoziții scurte în isiZulu pe un stadion foarte zgomotos. Undeva spune cuvântul *ikhodi*, care înseamnă „cod”. Cuvântul de imediat după *ikhodi* este flag-ul.

```
voice_note.wav  ->  unde e "ikhodi"?  ->  cuvântul de imediat după  ->  SIGMOID_<CUVÂNT>
```

Enunțul original de pe site nu e salvat în folder. Ce știm despre cerință vine din `README.md` și din `DETALII.md`.

Un model de recunoaștere a vorbirii este un program care primește sunet și scrie textul rostit. Mai jos îi spunem „model de recunoaștere”. Drumul direct ar fi să dai mesajul unui astfel de model și să cauți *ikhodi* în text. Drumul direct nu merge, din patru motive:

1. Zgomotul acoperă vocea. Goarnele de stadion sună exact în zona de frecvențe a vocii.
2. Limba e isiZulu. Un model general de recunoaștere nu o înțelege și scrie text inventat în engleză.
3. Răspunsul e un singur cuvânt. O literă greșită înseamnă flag greșit.
4. Mesajul are o capcană. Femeia spune de două ori *ikhono* („pricepere”). În zgomot, *ikhono* sună aproape la fel ca *ikhodi*.

## Ce am aflat din date

### Din ce e făcut sunetul

O spectrogramă e o imagine a sunetului. Pe orizontală e timpul. Pe verticală e frecvența, adică cât de gros sau de subțire e sunetul. Culoarea arată cât de tare e sunetul în acel punct. Zgomotul din mesaj are cinci părți:

| Zgomot | Ce este |
|---|---|
| Două goarne de tip vuvuzela | fiecare are un ton de bază și multiplii lui: 217.5 Hz × k și 233.1 Hz × k |
| Fluierături | tonuri fixe sau care alunecă încet |
| Vuietul mulțimii | zgomot pe toate frecvențele |
| Șuierat | o bandă între 5 și 7 kHz |
| Aplauze | pocnituri scurte |

„× k” înseamnă tonul de bază luat o dată, de două ori, de trei ori și tot așa. Goarnele pornesc și se opresc în blocuri de câteva secunde. În tot mesajul sunt 69 de astfel de comutări. Tonurile lor acoperă cea mai mare parte din zona vocii.

Vorbirea ocupă 267 de secunde din cele 10 minute și jumătate. Restul e pauză plină de zgomot.

### Din ce e făcut textul

Mesajul a fost tăiat în 138 de segmente. Un segment e o bucată de sunet care conține o propoziție citită. Mai jos sunt segmentele care contează în poveste. Timpii sunt din `segments/index.csv`, iar textele sunt cele stabilite la sfârșitul lucrului.

| Segment | Început | Sfârșit | Durată | Ce se spune acolo |
|---|---|---|---|---|
| 067 | 05:04.53 | 05:07.78 | 3.25 s | „kutshengisa ikhono elihle” (capcana) |
| 077 | 05:47.89 | 05:50.39 | 2.5 s | aceeași propoziție, a doua oară (capcana) |
| 097 | 07:14.78 | 07:20.75 | 5.97 s | „wentela uzosebenzisa ikhodi” și cuvântul-cheie |

Propozițiile citite de femeie sunt luate dintr-o colecție publică de vorbire în isiZulu, numită NCHLT. Pe Hugging Face colecția se numește `Max5ive/nchlt_speech_zulu`. Colecția are, pentru fiecare înregistrare, și textul scris. Asta schimbă problema: textul se poate căuta, nu trebuie ghicit după ureche. Sursele nu spun cum s-a observat că propozițiile vin din această colecție.

În toată colecția, *ikhodi* apare în 83 de înregistrări. Textele diferite sunt doar trei. Cifrele sunt din `DETALII.md` și se regăsesc în `work/nchlt_ikhodi_rows.json`.

| Textul din colecție | Câte înregistrări |
|---|---|
| khona ikhodi lomgwaqo | 44 |
| wentela uzosebenzisa ikhodi | 38 |
| ikhodi engagunyaziwe elimazayo | 1 |

## Ce am încercat, în ordine

Ordinea urmează `DETALII.md` și orele fișierelor. Scripturile sunt toate din 14 noiembrie. Cele patru scripturi ale soluției finale au ora 23:32 și sunt ultimele.

### 1. Curățarea zgomotului (`clean.py`)

Problema: pe sunetul brut modelele de recunoaștere nu scot nimic folositor. În `asr_ikhodi_phrase.tsv`, pe versiunea brută, un model nu scrie nimic. Altul repetă „lakwa lakwa lakwa” până la capăt.

Ce intră: mesajul brut. Ce iese: trei versiuni curățate ale întregului mesaj, în `cleaned/`.

```
voice_note.wav -> spectrogramă -> podea de zgomot + tonuri marcate -> volum pe fiecare punct -> sunet curățat
```

1. Sunetul e transformat în spectrogramă. Fereastra are 64 ms și pasul e de 8 ms.
2. Pentru fiecare frecvență se estimează podeaua de zgomot. Podeaua e percentila 25 a puterii, luată pe o fereastră de 3 secunde care alunecă în timp. Percentila 25 e valoarea sub care stă un sfert din măsurători. Ce durează mult ajunge în podea: tonurile goarnelor și vuietul. Silabele sunt scurte și se mișcă, deci nu ajung acolo.
3. Se marchează tonurile fixe. O frecvență e „vârf” dacă puterea ei e de peste 3 ori mediana vecinelor. Dacă rămâne vârf în cea mai mare parte a unui interval de 0.6 secunde, e marcată. Armonicele vocii alunecă și nu stau atât pe loc.
4. Se marchează separat cele două goarne. Când o goarnă cântă, primele ei patru armonice ies clar peste frecvențele vecine. Atunci toți multiplii tonului ei de bază sunt marcați.
5. Fiecare punct din spectrogramă primește un volum propriu, între 0 și 1. Punctele mult peste podea rămân aproape neatinse. Punctele aproape de podea sunt date încet. Punctele marcate ca ton sunt tăiate aproape de tot. Metoda se numește câștig de tip Wiener.
6. Tot ce e sub 120 Hz e șters. Ce e peste 7000 Hz e dat mai încet. La final volumul general e adus la același nivel.

| Fișier | Cum e făcut | Cum sună |
|---|---|---|
| `cleaned/voice_note_clean.wav` | setările de bază | versiunea de lucru |
| `cleaned/voice_note_clean_voiceband.wav` | versiunea de bază, păstrată doar între 250 și 3800 Hz | rămâne doar zona vocii |
| `cleaned/voice_note_clean_strong.wav` | curățare mai dură, păstrată între 200 și 5000 Hz | mai puțin zgomot, dar vocea sună „ca sub apă” |

De ce trei versiuni: fiecare strică vocea în alt fel. O citire care apare pe toate trei e mai de încredere decât una care apare pe una singură.

### 2. Tăierea în segmente (`segment.py`)

Scopul: aflăm unde se vorbește și scoatem fiecare propoziție separat.

1. Se măsoară energia sunetului curățat între 250 și 3500 Hz, din 10 în 10 ms.
2. E vorbire acolo unde energia trece cu cel puțin 6 dB peste propria podea. Decibelul (dB) e unitatea pentru tăria sunetului. Podeaua e percentila 20 pe o fereastră de 6 secunde.
3. Izbucnirile mai scurte de 60 ms sunt aruncate.
4. Bucățile aflate la mai puțin de 0.25 secunde una de alta devin un cuvânt. Cuvintele aflate la mai puțin de 0.7 secunde devin o propoziție.
5. Propozițiile mai scurte de 0.4 secunde sunt aruncate. Fiecare primește câte 0.25 secunde de margine în ambele capete.

Ce iese: 138 de segmente în `segments/`. Fiecare are un fișier WAV și o spectrogramă PNG, cu sunetul curățat sus și cel original jos. Lista cu timpii e în `segments/index.csv`.

Același script a încercat să găsească *ikhodi* direct după forma sunetului. Forma căutată începe cu „i” și cu suflul de la „kh”. Continuă cu „o”, cu închiderea scurtă de la „d” și cu încă un „i”. Clicurile făcute de goarne la pornire și la oprire au fost excluse, pentru că seamănă cu suflul „kh”. Cele mai bune 10 locuri sunt în `ikhodi_candidates/`. Pista e slabă. Niciunul dintre cele 10 locuri nu cade în segmentul 097 și niciunul în 067. Cel mai apropiat de 097 e la 07:28.13, după sfârșitul segmentului.

### 3. Prima citire: ISISHO

```
transcriere liberă pe 138 de segmente -> "ikhodo", "ikhoto" în segmentul 067 -> cuvântul de după: "isisho" -> ISISHO (greșit)
```

**Cum s-a ajuns la segmentul 067.** Toate segmentele au fost transcrise liber. „Liber” înseamnă că modelul scrie ce crede el că aude, fără niciun ajutor. Apoi s-a căutat în text ceva care seamănă cu *ikhodi*. Tabelul arată două variante ale modelului Whisper antrenate în plus pe isiZulu: `zionia/whisper-small-isizulu` și `zionia/whisper-small-isizulu-noisy`. A fost rulat și modelul Whisper general. Rezultatul lui e la punctul 4.

| Segment | `whisper-small-isizulu` | `whisper-small-isizulu-noisy` |
|---|---|---|
| 067 | kucingis abikhodo isisho | kutxingis abikhoto isisho |
| 077 | ucingisa ikholo elisi | ucingisa ikholo elesi |
| 097 | ayam u u u u u ... (buclă) | ayam kunyana njosebensisengikuthi ujwazi |

Rândurile sunt din `segments_whisper-small-isizulu.tsv` și `segments_whisper-small-isizulu-noisy.tsv`. O buclă înseamnă că modelul repetă același lucru la nesfârșit. E semn că nu aude nimic clar. Segmentul 067 avea „ikhodo” și „ikhoto”, urmate de „isisho”. A fost luat drept locul cu *ikhodi*. Segmentul 097, locul adevărat, nu semăna deloc: un model a intrat în buclă și celălalt a lipit cuvintele între ele.

**Cum a fost citită fraza din 067.** A fost luată drept „kutshengisa ikhodi X”, unde X e cuvântul căutat. Un model pe litere scrie literă cu literă, fără să știe ce cuvinte există. Modelul pe litere MMS a dat timpii. „Kutshengisa” ține de la 304.8 la 305.6 secunde. Presupusul „ikhodi” ține de la 305.7 la 306.3. Cuvântul X ține de la 306.37 la 307.0. Urmează o pauză până pe la 308.2.

Spectrograma cuvântului X e în `work/zoomword.png`. `DETALII.md` o descrie așa: o vocală între 306.4 și 306.65, apoi un șuierat puternic între 2 și 6 kHz până la 307.0. Șuieratul se potrivește cu „s” sau „sh”. Nu se potrivește cu „l”.

`asr_all.py` a transcris liber fraza cu patru modele, pe toate versiunile audio. Două sunt modelele Whisper de isiZulu. Celelalte două sunt modele pe litere: MMS și w2v-BERT (`kesbeast23/isizulu-w2v-bert-2.0`). Rezultatul e în `asr_ikhodi_phrase.tsv`. Pe cele trei versiuni curățate:

| Model | clean | strong | voiceband |
|---|---|---|---|
| wsm-isizulu | kucingisa rikhodo isisho kuhang | kucingisa ikhoto ikisho isuham | kucingisa likhodo eisisi isiham |
| wsm-isizulu-noisy | kutshingisari kodwa ufishu kuhari | kutshingisa rikhodo isishizwa | kutshingisa rikhodo esisho isiham |
| mms | kutshengisa yinkhoto enisu | kutshengisa inkhoto efushu | kutsengisa inkodo esisi |
| w2vbert | kushengiza iyikhodwa elishle | kutshengiza igkholoa eblishewahe | kutshengiza liyikhoo eishlewai |

Tabelul scurt din `DETALII.md` are litere puțin diferite la ultimele două modele (de exemplu „ekisu” la MMS). Acolo sunt citate și rulări cu `ctc_zul.py`, pe altă fereastră de timp.

**Cum a fost ales cuvântul.** Aici apare prima dată costul de potrivire. Costul răspunde la întrebarea: cât de bine se potrivește un text dat cu un sunet dat?

```
sunet + text dat  ->  model de recunoaștere  ->  cost de potrivire (un număr)
                                                 mic  = modelul ar fi scris și singur ceva apropiat
                                                 mare = textul nu se potrivește cu sunetul
```

Modelul primește sunetul și e obligat să scrie exact textul nostru, bucățică cu bucățică. La fiecare bucățică, modelul spune ce probabilitate îi dădea el. Costul adună cât de improbabile au fost bucățelele. În scripturi costul apare cu numele NLL (negative log-likelihood).

`decide.py` a calculat costul doar pentru cuvântul X:

1. Ia fereastra de la 304.4 la 307.7 secunde, pe cele trei versiuni curățate.
2. Calculează costul pentru „kutshengisa ikhodi W” și scade costul pentru „kutshengisa ikhodi”. Rămâne costul cuvântului W.
3. Împarte la numărul de litere. Altfel cuvintele scurte câștigă automat.
4. Face asta cu patru modele și 17 cuvinte candidate. Fiecare model pune cuvintele în ordine.
5. Face media locurilor din cele patru clasamente. Mai mic e mai bine.

| Cuvânt | Cuvânt real | MMS | w2v-BERT | Whisper | Whisper-noisy | Loc mediu |
|---|---|---|---|---|---|---|
| esisi | nu | 1 | 2 | 3 | 2 | 2.00 |
| isisi | nu | 3 | 6 | 1 | 1 | 2.75 |
| isisho | da | 6 | 7 | 2 | 3 | 4.50 |
| esisho | da | 4 | 3 | 11 | 7 | 6.25 |
| elihle | da | 11 | 5 | 15 | 15 | 11.50 |
| imfihlo | da | 15 | 16 | 16 | 16 | 15.75 |

Tabelul complet e în `work/decide_out.tsv`. „Isisho” era cel mai bun cuvânt real, deci răspunsul a fost ISISHO.

**De ce a fost greșit.**

- Toată notarea pornea de la ideea că în 067 se spune *ikhodi*. Ideea nu a fost verificată.
- În 067 femeia spune „kutshengisa ikhono elihle”. Cuvântul din mijloc e *ikhono*, nu *ikhodi*.
- Semnul era deja în transcrieri. Aproape toate citirile cuvântului din mijloc se termină în „o”, de exemplu rikhodo, ikhoto și inkodo. *Ikhodi* se termină în „i”.
- Clasamentul nu a dat greșeala de gol. Cuvântul adevărat, „elihle”, era printre candidați și a ieșit pe locul 13 din 17. Sursele nu explică de ce a ieșit atât de jos.

### 4. Alte drumuri închise

Cele mai multe au fost încercate ca să citească mai sigur cuvântul din 067.

**Model general pe tot mesajul.** `asr.py` a rulat modelul `openai/whisper-large-v3-turbo` pe tot mesajul. Rezultatul e în `transcript.txt`. Primele secunde sunt „Thank you.” rând după rând. Urmează „I'm not going to say anything.” de zeci de ori, apoi „Gracias.”. Nu e nimic în isiZulu. Același model, rulat pe segmente cu limba setată pe swahili (`asr_segments.py`), scrie pentru 097: „Kwa hengena, jesu sebe nesiline kutu. Mungu asa.” În așa ceva nu ai ce căuta.

**Ferestre care alunecă.** `asr_slide.py` a transcris ferestre de 12 secunde, din 4 în 4 secunde, cu cele două modele de isiZulu. Rezultatul e în `slide_clean.txt`. `DETALII.md` nu spune ce s-a tras din el atunci. Ce se vede în el acum e descris la „Cum a fost verificată”.

**Alte feluri de a transcrie fraza.** `asr_prompt.py` îi dă modelului un text de pornire care conține *ikhodi*. `asr_var.py` și `asr_var2.py` schimbă felul în care modelul își alege cuvintele. `asr_words.py` scoate timpi pe cuvinte. `asr_pick.py` încearcă alte coduri de limbă. Rezultatele lor nu sunt salvate în folder.

**Îmbunătățire neurală (`enhance.py`).** Ideea: rețele antrenate să separe vocea de zgomot. Au fost încercate patru: HTDemucs, MetricGAN+, SepFormer WHAM16k și SepFormer DNS4. Au fost aplicate pe intervalul de la 290 la 320 de secunde, pe sunetul brut și pe cel curățat. Au ieșit 10 fișiere în `work/enh/`. Două dintre ele sunt lanțuri: HTDemucs pe sunetul brut, urmat de încă o rețea. Rezultatul a fost mai rău decât fără rețele:

| Versiune | whisper-isizulu | MMS |
|---|---|---|
| clean__sepdns | kudingi samikholo isivisa kuthi | kuqengisa ikikhotho ekuzengiso |
| clean__demucs | u kusengisa dukota u | kucengesali nkono uuu |
| clean__metricgan | kuthi kuthi kuthi ... | kukhengezamnimuuakne |
| raw__demucs | kuthi angis angis ... | kutshingia |
| raw__sepdns | ucingis ... uccingis ... | a |

Pe sunetul brut, rețelele iau goarnele drept „voce”. Pe sunetul curățat, vocea iese înfundată și Whisper intră în buclă. Doar SepFormer DNS4 pe sunetul curățat rămâne cât de cât citibil.

**Căutare în tot dicționarul (`lexicon_search.py`).** Ideea: nu alegem noi candidații, încercăm toate cuvintele. `make_lexicon.py` a făcut o listă de 29.7 mii de cuvinte din Wikipedia în isiZulu. Pentru fiecare cuvânt W s-a calculat costul textului „ikhodi W” între 305.6 și 307.6 secunde. Rezultatul e în `lexicon_scores.tsv` și e inutil. Primele locuri sunt toate șiruri de două litere. Primul e „ie” cu 148.8, al doilea „ii” cu 159.7, al treilea „ni” cu 161.2. Cauza: costul adunat pe o fereastră fixă favorizează textele scurte. Mai puține litere înseamnă mai puțin de plătit. `decide.py` a reparat asta prin trei măsuri: scade costul începutului, împarte la litere și folosește doar candidați aleși.

**Notare pe toate versiunile audio (`score_words.py`).** Aceeași idee de cost, pe 26 de versiuni, inclusiv cele îmbunătățite neural. Rezultatul e în `work/score_words_out.txt`. Versiunile stricate de rețele trag clasamentul după ele. De exemplu `raw__metricgan`, `raw__sepdns` și `raw__sepwham` pun toate „ifu” pe primul loc. Pe total a ieșit „ekisu” cu 134.1, apoi „ifu” cu 137.5. „Isisho” avea 212.8. De aici s-a reținut că doar cele trei versiuni curățate sunt bune de folosit. `score_whisper.py` a făcut aceeași notare doar cu modelele Whisper. Totalurile sunt în `work/score_whisper_out.txt`. Primele trei sunt isisi cu 5.7, esisi cu 21.5 și isisho cu 30.1. Sunt costuri adunate, neîmpărțite la litere.

**Comparație cu voce sintetică (`tts_dtw.py`).** Ideea: un program rostește fiecare candidat cu voce artificială, apoi comparăm sunetul lui cu sunetul real. Comparația se face cu DTW. DTW întinde și strânge în timp două sunete ca să le suprapună, apoi măsoară cât de diferite rămân. Au fost două piedici:

- Vocea de isiZulu `facebook/mms-tts-zul` nu există pe Hugging Face. Lipsesc și vocile MMS pentru xhosa, ndebele și swati. La fel pentru sotho și tswana. `tts_zul.py` a rămas scris, dar nu poate rula.
- S-au folosit în loc o voce de xitsonga și una de swahili. Nu a ieșit niciun semnal.

| Voce | Fișier | Cea mai mică valoare | Cea mai mare valoare | Locul lui „isisho” (frază / cuvânt) |
|---|---|---|---|---|
| xitsonga | `work/tts_dtw_tso.txt` | 5.848 | 6.703 | 12 din 13 / 12 din 13 |
| swahili | `work/tts_dtw_swh.txt` | 5.923 | 6.738 | 13 din 13 / 10 din 13 |

Toți candidații stau într-un interval îngust. Ordinea diferă între cele două voci. O voce din altă limbă e prea departe de femeia din zgomot.

**Neîncercate.** DeepFilterNet și MossFormer2 nu erau instalate. Îmbunătățirea neurală arătase deja că sunetul suferă mai mult de la ce adaugă rețelele decât de la zgomotul rămas. Nici modelul `lsgmgbl/wav2vec2-xls-r-isizulu` nu a fost încercat.

### 5. Căutarea în colecția publică

Aici s-a schimbat întrebarea. Întrebarea veche era deschisă: ce cuvinte spune femeia aici? Orice șir de litere putea fi răspuns. Întrebarea nouă e închisă: care dintre trei propoziții cunoscute se aude aici? Modelul nu mai scrie singur. Dă doar un cost pentru un text primit de-a gata.

Întrebarea nouă e mult mai ușoară în zgomot. Un model care nu reușește să scrie singur propoziția poate totuși să spună care text dat se potrivește mai bine. Segmentul 097 e exemplul: la transcriere liberă a ieșit buclă, la cost a ieșit clar primul.

Sursele nu spun ce a declanșat reverificarea după ISISHO. Rezultatul ei: *ikhodi* e în segmentul 097, iar cuvântul de după e *ulwazi*. Pașii sunt în secțiunea următoare.

## Soluția finală, pas cu pas

```
voice_note.wav  --clean.py-->  cleaned/ (3 versiuni)  --segment.py-->  segments/ (138 de segmente)
colecția NCHLT (text scris)  --corpus_ikhodi.py-->  3 propoziții cu "ikhodi"

segments/ + 3 propoziții  --find_phrase.py-->  segmentul 097 = "wentela uzosebenzisa ikhodi"
segmentul 097  --next_word.py, mms_check.py-->  cuvântul de după: ulwazi  -->  SIGMOID_ULWAZI
```

**Pașii 1 și 2. Curățare și tăiere.** `clean.py` scrie cele trei versiuni din `cleaned/`. `segment.py` scrie cele 138 de segmente din `segments/`. Detaliile sunt mai sus.

**Pasul 3. Căutarea în colecție (`corpus_ikhodi.py`).**

- Ce intră: cuvântul „ikhodi”.
- Ce face: întreabă serviciul de căutare în text al Hugging Face, pentru colecția `Max5ive/nchlt_speech_zulu`. Caută în ambele părți ale colecției (train și test), câte 100 de rânduri pe pagină.
- Ce iese: `work/nchlt_ikhodi_rows.json`, cu rândurile găsite, fără sunet. Pe ecran tipărește câte înregistrări are fiecare text.
- Rezultat: 83 de înregistrări și doar trei texte, cele din tabelul de la „Ce am aflat din date”.

**Pasul 4. Care segment e una din cele trei propoziții (`find_phrase.py`).**

- Ce intră: cele 138 de fișiere WAV din `segments/` și cinci texte.
- Cele cinci texte: cele trei propoziții cu *ikhodi*, o propoziție de control și o propoziție care nu ar trebui să se potrivească cu nimic.
- Ce face: calculează costul de potrivire pentru fiecare pereche (segment, text). Folosește cele două modele Whisper de isiZulu și adună cele două costuri.
- Ce iese: `work/find_phrase_nll.json`, cu toate costurile brute. Pe ecran tipărește cele mai bune 7 segmente pentru fiecare text.

Costurile brute nu se pot compara direct între texte. Un text lung costă mai mult decât unul scurt pe orice sunet. De aceea scriptul calculează un scor z pentru fiecare segment:

```
scor z = (costul segmentului - mediana celor 138 de costuri) / abaterea standard a celor 138 de costuri
```

Mediana e valoarea din mijloc. Abaterea standard arată cât de împrăștiate sunt costurile. Scorul z spune deci cât de departe e un segment de grosul celorlalte, pentru același text. Un z de -5 înseamnă un cost cu cinci „împrăștieri” sub mijloc. Negativ înseamnă potrivire mai bună decât de obicei.

Câteva costuri brute din `work/find_phrase_nll.json`, rotunjite la o zecimală:

| Text | Segment | whisper-small-isizulu | whisper-small-isizulu-noisy |
|---|---|---|---|
| wentela uzosebenzisa ikhodi | 097 | 164.7 | 147.4 |
| wentela uzosebenzisa ikhodi | 023 | 171.1 | 158.9 |
| wentela uzosebenzisa ikhodi | 001 | 186.8 | 180.1 |
| kutshengisa ikhono elihle | 077 | 144.0 | 134.8 |
| kutshengisa ikhono elihle | 067 | 153.6 | 147.7 |
| kutshengisa ikhono elihle | 001 | 182.1 | 178.6 |

Segmentul 001 e pus doar pentru comparație. Pentru propoziția cu *ikhodi*, segmentul 097 are cel mai mic cost la ambele modele. Pentru propoziția de control, cele mai mici costuri sunt la 077 și la 067. Scorurile z sunt în secțiunea „Cât de bine merge”.

**Pasul 5. Cuvântul de după *ikhodi* (`next_word.py`).** În colecție propoziția se termină cu *ikhodi*. În mesaj, în aceeași respirație, mai urmează un cuvânt. Acel cuvânt e flag-ul.

- Ce intră: fereastra de la 434.78 la 440.75 secunde (segmentul 097), pe cele trei versiuni curățate. Plus o listă de 30 de candidați.
- Ce face: pentru fiecare cuvânt W calculează costul pentru „wentela uzosebenzisa ikhodi W” și scade costul pentru „wentela uzosebenzisa ikhodi”. Rămâne costul cuvântului W. Adună peste 2 modele × 3 versiuni.
- Ce iese: lista cuvintelor în ordinea costului, tipărită pe ecran.
- La final scriptul ia propoziția întreagă de patru cuvinte, cu *ulwazi*, și o notează pe toate cele 138 de segmente. Verifică astfel că propoziția se potrivește doar cu 097.

Cifrele sunt în secțiunea următoare. Dintre cuvintele reale încercate, *ulwazi* are cel mai mic cost.

**Pasul 6. A doua părere, de la un model pe litere (`mms_check.py`).** Modelele Whisper scriu bucăți de cuvinte și au învățat ce cuvinte sunt obișnuite în isiZulu. Pot deci să „tragă” spre un cuvânt cunoscut. Modelul MMS (`facebook/mms-1b-all`, cu adaptorul `zul`) lucrează altfel. Pentru fiecare felie mică de timp scrie o literă sau nimic. E un martor mai simplu, care scrie doar ce aude.

| Segment | Versiune | Ce scrie MMS |
|---|---|---|
| 097 | clean | wenela uzosobenziisiee nikunji uulwazu |
| 097 | strong | wenena izosebenzise ikunji undwazi |
| 077 | (nu e notată) | kutjengisa ikholo elihla |

Ultimul cuvânt din 097 iese „uulwazu” și „undwazi”. Segmentul 077 iese foarte aproape de „kutshengisa ikhono elihle”. Citirea lui 097 pe versiunea voiceband nu e notată în surse.

**Pasul 7. Alegerea scrierii.** Citirile ultimului cuvânt diferă puțin între ele. Flag-ul cere ortografia standard. *Ulwazi* e forma standard și singurul cuvânt de dicționar printre citirile de sus. De aici `SIGMOID_ULWAZI`.

## Cât de bine merge

Nu există un procent de acuratețe. Răspunsul e un singur cuvânt, iar verdictul platformei nu se știe. Ce se poate măsura e cât de clar se desparte răspunsul ales de restul.

**Cât de clar iese segmentul.** Scorurile z din `DETALII.md`:

| Propoziția căutată | Cel mai bun segment | Scor z | Următorul |
|---|---|---|---|
| kutshengisa ikhono elihle (control) | 077 | -6.4 | 067, cu -4.7 |
| wentela uzosebenzisa ikhodi + ulwazi | 097 | -5.1 | 023, cu -2.6 |
| khona ikhodi lomgwaqo | 034 | -2.7 | nicio potrivire |
| ikhodi engagunyaziwe elimazayo | 118 | -2.8 | nicio potrivire |
| ulwazi oluthile mayelana (nu e în mesaj) | 118 | -2.7 | nivel de zgomot |

- Ultimele trei rânduri arată nivelul de zgomot. Un text care nu e în mesaj tot are un „cel mai bun segment”, pe la -2.7.
- Propoziția de control ajunge la -6.4 și propoziția cu *ikhodi* la -5.1. Amândouă sunt mult sub nivelul de zgomot.
- Pentru propoziția cu *ikhodi*, al doilea segment (023) e deja la nivelul de zgomot. Există deci un singur loc bun.

**Cât de clar iese cuvântul.** Tabelul dă costul cuvântului de după *ikhodi*, din `DETALII.md`. Mai mic e mai bine. Scriptul are 30 de candidați, dar sursele dau cifre doar pentru aceștia 12.

| Cuvânt și cost | Ce este |
|---|---|
| uluwazi 41, ucwazi 54, ulwasi 57, ulazi 64 | scrieri ale aceluiași sunet, care nu sunt cuvinte |
| ulwazi 64 | cel mai bun cuvânt real |
| ngolwazi 71, uyazi 73, umlazi 76, ukwazi 78, ubuntu 87 | alți candidați |
| isisho 120 | răspunsul vechi |
| lomgwaqo 205 | ultimul cuvânt din altă propoziție cu *ikhodi* |

## Cum a fost verificată

1. **Propoziție de control.** Metoda a fost încercată pe o propoziție deja știută: „kutshengisa ikhono elihle”. A găsit-o în segmentul 077 (z = -6.4) și în 067 (z = -4.7). Deci metoda găsește o propoziție atunci când ea chiar e în mesaj.
2. **Propoziție care nu e în mesaj.** „Ulwazi oluthile mayelana” nu e citită de femeie. Metoda i-a dat cel mult -2.7, adică nivel de zgomot. Deci metoda nu „găsește” lucruri care nu sunt acolo. Propoziția asta conține chiar cuvântul *ulwazi* și tot nu se potrivește cu nimic.
3. **Celelalte două propoziții cu *ikhodi*.** Rămân la -2.7 și -2.8. Deci *ikhodi* se spune într-un singur loc.
4. **Alt fel de model.** MMS, care scrie literă cu literă, citește în 097 „uzosobenziisiee” și „uulwazu”. Două feluri diferite de modele ajung la aceeași propoziție.
5. **Explică anomalia veche.** În 067 modelele auzeau un „o” la sfârșitul cuvântului din mijloc. Cu *ikhono* în loc de *ikhodi*, „o”-ul are sens.
6. **Trei versiuni audio.** Costul cuvântului e adunat pe clean, strong și voiceband. Nu depinde de o singură curățare.

Mai există o urmă veche, care nu e trecută ca dovadă în `DETALII.md`. Fișierul `slide_clean.txt` a fost făcut înainte de căutarea în colecție. Ferestrele lui care acoperă 07:14 arată așa:

| Fereastra începe la | Model | Ce scrie în zona propoziției |
|---|---|---|
| 07:12 | noisy | wenyena josebenzisengikhundi uluyasi |
| 07:12 | izulu | uwenyena ujosebenzisengikuni ulwasu |
| 07:16 | noisy | wenyana jo semenzisi ikundi ucazu |
| 07:16 | izulu | wenyana jo semenziyisi indikhundi ucazu |

Se recunoaște ceva ca „uzosebenzisa” și, la capăt, un cuvânt de forma „ulwasu” sau „uluyasi”. Între ele modelele scriu „ikhundi” sau „ikundi”, nu „ikhodi”. De aceea căutarea în text după *ikhodi* nu a prins locul.

## Ce nu e sigur

- **Acceptarea flag-ului.** Nu se știe.
- **Urechea omului.** Au fost pregătite clipuri pentru ascultat, dar nu e notat ce s-a auzit în ele. Sursele nu pomenesc nici părerea vreunui vorbitor de isiZulu.
- **Scrierea ultimului cuvânt.** E singurul punct lăsat deschis în `DETALII.md`. Cele mai mici costuri le au scrieri care nu sunt cuvinte. *Ulwazi* e la egalitate cu „ulazi”, la 64. Nicio transcriere liberă nu dă exact „ulwazi”. MMS dă „uulwazu” și „undwazi”. Whisper dă „ujwazi” pe segment și „uluyasi”, „ulwasu”, „ucazu” pe ferestre. *Ulwazi* a fost ales pentru că e forma standard și singurul cuvânt de dicționar printre citirile de sus.
- **Forme cu prefix.** „Ngolwazi” are cost 71. Scriptul a încercat și „nolwazi”, „lolwazi”, „yolwazi”. Costurile lor nu apar în surse.
- **Cuvântul *ikhodi* nu apare literal în nicio transcriere salvată a segmentului 097.** MMS scrie „nikunji” și „ikunji”. Whisper scrie „ikhundi”. Locul a fost stabilit prin costul întregii propoziții, nu prin auzirea cuvântului.
- **Scorul z de -5.1.** În tabel e trecut pentru propoziția cu *ulwazi* adăugat. Scorul pentru propoziția de trei cuvinte, fără *ulwazi*, nu e scris în surse.
- **Ieșirile scripturilor finale nu sunt salvate ca text.** Scorurile z și costurile cuvintelor există doar în `DETALII.md`. Pentru `find_phrase.py` există costurile brute în JSON. Pentru `next_word.py` și `mms_check.py` nu există niciun fișier de ieșire.
- **De unde se știe că propozițiile vin din NCHLT.** Sursele afirmă asta, dar nu arată cum s-a aflat.
- **Vocea din mesaj.** `corpus_ikhodi.py` listează și înregistrările cu voce de femeie din colecție. Sursele nu spun ce s-a făcut cu lista. Nu se știe dacă vocea din mesaj e chiar o înregistrare din colecție.
- **Enunțul original.** Nu e în folder. Regula cu ortografia standard și majusculele e luată din `DETALII.md`.
- **Jumătatea de jos din `DETALII.md` e depășită.** De la titlul „Earlier reading (superseded)” în jos, textul spune că *ikhodi* e în segmentul 067. Tot acolo `decide.py` e numit „final ranking”. Partea de sus a fișierului și `README.md` sunt mai noi și spun 097. Aici am urmat varianta nouă.
- **`work/makeclips.py` e rămas în urmă.** Scriptul taie clipuri între 305.55 și 307.20 secunde, adică din zona segmentului 067. Clipurile noi de la 07:14 există pe disc. Scriptul care le-a făcut nu e în folder.
- **O diferență mică de cifre.** `DETALII.md` spune că valorile DTW stau între 5.85 și 6.70. În `work/tts_dtw_swh.txt` cea mai mare valoare e 6.738.

## Unde e fiecare lucru

| Scop | Fișier sau folder | La ce e bun |
|---|---|---|
| Documente | `README.md` | descrierea scurtă, la zi |
| Documente | `DETALII.md` | descrierea lungă, în engleză; sus e la zi, jos e prima încercare |
| Documente | `EXPLICATIE.md` | documentul de față |
| Intrare | `../task_south_african_republic/voice_note.wav` | mesajul vocal brut |
| Soluția finală | `clean.py` | curăță zgomotul; scrie cele trei versiuni din `cleaned/` și `work/noise_profile.npz` |
| Soluția finală | `segment.py` | taie mesajul; scrie `segments/` (138 WAV, 138 PNG, `index.csv`) și `ikhodi_candidates/` |
| Soluția finală | `corpus_ikhodi.py` | caută *ikhodi* în colecția NCHLT; scrie `work/nchlt_ikhodi_rows.json` |
| Soluția finală | `find_phrase.py` | costul fiecărei propoziții pe fiecare segment; scrie `work/find_phrase_nll.json` |
| Soluția finală | `next_word.py` | compară cuvintele posibile de după *ikhodi* în segmentul 097 |
| Soluția finală | `mms_check.py` | citește literă cu literă segmentele 097, 067 și 077 |
| Transcriere liberă | `asr.py` | Whisper general pe tot mesajul; scrie `transcript.txt` și `transcript.json` |
| Transcriere liberă | `asr_segments.py` | Whisper general pe segmente; scrie `segments_sw.tsv` |
| Transcriere liberă | `asr_pick.py` | Whisper general pe segmente alese, cu mai multe coduri de limbă |
| Transcriere liberă | `asr_zulu.py` | modelele Whisper de isiZulu pe segmente; scrie `segments_whisper-small-isizulu.tsv` și `segments_whisper-small-isizulu-noisy.tsv` |
| Transcriere liberă | `asr_slide.py` | aceleași modele pe ferestre care alunecă; scrie `slide_clean.txt` |
| Transcriere liberă | `asr_prompt.py` | transcriere cu text de pornire care conține *ikhodi* |
| Transcriere liberă | `asr_var.py`, `asr_var2.py` | variante de decodare pe un interval dat |
| Transcriere liberă | `asr_words.py` | timpi pe cuvinte de la modelele Whisper de isiZulu |
| Transcriere liberă | `asr_mms.py` | MMS pe segmente și pe fraza de la 05:04 |
| Transcriere liberă | `asr_all.py` | patru modele pe toate versiunile audio ale frazei din 067; scrie `asr_ikhodi_phrase.tsv` |
| Transcriere liberă | `ctc_zul.py` | unealta comună pentru modelele pe litere: litere cu timpi și cost |
| Notare pentru 067 | `decide.py` | clasamentul din care a ieșit ISISHO; scrie `work/decide_out.tsv` și `work/decide_log.txt` |
| Notare pentru 067 | `score_cand.py` | prima notare, cu fraze întregi date modelelor Whisper |
| Notare pentru 067 | `score_ctc.py` | cost cu modelele pe litere pentru o listă mare de candidați |
| Notare pentru 067 | `score_whisper.py` | același lucru cu modelele Whisper; ieșirea e `work/score_whisper_out.txt` |
| Notare pentru 067 | `score_words.py` | cost pe 26 de versiuni audio; ieșirea e `work/score_words_out.txt` |
| Notare pentru 067 | `make_lexicon.py` | lista de cuvinte din Wikipedia în isiZulu; scrie `work/zulu_words.tsv` |
| Notare pentru 067 | `lexicon_search.py` | căutarea în toată lista; scrie `lexicon_scores.tsv` |
| Rețele și voce sintetică | `enhance.py` | aplică cele patru rețele; scrie cele 10 fișiere din `work/enh/` |
| Rețele și voce sintetică | `work/sb_metricgan/`, `work/sb_sepdns/`, `work/sb_sepwham/` | modelele SpeechBrain descărcate de `enhance.py` |
| Rețele și voce sintetică | `work/enh_pad_*.wav` | patru fișiere audio; rostul lor nu e descris în surse |
| Rețele și voce sintetică | `tts_zul.py` | voce sintetică de isiZulu; nu poate rula, modelul nu există |
| Rețele și voce sintetică | `tts_dtw.py` | comparația cu voce sintetică; ieșirile sunt `work/tts_dtw_tso.txt` și `work/tts_dtw_swh.txt` |
| Imagini și unelte | `work/viz.py`, `work/viz2.py`, `work/spec_zoom.py` | desenează spectrograme pe un interval dat |
| Imagini și unelte | `work/zoomword.py` | desenează `work/zoomword.png`, cuvântul de la 306.37 |
| Imagini și unelte | `work/z067.png`, `work/z067w.png`, `work/z067_enh.png` | spectrograme ale frazei din 067 |
| Imagini și unelte | `work/overview.png`, `work/cmp_0_30.png`, `work/cmp_z.png`, `work/z2.png`, `work/z3.png`, `work/z4.png` | alte imagini; nu sunt descrise în surse |
| Imagini și unelte | `work/horn_fit.py` | măsoară cât de regulate sunt goarnele |
| Imagini și unelte | `work/makeclips.py` | taie clipuri pentru ascultat (varianta veche, vezi „Ce nu e sigur”) |
| Imagini și unelte | `work/decide_err.txt`, `work/tts_dtw_tso_err.txt`, `work/tts_dtw_swh_err.txt` | ieșirile de eroare ale rulărilor; nu au fost citite pentru acest document |
| Mediu | `.venv/` | mediul Python pentru `clean.py` și `segment.py` |

Clipurile pentru ascultat sunt în afara folderului: `C:\Users\user\Downloads\LMML_upload\South Africa\asculta\`. Fraza întreagă e în `IKHODI_phrase_07m14-07m21_clean.wav`. Cuvântul de după *ikhodi* e în `NEXT_WORD_07m19-07m21_clean.wav`. Fiecare are și o variantă „strong” și variante încetinite la 0.75x. Clipurile nu sunt în repo.

## Cum refaci totul

Ce îți trebuie:

- Mesajul brut la `../task_south_african_republic/voice_note.wav`.
- Mediul `.venv` din folder, pentru curățare și tăiere.
- Un Python cu torch pe placă video și cu transformers. În `DETALII.md` e `C:\Users\user\miniconda3\python.exe`, scris mai jos `py`.
- O placă video NVIDIA. Scripturile cer explicit `cuda`. După `DETALII.md`, memoria folosită rămâne sub circa 6 GB.
- Internet. Modelele se descarcă de pe Hugging Face la prima rulare, iar `corpus_ikhodi.py` întreabă un serviciu online.

Toate comenzile se dau din folderul taskului, pentru că scripturile folosesc căi relative. Soluția finală, cu rezultatul așteptat după surse:

```
.venv\Scripts\python clean.py      # scrie cleaned/ (circa 3 minute)
.venv\Scripts\python segment.py    # scrie segments/; 138 de propoziții, 267 s de vorbire, 69 de comutări
py corpus_ikhodi.py                # scrie work/nchlt_ikhodi_rows.json; 83 de rânduri: 44, 38 și 1
py find_phrase.py                  # scrie work/find_phrase_nll.json; control: 077 primul; ikhodi: 097 primul
py next_word.py                    # ulwazi în jur de 64, isisho în jur de 120; la final 097 cu z = -5.1
py mms_check.py                    # 097 pe clean: wenela uzosobenziisiee nikunji uulwazu
```

Primele două comenzi sunt din `DETALII.md`. Comenzile pentru cele patru scripturi noi nu sunt scrise acolo. Scripturile nu cer argumente, deci se rulează simplu, ca mai sus. Durata lor nu e notată.

De știut înainte să rulezi:

- `segment.py` șterge și reface folderele `segments/` și `ikhodi_candidates/`.
- `find_phrase.py`, `next_word.py` și `mms_check.py` tipăresc rezultatele doar pe ecran. Trimite ieșirea într-un fișier text dacă vrei să păstrezi cifrele.
- `corpus_ikhodi.py` folosește doar biblioteca standard din Python. Merge cu orice Python.

Prima încercare se poate reface cu comenzile din `DETALII.md`. Ele duc la ISISHO și nu sunt necesare pentru flag-ul final:

```
py enhance.py                 # scrie work/enh/ (cere demucs și speechbrain)
py asr_all.py                 # scrie asr_ikhodi_phrase.tsv
py ctc_zul.py mms 304.0 308.2 cleaned/voice_note_clean_voiceband.wav
py ctc_zul.py w2vbert 304.0 308.2 cleaned/voice_note_clean.wav
py decide.py                  # scrie work/decide_out.tsv (circa 3 minute)
py score_whisper.py cleaned/voice_note_clean.wav:0 cleaned/voice_note_clean_strong.wav:0 cleaned/voice_note_clean_voiceband.wav:0
py tts_dtw.py facebook/mms-tts-tso
py work/zoomword.py           # scrie work/zoomword.png
py work/makeclips.py          # scrie clipuri în folderul din Downloads
```
