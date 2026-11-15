# Africa de Sud: Ngiyakuzwa?

**Flag: `SIGMOID_ULWAZI`**

*Ulwazi* înseamnă „cunoaștere, informație” în isiZulu.

## Ce cere taskul

Avem un mesaj vocal de 10 minute și jumătate. O femeie citește propoziții scurte în isiZulu pe un stadion foarte zgomotos, cu vuvuzele. Undeva spune cuvântul *ikhodi* („cod”). Cuvântul de imediat după el e flag-ul.

## Ideea

Nu am încercat să înțelegem tot mesajul. Am aflat de unde vin propozițiile și am căutat doar pe cele care conțin *ikhodi*.

Propozițiile citite de femeie sunt luate dintr-o colecție publică de vorbire în isiZulu (NCHLT). Colecția are și textul scris. Deci textul se poate căuta, nu trebuie ghicit după ureche.

## Pașii

```
mesaj -> 138 de propoziții -> care dintre ele e una din cele 3 propoziții cu "ikhodi"? -> propoziția 097 -> cuvântul de după
```

1. Mesajul a fost curățat de zgomot și tăiat în 138 de propoziții.
2. În colecția publică, *ikhodi* apare în doar trei propoziții diferite.
3. Două modele care recunosc vorbirea în isiZulu au dat fiecărei perechi (propoziție scrisă, bucată de sunet) o notă: cât de bine se potrivesc.
4. Propoziția „wentela uzosebenzisa ikhodi” se potrivește clar cu bucata 097, între 07:14,78 și 07:20,75. Celelalte două propoziții nu se potrivesc cu nimic.
5. În aceeași respirație, după *ikhodi*, urmează încă un cuvânt. Dintre cuvintele reale încercate, *ulwazi* se potrivește cel mai bine.

Ca să fim siguri că metoda merge, am încercat-o pe o propoziție pe care o știam deja. A găsit-o corect.

## Prima greșeală

Primul răspuns a fost ISISHO și era greșit. Venea din bucata 067, unde femeia spune *ikhono* („pricepere”), nu *ikhodi*. Cele două cuvinte sună aproape la fel în zgomot. Bucata aceea e o capcană.

## Cât de sigur e

Sigur: locul. Propoziția cu *ikhodi* e la 07:14.

Mai puțin sigur: scrierea exactă a ultimului cuvânt. Modelele au auzit variante ca „uluwazi”, „ulwasi”, „ulazi”. Niciuna nu e cuvânt real. *Ulwazi* e forma standard și singurul cuvânt din dicționar printre ele.

## Fișiere

- `FLAG.txt`: flag-ul.
- `IKHODI_phrase_07m14-07m21_clean.wav`: propoziția întreagă, curățată.
- `NEXT_WORD_07m19-07m21_clean.wav`: doar cuvântul de după *ikhodi*.

Alte variante audio, inclusiv încetinite, sunt în `C:\Users\user\Downloads\LMML_upload\South Africa\asculta\`.

## Repo

Soluția e și în repo-ul local, în `02 South Africa\01 Ngiyakuzwa`, cu același README și cu descrierea lungă în `DETALII.md`. Nu e comisă și nu e urcată pe GitHub.
