# Kazahstan: Stamp of Disapproval

**Flag: `SIGMOID_SARSA`**

## Ce cere taskul

Sunt 5 albume de timbre, fiecare cu 26 de pagini. Pe o pagină sunt cam 3 timbre. Unul singur a fost falsificat, într-un loc mic: o cifră schimbată, un element copiat sau o inscripție ștearsă.

Dacă marchezi locul falsificat de pe fiecare pagină a unui album pe aceeași foaie, cele 26 de semne desenează o literă. Cinci albume dau cinci litere, adică cuvântul din flag.

Ca exemple avem 30 de timbre falsificate la care se știe exact locul.

## Ideea

O poză JPEG e salvată în pătrățele de 8×8 pixeli. Numerele din fiecare pătrățel sunt rotunjite la niște trepte fixe. Treptele depind de calitatea la care a fost salvată poza.

Fiecare timbru a fost salvat ca JPEG o dată, înainte să fie falsificat. Tot ce e neatins pe timbru stă încă pe treptele acelei prime salvări. Locul falsificat a fost desenat după aceea, deci nu mai stă pe trepte. Acolo e falsul.

Metoda nu are nevoie de o copie curată a timbrului.

## Pașii

```
pagină -> timbre -> prima salvare JPEG a fiecărui timbru -> locul care nu stă pe trepte -> un semn pe pagină
26 de semne pe aceeași foaie -> o literă
5 albume -> S A R S A
```

1. Pagina e tăiată în timbre (`solutie/code/segment.py`).
2. Pentru fiecare timbru se află unde încep pătrățelele și la ce calitate a fost prima salvare (`dq.py`). Calitatea iese între 82 și 92.
3. Fiecare pătrățel primește un scor: cât de bine stă pe trepte. Scor negativ înseamnă că nu stă.
4. Se caută fereastra de 33×33 pixeli cu scorul cel mai negativ. Dacă e sub −50, acela e locul. Așa se hotărăsc 58 de pagini.
5. Pe celelalte 72 de pagini urma e prea slabă, de exemplu când falsul e pe hârtie complet albă. Acolo hotărăște un clasificator antrenat pe cele 30 de exemple. El se uită și la zgomotul pozei: zonele umplute sunt prea netede sau au zgomot artificial.
6. Cele 26 de locuri ale unui album se desenează pe o foaie și se citește litera.

## Ce am verificat pe datele noastre

| Verificare | Rezultat |
|---|---|
| Regula de la pasul 4, pe cele 30 de exemple | se declanșează la 13 și nimerește locul adevărat la toate 13 |
| Scorul de la pasul 4, recalculat pe toate cele 130 de pagini | același cu cel din tabelul soluției la 130 din 130 |
| Cele 58 de pagini hotărâte de pasul 4 | locul găsit de mine cade în zona din soluție la 58 din 58 |
| Celelalte timbre de pe aceleași 58 de pagini | nu arată nimic (scor în jur de −3, față de −120 la timbrul fals) |
| Comparație cu metoda noastră veche, pe 42 de pagini unde citisem o valoare imposibilă (202, 155) | același timbru la 34 din 42; la întâmplare ar fi ieșit cam 14 |
| Litere potrivite cu alte fonturi și alt scor decât în soluție | S, A, R sau Q, S, A |

Poza `verificare_litere.png` arată punctele fiecărui album peste litera potrivită. Negru înseamnă pagină hotărâtă de pasul 4, gri de clasificator. Cercurile roșii sunt punctele metodei noastre vechi.

## Ce nu e sigur

- Cele 72 de pagini hotărâte de clasificator nu au fost rerulate de mine. Antrenarea durează cam 40 de minute. Autorii raportează 25 din 30 la testul în care se scoate pe rând câte un exemplu.
- Literele albumelor 2 și 3 (A și R) se citesc doar cu toate cele 26 de puncte. Doar din punctele verificate de mine ies sigur S, S și A, la albumele 1, 4 și 5.
- La albumul 3, R și Q ies la egalitate. Cu Q nu se formează niciun cuvânt. SARSA e numele unui algoritm de învățare prin recompensă, deci se potrivește cu celelalte flag-uri.

## Fișiere

- `FLAG.txt`: flag-ul.
- `solutie/code/`: codul. `solutie/outputs/boxes.csv`: locul găsit pe fiecare pagină. `solutie/outputs/sheets/`: foile cu litere.
- `solutie/README_original_en.md`: descrierea autorilor, în engleză.
- `verificare_litere.png`: verificarea mea.

## Cum rulezi

```
pip install numpy opencv-python scikit-learn scikit-image scipy pandas Pillow
python solutie/code/run_all.py --data <folderul task_kazakhstan> --work ./work --loo
```

Durează cam 30 de minute pentru trăsături și 10 pentru antrenare, pe 2 nuclee.

## Repo

Soluția e și în repo-ul local, în `31 Kazakhstan\01 Stamp of Disapproval`, cu același README și cu descrierea lungă în `DETALII.md`. Nu e comisă și nu e urcată pe GitHub.
