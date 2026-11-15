# Cuba: If you know you know

**Ce trimiți: o arhivă zip cu `main.py` și folderul `index/`.** Arhiva gata făcută se numește `cuba_v4.zip`. Nu e în repo, pentru că fișierele zip sunt excluse.

Dacă platforma spune că nu poate importa `onnxruntime`, se trimite rezerva `cuba_v3.zip` (versiunea 3 din tabelul de mai jos). Are nevoie doar de numpy.

## Ce cere taskul

Avem o bibliotecă de 1.552 de documente despre Cuba: articole Wikipedia, capitole de cărți, rapoarte. Primești un bilet care descrie un capitol („partea în care...”). Trebuie să dai cele 3 documente cele mai probabile.

Treci dacă documentul corect e printre cele 3 la cel puțin 80% din întrebările ascunse. Mai sunt trei limite: totul are cel mult 53 MB, o căutare durează cel mult 75 de milisecunde pe un singur nucleu, iar încărcarea cel mult 10 secunde.

## De ce a picat prima versiune

Prima versiune a luat 0,793 pe platformă, deși local lua 0,873. Toate greșelile erau la capitole de cărți. La întrebările despre Wikipedia răspundea corect aproape mereu.

Cauza: mai multe cărți povestesc aceleași evenimente. Subiectul era găsit, dar câștiga capitolul din altă carte. Întrebările acestea ocolesc cuvintele evidente, de exemplu „africani captivi” în loc de „sclavi”. Potrivirea pe cuvinte nu le prinde.

## Soluția

Două feluri de căutare, adunate.

```
întrebare -> curățare -> potrivire pe cuvinte  --\
                      -> potrivire pe sens      --+-> scor adunat -> primele 3 documente
```

1. **Curățarea întrebării.** Se repară greșelile de tastare, se despart cuvintele lipite, se traduc câteva cuvinte spaniole.
2. **Potrivire pe cuvinte.** Se numără cuvintele comune dintre întrebare și text. Cuvintele rare contează mai mult. Metoda se numește BM25.
3. **Potrivire pe sens.** Un model mic de limbă (gte-small) transformă întrebarea într-un șir de 256 de numere. Fiecare pasaj din bibliotecă are deja șirul lui, calculat dinainte. Două texte cu sens apropiat au șiruri apropiate, chiar dacă nu au cuvinte comune.
4. **Adunarea.** Cele două scoruri se aduc la aceeași scară și se adună cu ponderi. Se întorc primele 3 documente.

Față de versiunea 2, modelul de limbă e mai mare (12 straturi în loc de 6). Asta a adus câștigul la capitolele de cărți.

## Cât de bine merge

Pe cele 268 de întrebări publice, cât de des e documentul corect în primele 3:

| Versiune | Toate | Wikipedia (108) | Restul (160) |
|---|---|---|---|
| 1 (picată pe platformă cu 0,793) | 0,873 | 0,991 | 0,794 |
| 2 | 0,918 | 1,000 | 0,863 |
| 3 (rezerva, doar numpy) | 0,914 | 1,000 | 0,856 |
| **4 (aceasta)** | **0,940** | 1,000 | **0,900** |

Coloana „Restul” e estimarea cinstită pentru platformă: la versiunea 1 a dat 0,794, iar platforma a dat 0,793.

Mărime: 51,26 MB despachetat, sub limita de 53. În 95% din cazuri o căutare durează cel mult 17 până la 21 de milisecunde, față de limita de 75.

## Ce nu e sigur

- Versiunea 4 are nevoie de biblioteca `onnxruntime` pe platformă. Scriptul de verificare al organizatorilor o trece printre cele permise, dar lista lor de cerințe nu o conține. De aceea există rezerva.
- Scorul de 0,900 e pe întrebările publice. Cele ascunse pot fi mai grele.

## Verificare

```
python evaluate.py
```

Scrie scorul pe întrebările publice, împărțit pe Wikipedia și restul, plus timpul unei căutări.

Verificarea organizatorilor (mărime, timp de încărcare, timp de căutare) a trecut la toate. Rezultatul ei e în `local_check_output.txt`.

`cuba_v4.zip` are sha256 `45a8499d35c2b74b33a45ac7239345fc1ce0da79c1fa5fdd0ee7a15c4a0e41e8`.

## Fișiere

- `main.py`, `index/`: ce se trimite. Regulile repo-ului exclud `index/dense.npz`, `index/lex.npz` și `index/encoder.onnx`. Se adaugă cu `git add -f` sau se refac cu scripturile de mai jos.
- `build_index.py`, `encode_passages.py`, `export_onnx.py`: construiesc `index/` din bibliotecă.
- `evaluate.py`: măsoară scorul local.
- [DETALII.md](DETALII.md): toate versiunile și încercările, cu cifre.
