# Samoa: Island Duel

**Flag: `SIGMOID_OCEANICWAYFINDING`**

Agentul din acest folder s-a calificat la evaluarea oficială, iar harta primită după calificare e decodată.

## Ce cere taskul

Două părți.

**Partea 1: jocul.** Pe o tablă de 15×15 doi jucători ocupă pătrățele. Fiecare pătrățel are o valoare și un cost. Vezi doar pătrățelele tale și vecinii lor. Câștigă cine are valoarea mai mare la sfârșit. Adversarul e un program al organizatorilor, „campionul”.

Scrii un program care alege mutarea. Te califici dacă el câștigă cel puțin 308 din 512 meciuri și câștigă în cel puțin 24 din cele 32 de grupe de hărți.

**Partea 2: harta stelelor.** După calificare primești o hartă cu stele și o poezie. Din ele se scoate cuvântul din flag.

## Partea 1: agentul

```
stare -> ține minte ce a văzut pe tablă -> dă o notă fiecărei mutări -> alege nota cea mai mare
ultimele 10 mutări: încearcă toate variantele și o alege pe cea mai bună
```

1. **Memorie.** Agentul ține minte tot ce a văzut, pentru că ceața ascunde mare parte din tablă.
2. **Nota unei mutări.** Valoarea câștigată împărțită la energia cheltuită. Se adaugă câteva corecții: ce pătrățel bun se deschide după mutare, cât de expus rămâi lângă adversar, când e mai bine să te odihnești ca să strângi energie.
3. **Reglarea.** Ponderile corecțiilor au fost alese automat, jucând multe meciuri de practică.
4. **Sfârșitul jocului.** Pe ultimele 10 mutări ale jocului, numărate la ambii jucători, rămân puține variante. Agentul le încearcă pe toate și alege drumul cu cel mai bun scor final.

Fișierul e `my_agent.py`. Folosește `agent4.py`. Ponderile sunt scrise direct în el și sunt aceleași ca în `params/a4_eg10.json`.

## Cât de bine merge

| | Meciuri câștigate | Grupe câștigate |
|---|---|---|
| Evaluarea oficială (o singură rulare) | 319 din 512 | 32 din 32 |
| Pragul | 308 din 512 | 24 din 32 |
| Practică, 1.280 de meciuri | 60,4% | toate |
| Un al doilea agent, cu căutare scrisă în C++, pe aceleași hărți (nefolosit) | 44,1% | toate |

Dovada evaluării oficiale e în `evals/`: `attempt-1.json`, harta `attempt-1.chart.json` și jurnalul `LOG.md`.

## Partea 2: harta stelelor

Poezia de pe hartă spune, pe rând, ce trebuie făcut:

1. Cele 5 stele mari sunt Crucea Sudului, de pe steagul Samoei. Se iau în ordinea literelor grecești.
2. Fiecare stea mare are stele mici lângă ea. Se citește litera stelei mari, apoi literele stelelor mici, de la cea mai apropiată. Iese `TCEVNMKHEYYCNVIZU`.
3. Cheia e deviza Samoei, „Fa'avae i le Atua Samoa”, scrisă fără spații: `FAAVAEILEATUASAMOA`.
4. Din fiecare literă a textului se scade litera corespunzătoare din cheie (cifrul Vigenère).

```
text    TCEVNMKHEYYCNVIZU
cheie   FAAVAEILEATUASAMO
rezultat OCEANICWAYFINDING
```

„Oceanic wayfinding” e navigația polineziană după stele. Am refăcut scăderea literă cu literă și iese la fel.

## Ce nu e sigur

- Marja la calificare a fost mică: 319 față de 308. Harta e însă aceeași la orice rulare calificată, deci nu mai e nevoie de altă evaluare.

## Fișiere

- `my_agent.py`, `agent4.py`, `agent2.py`, `params/`: agentul calificat și ponderile lui.
- `practice.py`, `tune.py`: meciuri de practică în serie și reglarea ponderilor.
- `evals/`: evaluarea oficială și harta primită.
- `decode_chart.py`, `motto_keys.py`: decodarea hărții.
- `brute*.py`, `heur_agent.py`, `mc_agent.py`, `agent3.py`: încercări mai vechi.
- [DETALII.md](DETALII.md): descrierea lungă, cu toate versiunile agentului și cu felul în care a fost găsită cheia.
