# Mozambic: The Professor of Maputo Bay

**Nu e gata de trimis. Agentul e mult sub prag.**

## Ce cere taskul

Un joc pe hartă, în doi, împotriva unui program al organizatorilor, „Profesorul”. Fiecare jucător adună materiale, își alege un loc pentru atelier, fabrică produse și livrează contracte. Câștigă cine strânge mai multe puncte.

Evaluarea se face pe 24 de hărți, cu câte 5 meciuri pe fiecare. Pentru o hartă contează **cel mai slab** dintre cele 5 meciuri. Pragul e 65%.

## Ce avem

`main.py` din acest folder. La fiecare mutare încearcă multe șiruri de câte 14 mutări proprii, pe o copie locală a regulilor, și păstrează cele mai bune 24. Alege șirul cu cel mai bun scor: puncte livrate, plus valoarea produselor și a materialelor strânse. Locul atelierului îl alege la început, comparând cele două locuri posibile.

## Cât de bine merge

| Cum e măsurat | Rezultat | Prag |
|---|---|---|
| Arena oficială, 3 rulări | 0,27 / 0,23 / 0,31 | 0,65 |
| Verificarea noastră pe 96 de hărți, nota pe hartă | 0,40 | 0,65 |
| Aceeași verificare, meciuri câștigate luate câte unul | 63% | |
| Măsurători mai noi ale aceluiași agent, pe 96 și pe 192 de hărți | 0,28 și 0,30 | 0,65 |

Agentul câștigă cele mai multe meciuri luate separat. Pierde la regula „cel mai slab din 5”: un singur meci pierdut sau egal pe o hartă strică toată harta. Sunt multe egaluri 34 la 34 și înfrângeri la un singur contract.

## De ce e sub prag

Agentul fabrică prea încet. La cea mai bună variantă de cercetare, produsele ies în jurul rundelor 29, 36 și 42. După estimarea noastră, al treilea ar trebui să iasă cam cu 10 runde mai devreme ca să bată Profesorul constant. Pentru `main.py` aceste runde nu au fost măsurate.

## Fișiere

- `main.py`: agentul.
- `research/`: încercările de îmbunătățire și măsurătorile. Rezumatul e în `research/RESULTS.txt`.
- `work/`: scripturi de probă.
- [DETALII.md](DETALII.md): notele scurte de la prima versiune.
