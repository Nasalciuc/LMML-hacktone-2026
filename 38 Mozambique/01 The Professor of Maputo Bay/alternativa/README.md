# Mozambic: The Professor of Maputo Bay

**Nu e gata de trimis. Agentul e mult sub prag.**

## Ce cere taskul

Un joc pe hartă, în doi, împotriva unui program al organizatorilor, „Profesorul”. Fiecare jucător adună materiale, își alege un loc pentru atelier, fabrică produse și livrează contracte. Câștigă cine strânge mai multe puncte.

Evaluarea se face pe 24 de hărți, cu câte 5 meciuri pe fiecare. Pentru o hartă contează **cel mai slab** dintre cele 5 meciuri. Pragul e 65%.

## Ce avem

`main.py` din acest folder. La fiecare mutare încearcă multe șiruri de câte 14 mutări proprii, pe o copie locală a regulilor, și păstrează cele mai bune 24. Alege șirul cu cele mai multe puncte livrate. Locul atelierului îl alege la început, comparând cele două locuri posibile.

## Cât de bine merge

| Cum e măsurat | Rezultat | Prag |
|---|---|---|
| Arena oficială, 3 rulări | 0,27 / 0,23 / 0,31 | 0,65 |
| Verificarea noastră pe 96 de hărți, nota pe hartă | 0,40 | 0,65 |
| Aceeași verificare, meciuri câștigate luate câte unul | 63% | |

Agentul câștigă cele mai multe meciuri luate separat. Pierde la regula „cel mai slab din 5”: un singur meci pierdut sau egal pe o hartă strică toată harta. Sunt multe egaluri 34 la 34 și înfrângeri la un singur contract.

## De ce e sub prag

Agentul fabrică prea încet. Al treilea produs iese în jurul rundei 42. Ca să bată Profesorul constant ar trebui să iasă în jurul rundei 32.

## Repo

Soluția e și în repo-ul local, în `38 Mozambique\01 The Professor of Maputo Bay`, cu același README și cu descrierea lungă în `DETALII.md`. Nu e comisă și nu e urcată pe GitHub.
