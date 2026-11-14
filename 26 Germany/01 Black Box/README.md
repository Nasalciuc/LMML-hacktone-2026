Black Box (Germany). Run: `python evaluation_script.py "../01 Black Box/main.py" --eval-type train` from task_germany.
Method: 32 cube vertices + centre + 30 LHS points, full quadratic fit; the step is found by scanning half-space cuts of the residuals until they vanish, then the boundary is bracketed by bisection on segments between opposite-side points and a plane is fitted, dropping variables it does not need.
Practice function: 100/100 with 133 of 150 queries.
Synthetic functions of the same family (random inert/relevant vars, interaction pair and sign, 1-3 gate vars, jump sign/size): 700/700 scored 100, including slivers down to 0.3% of the cube, jumps of 0.1 and linear effects of 0.03. Queries 130-136, hard cap set at 140.
A jump has no sign by itself (g + J*[H] = g + J - J*[not H]); the side with the upward step is reported as active.
