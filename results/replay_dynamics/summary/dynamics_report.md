# Search dynamics (B6, reviewer R1.4)

Verification: 168 units expected, all present and every seed bit-identical to the reference run (algos: re).

## Regularised evolution, dataset-averaged

| Space | Rej./prop. | Found by 500 | Distinct final | Div@100 | Div@1000 | Unique@1000 |
|---|---|---|---|---|---|---|
| FULL | 0.25 | 0.54 | 10.0 | 2.88 | 2.89 | 768 |
| NO-NONE | 0.66 | 0.67 | 5.7 | 2.78 | 2.77 | 600 |
| SYNFLOW-50 | 0.37 | 0.57 | 7.3 | 2.82 | 2.83 | 716 |
| NASWOT-50 | 0.42 | 0.58 | 8.0 | 2.84 | 2.85 | 694 |
| PARAM-50 | 0.37 | 0.59 | 8.3 | 2.83 | 2.84 | 722 |
| RAND-4096 | 3.73 | 0.68 | 23.7 | 2.44 | 2.42 | 316 |
| RAND-7813 | 1.49 | 0.61 | 14.3 | 2.77 | 2.72 | 572 |

## Exploration breadth, re

| Space | Unique@100 | Unique@1000 | Revisit@1000 |
|---|---|---|---|
| FULL | 90 | 768 | 0.232 |
| NO-NONE | 85 | 600 | 0.400 |
| SYNFLOW-50 | 88 | 716 | 0.284 |
| NASWOT-50 | 87 | 694 | 0.306 |
| PARAM-50 | 88 | 722 | 0.278 |
| RAND-4096 | 60 | 316 | 0.684 |
| RAND-7813 | 79 | 572 | 0.428 |
