| task | condition | cat | params | nonzero | max abs w | sigma50 abs | rel | nonzero-only | cert radius | attack first break | attack/cert | abs/cert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dyck8 | constrained | 3b | 25698 | 0.001 | 240 | 2.43e-03 | 7.57e-03 | 4.69e-02 | 6.78e-06 | 1.02e-05 | 1.5 | 359 |
| dyck8 | unconstrained | 3b | 378 | 0.071 | 60 | 5.63e-03 | 3.02e-02 | 2.44e-02 | 6.77e-06 | 1.02e-05 | 1.5 | 832 |
| hist | constrained | 3b | 1881 | 0.023 | 170 | 8.03e-04 | 1.09e-03 | 9.85e-04 | 1.68e-05 | 1.19e-04 | 7.1 | 48 |
| hist | unconstrained | 3b | 550 | 0.078 | 159 | 7.48e-04 | 1.25e-03 | 1.44e-03 | 1.68e-05 | 1.19e-04 | 7.1 | 44 |
| length | constrained | 3b | 1881 | 0.012 | 56 | 4.58e-03 | 2.09e-02 | 9.47e-03 | 6.32e-04 | 1.17e-03 | 1.9 | 7 |
| length | constrained | 4 | 1881 | 0.012 | 56 | 4.58e-03 | 2.09e-02 | 9.47e-03 | 6.32e-04 | 1.17e-03 | 1.9 | 7 |
| length | unconstrained | 3b | 146 | 0.158 | 56 | 4.28e-03 | 4.87e-02 | 4.51e-03 | 6.36e-04 | 1.18e-03 | 1.9 | 7 |
| length | unconstrained | 4 | 146 | 0.158 | 56 | 4.28e-03 | 4.87e-02 | 4.51e-03 | 6.36e-04 | 1.18e-03 | 1.9 | 7 |
| majority8 | constrained | 3b | 25698 | 0.001 | 240 | 2.91e-02 | 5.42e-02 | 6.40e-02 | 3.50e-05 | 1.00e-04 | 2.9 | 832 |
| majority8 | unconstrained | 3b | 196 | 0.112 | 60 | 6.84e-03 | 5.27e-02 | 1.82e-02 | 3.49e-05 | 9.97e-05 | 2.9 | 196 |
| parity | constrained | 3b | 6706 | 0.005 | 170 | 2.44e-03 | 5.03e-03 | 5.35e-03 | 7.86e-06 | 1.29e-04 | 16.4 | 311 |
| parity | constrained | 4 | 6706 | 0.005 | 170 | 2.44e-03 | 5.03e-03 | 5.35e-03 | 7.86e-06 | 1.29e-04 | 16.4 | 311 |
| parity | unconstrained | 3b | 160 | 0.194 | 98 | 2.11e-03 | 3.46e-03 | 3.16e-03 | 7.83e-06 | 1.87e-04 | 23.9 | 270 |
| parity | unconstrained | 4 | 160 | 0.194 | 98 | 2.11e-03 | 3.46e-03 | 3.16e-03 | 7.83e-06 | 1.87e-04 | 23.9 | 270 |
| reverse | constrained | 3b | 17543 | 0.004 | 320 | 1.03e-03 | 1.53e-02 | 4.53e-03 | 9.69e-06 | 1.22e-04 | 12.6 | 106 |
| reverse | unconstrained | 3b | 1967 | 0.033 | 196 | 6.08e-04 | 1.71e-02 | 8.80e-03 | 9.69e-06 | 1.22e-04 | 12.6 | 63 |

| task | condition | embeddings | attn q/k | attn v/o | mlp in | mlp out | unembed | zero-only |
|---|---|---|---|---|---|---|---|---|
| dyck8 | constrained | 4.10e-03 | inf | 7.90e-02 | 1.22e-01 | 4.48e-02 | 3.08e-01 | 2.60e-03 |
| dyck8 | unconstrained | 2.93e-03 | 9.41e-01 | 8.86e-02 | 1.59e-01 | 3.05e-02 | 3.08e-01 | 7.18e-03 |
| hist | constrained | 1.09e-03 | 1.53e-03 | 2.63e-02 | 3.64e-03 | inf | 3.07e-01 | 1.23e-03 |
| hist | unconstrained | 1.06e-03 | 2.10e-03 | 2.94e-02 | 4.23e-03 | 6.95e-01 | 3.07e-01 | 1.08e-03 |
| length | constrained | 7.46e-03 | 5.21e-01 | 1.28e-02 | 9.47e-03 | 7.26e-01 | 2.33e-01 | 5.62e-03 |
| length | unconstrained | 8.44e-03 | inf | 1.21e-02 | 4.51e-03 | 7.26e-01 | 2.33e-01 | 5.38e-03 |
| majority8 | constrained | 3.16e-02 | inf | 1.27e-01 | 5.61e-01 | 7.91e-01 | 4.05e-01 | 3.36e-02 |
| majority8 | unconstrained | 1.31e-02 | 3.66e-01 | 2.81e-02 | 6.45e-01 | 5.71e-01 | 5.63e-01 | 7.84e-03 |
| parity | constrained | 7.07e-03 | 4.93e-02 | 7.46e-02 | 5.57e-03 | 2.47e-01 | 3.00e-01 | 4.14e-03 |
| parity | unconstrained | 8.22e-03 | 7.24e-02 | 2.10e-01 | 2.99e-03 | 9.16e-01 | 8.00e-01 | 3.53e-03 |
| reverse | constrained | 7.92e-03 | 1.58e-02 | 2.85e-03 | 1.24e-03 | 3.24e-01 | 2.44e-01 | 1.14e-03 |
| reverse | unconstrained | 8.62e-03 | 1.63e-02 | 7.00e-03 | 7.56e-04 | 2.85e-01 | 2.44e-01 | 6.32e-04 |

| task | Tracr (compiler, category 1) abs | LLM constrained abs | LLM unconstrained abs | Tracr rel | LLM constrained rel | LLM unconstrained rel | Tracr params | LLM constrained / unconstrained params |
|---|---|---|---|---|---|---|---|---|
| length | 2.69e-04 | 4.58e-03 | 4.28e-03 | 9.90e-04 | 2.09e-02 | 4.87e-02 | 2391 | 1881 / 146 |
| hist | 1.38e-03 | 8.03e-04 | 7.48e-04 | 2.78e-03 | 1.09e-03 | 1.25e-03 | 2391 | 1881 / 550 |
| reverse | 6.10e-04 | 1.03e-03 | 6.08e-04 | 1.65e-03 | 1.53e-02 | 1.71e-02 | 58289 | 17543 / 1967 |
