### `multitask-train` — 200000 molecules, 1000-bit budget

| codec | family | accuracy | 95% low | ceiling | median bits | p99 | max | fits budget | native bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smiles+arith-o3` **OK** | entropy | 100.0000% | 99.9981% | 100.00% | 80 | 177 | 568 | 100.0% | - |
| `selfies+arith-o3` **OK** | entropy | 99.9990% | 99.9964% | 100.00% | 83 | 185 | 388 | 100.0% | - |
| `smiles+huffman` **OK** | entropy | 100.0000% | 99.9981% | 100.00% | 174 | 298 | 764 | 100.0% | - |
| `smiles` **OK** | string | 100.0000% | 99.9981% | - | 276 | 468 | 888 | 100.0% | - |
| `selfies` | string | 99.9990% | 99.9964% | 0.00% | 294 | 518 | 1008 | 100.0% | - |
| `graph-bits` | structural | 100.0000% | 99.9981% | 100.00% | 619 | 1057 | 2336 | 97.7% | - |