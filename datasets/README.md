# Datasets

## `cmapss_fd001.npz` — NASA C-MAPSS turbofan run-to-failure (FD001)

Real run-to-failure sensor logs for 100 simulated turbofan engines, used to validate the
RUL estimator on data no one designed for this project (see *Results on real C-MAPSS
data* in the top-level README). C-MAPSS is the canonical public benchmark for Remaining
Useful Life.

**Source.** The Hugging Face mirror
[`penikmatrumput/nasa-cmapss-rul`](https://huggingface.co/datasets/penikmatrumput/nasa-cmapss-rul),
read through the public datasets-server rows API (no authentication):

```
https://datasets-server.huggingface.co/rows?dataset=penikmatrumput%2Fnasa-cmapss-rul&config=default&split=train&offset=N&length=100
```

That mirror concatenates all four C-MAPSS sub-datasets (FD001–FD004, 160,359 rows total),
and `unit_number` **restarts at 1 for each sub-dataset**. We vendor **FD001 only** — the
leading block of rows, up to the point where `unit_number` first resets — because FD001 is
the canonical single-operating-condition, single-fault-mode set and keeping to one
sub-dataset avoids mixing engines from different operating regimes under colliding unit
numbers. That block is exactly 20,631 rows for engines 1–100.

**Original data.** NASA Ames Prognostics Center of Excellence (PCoE), "Turbofan Engine
Degradation Simulation Data Set", generated with the C-MAPSS simulator. Cite:

> A. Saxena, K. Goebel, D. Simon, and N. Eggleman, "Damage Propagation Modeling for
> Aircraft Engine Run-to-Failure Simulation," *International Conference on Prognostics and
> Health Management (PHM08)*, Denver, CO, 2008.

**Columns.**

| Column            | Meaning                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `unit_number`     | engine id (1–100 within FD001)                                          |
| `time_in_cycles`  | operating cycle, 1 at the start of life                                 |
| `op_setting_1..3` | three operating settings (op_setting_3 is constant in FD001)            |
| `sensor_1..21`    | 21 sensor channels (7 are flat in FD001 and carry no wear signal)       |
| `RUL`             | Remaining Useful Life — cycles left before that engine fails, 0 at failure |

Each engine runs from a healthy state to failure, so within an engine `RUL` decreases by 1
each cycle and reaches 0 on the last. The file is stored as a compressed NumPy archive
(`data` = the row matrix, `columns` = the names); `rotorwatch.real_data.load_cmapss` reads
it back into a DataFrame with the integer columns restored. Size on disk is ~0.6 MB.

### Regenerating

```bash
python -c "from rotorwatch.real_data import fetch_cmapss; print(fetch_cmapss())"
```

`fetch_cmapss` pages the datasets-server from the start, gathers whole engines until
`unit_number` resets (the FD001 → FD002 boundary), and rewrites the `.npz`. It needs
outbound access to `datasets-server.huggingface.co`. The licence and terms of the original
NASA PCoE data and of the Hugging Face mirror apply; this FD001 slice is vendored for
reproducible benchmarking only.
