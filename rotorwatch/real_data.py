"""Real NASA C-MAPSS turbofan run-to-failure data, for validation on data no one shaped.

The synthetic fleet (:mod:`rotorwatch.data`) proves the method recovers a signal we
injected. This module adds the real half: the NASA C-MAPSS turbofan degradation
benchmark, the canonical public dataset for Remaining Useful Life. Each engine is run
from a healthy state to failure while 21 sensors and three operating settings are logged
every cycle; the training split carries a per-cycle RUL that counts down to 0 at failure.

Source: the Hugging Face mirror ``penikmatrumput/nasa-cmapss-rul``, read through the
public datasets-server rows API (no authentication). That mirror concatenates all four
C-MAPSS sub-datasets (FD001-FD004); ``unit_number`` restarts at 1 for each. We vendor
**FD001** only — the canonical single-operating-condition, single-fault-mode set of 100
engines — which is the leading block of rows, up to the point where ``unit_number``
resets. Keeping to one sub-dataset avoids mixing engines from different operating
regimes under colliding unit numbers.

The original data is NASA's: A. Saxena, K. Goebel, D. Simon and N. Eggleman, "Damage
Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008; NASA
Prognostics Center of Excellence (PCoE) data repository. See ``datasets/README.md`` for
provenance, licence notes, and how to regenerate the vendored slice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .features import OP_COLS, RUL_COL, SENSOR_COLS, TIME_COL, UNIT_COL

COLUMNS = [UNIT_COL, TIME_COL, *OP_COLS, *SENSOR_COLS, RUL_COL]
_INT_COLUMNS = [UNIT_COL, TIME_COL, RUL_COL]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPO_ROOT / "datasets" / "cmapss_fd001.npz"

_DATASET = "penikmatrumput/nasa-cmapss-rul"
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={ds}"
    "&config=default&split=train&offset={off}&length={length}"
)
_PAGE = 100


def load_cmapss(path: str | Path | None = None) -> pd.DataFrame:
    """Load the vendored FD001 run-to-failure frame.

    Returns a DataFrame with :data:`COLUMNS`: ``unit_number``, ``time_in_cycles``, the
    three operating settings, the 21 sensors, and ``RUL`` (cycles remaining before that
    engine fails). ``unit_number``, ``time_in_cycles`` and ``RUL`` are integers.
    """
    target = Path(path) if path is not None else _DATASET_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Vendored C-MAPSS data not found at {target}. Regenerate it with "
            "`python -c 'from rotorwatch.real_data import fetch_cmapss; fetch_cmapss()'`."
        )
    with np.load(target, allow_pickle=False) as npz:
        data = npz["data"]
        columns = [str(c) for c in npz["columns"]]
    df = pd.DataFrame(data, columns=columns)
    for col in _INT_COLUMNS:
        df[col] = df[col].round().astype(int)
    return df[COLUMNS].reset_index(drop=True)


def _fetch_page(offset: int, length: int = _PAGE, tries: int = 6) -> list[dict]:
    """Fetch one page of rows from the datasets-server, with retry and backoff."""
    url = _ROWS_URL.format(ds=quote(_DATASET, safe=""), off=offset, length=length)
    for attempt in range(tries):
        try:
            with urlopen(url, timeout=90) as resp:  # noqa: S310 - fixed https host
                return json.load(resp).get("rows", [])
        except Exception:  # noqa: BLE001 - network retry with backoff
            if attempt == tries - 1:
                raise
            time.sleep(min(30, 3 * (attempt + 1)))
    return []


def fetch_cmapss(
    dest: str | Path | None = None,
    max_units: int = 100,
    max_offset: int = 25000,
    pause: float = 0.0,
) -> int:
    """Download the FD001 block and vendor it as a compressed ``.npz``.

    Pages the Hugging Face datasets-server from the start, collecting whole engines until
    ``unit_number`` resets (the FD001 -> FD002 boundary) or ``max_units`` engines have
    been gathered. Returns the number of vendored rows. Needs outbound access to
    ``datasets-server.huggingface.co``.
    """
    target = Path(dest) if dest is not None else _DATASET_PATH
    rows: list[list[float]] = []
    prev_unit = 0
    units_seen = 0
    offset = 0
    while offset < max_offset:
        page = _fetch_page(offset)
        if not page:
            break
        done = False
        for row in page:
            r = row["row"]
            unit = int(r[UNIT_COL])
            if unit < prev_unit:  # unit numbering reset -> next sub-dataset begins
                done = True
                break
            if unit != prev_unit:
                units_seen += 1
                if units_seen > max_units:
                    done = True
                    break
                prev_unit = unit
            rows.append([float(r[c]) for c in COLUMNS])
        offset += _PAGE
        if done:
            break
        if pause:
            time.sleep(pause)

    data = np.asarray(rows, dtype=np.float64)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, data=data, columns=np.asarray(COLUMNS))
    return len(rows)
