"""A synthetic run-to-failure generator, for fast and fully controlled testing.

The real validation runs on NASA C-MAPSS (:mod:`rotorwatch.real_data`). This module is
the other half of the same idea GridCast uses: a synthetic fleet whose degradation we
build ourselves, so every part is known. Because we choose which sensors carry the wear
signal and which are dead, a test can check that feature selection drops the dead ones,
that the model recovers the signal we injected, and that shuffling the target destroys
it — all without a network call and in a fraction of a second.

The frame has exactly the C-MAPSS columns (:data:`rotorwatch.features.UNIT_COL`,
:data:`~rotorwatch.features.TIME_COL`, the three operating settings, the 21 sensors, and
``RUL``), so the same feature pipeline and model run on it unchanged.

Degradation is deliberately *latent early and visible late*: each engine's sensors stay
flat while remaining life is above the cap and only start moving as failure approaches.
That is the real shape of run-to-failure data and the reason the RUL target is clipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OP_COLS, RUL_CAP, SENSOR_COLS, TIME_COL, UNIT_COL

# Mirror the C-MAPSS FD001 fingerprint: these channels are dead (report a constant), so
# feature selection must drop them. The rest carry the injected wear signal.
_CONSTANT_SENSORS = {
    "sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19",
}
_CONSTANT_VALUE = 100.0


def make_run_to_failure(
    n_units: int = 60,
    seed: int = 42,
    min_life: int = 130,
    max_life: int = 250,
    rul_cap: int = RUL_CAP,
    noise: float = 0.08,
) -> pd.DataFrame:
    """Return a synthetic run-to-failure fleet with the C-MAPSS column schema.

    ``n_units`` engines each run from cycle 1 to failure (a random lifetime between
    ``min_life`` and ``max_life``). ``RUL`` is the true remaining life (cycles to
    failure, reaching 0 at the last cycle). Informative sensors drift monotonically as
    failure nears — flat while ``RUL`` is above ``rul_cap``, then moving — while the
    channels in :data:`_CONSTANT_SENSORS` stay constant.
    """
    rng = np.random.default_rng(seed)
    informative = [s for s in SENSOR_COLS if s not in _CONSTANT_SENSORS]
    # A base level and a drift amplitude/direction per informative sensor, on varied
    # scales so standardisation has something to do.
    bases = rng.uniform(20.0, 9000.0, size=len(informative))
    amps = rng.uniform(0.05, 0.25, size=len(informative)) * bases
    directions = rng.choice([-1.0, 1.0], size=len(informative))

    frames = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(min_life, max_life + 1))
        cycles = np.arange(1, life + 1)
        rul = life - cycles  # true remaining useful life, 0 at failure
        # Degradation progress: 0 while still far from failure (RUL >= cap), rising
        # linearly to 1 at failure. This is why RUL above the cap is unobservable.
        progress = np.clip(1.0 - np.minimum(rul, rul_cap) / rul_cap, 0.0, 1.0)

        cols: dict[str, np.ndarray] = {
            UNIT_COL: np.full(life, unit, dtype=int),
            TIME_COL: cycles,
        }
        # Operating settings: two with mild jitter, one constant (as in FD001).
        cols[OP_COLS[0]] = rng.normal(0.0, 0.002, size=life)
        cols[OP_COLS[1]] = rng.normal(0.0, 0.0002, size=life)
        cols[OP_COLS[2]] = np.full(life, _CONSTANT_VALUE)

        for sensor in SENSOR_COLS:
            if sensor in _CONSTANT_SENSORS:
                cols[sensor] = np.full(life, _CONSTANT_VALUE)
            else:
                idx = informative.index(sensor)
                signal = bases[idx] + directions[idx] * amps[idx] * progress
                jitter = rng.normal(0.0, noise * amps[idx], size=life)
                cols[sensor] = signal + jitter

        cols["RUL"] = rul
        frames.append(pd.DataFrame(cols))

    return pd.concat(frames, ignore_index=True)
