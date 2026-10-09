"""RotorWatch: Remaining Useful Life estimation for rotating equipment."""

from __future__ import annotations

import os as _os

__version__ = "1.0.0"


def _cap_native_threads() -> None:
    """Cap native (OpenMP/BLAS) thread counts to the CPUs this process may actually use.

    scikit-learn's gradient boosting parallelises through OpenMP. In a container the
    OpenMP runtime often reads the *host's* core count instead of the cgroup limit and
    oversubscribes massively (dozens of threads fighting over a few cores), which can
    turn a one-second fit into minutes. We default the thread-count variables to the
    number of CPUs actually available to this process, before scikit-learn is imported.
    ``setdefault`` means an explicit choice by the user or CI always wins.
    """
    try:
        usable = len(_os.sched_getaffinity(0))  # respects cgroup / taskset limits
    except (AttributeError, OSError):
        usable = _os.cpu_count() or 1
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(var, str(max(1, usable)))


_cap_native_threads()
