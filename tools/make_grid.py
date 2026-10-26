"""Rechnet das Raster der Messreihen vor (E1 bis E7) und schreibt data/cointegration_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/cointegration_grid.json)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import koi_constants as C  # noqa: E402
import koi_evaluation as ev  # noqa: E402

JOBS = {"power": ev.power_cell, "null": ev.null_cell, "diff": ev.diff_cell, "adapt": ev.adapt_cell, "granger": ev.granger_cell, "bias": ev.bias_cell, "lagged": ev.lagged_cell}


def _run(task):
    kind, args, kwargs = task
    return kind, JOBS[kind](*args, **kwargs)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = [("power", (T, a), {}) for T in C.E1_TS for a in C.E1_AS]
    tasks += [("null", (T,), {}) for T in C.E2_TS]
    tasks += [("diff", (g, a), {}) for g in C.E3_GAMMAS for a in C.E3_AS]
    tasks += [("adapt", (T, a, s), {}) for T in C.E4_TS for a in C.E4_AS for s in C.E4_SHARES]
    tasks += [("granger", (c,), {}) for c in C.E5_COUPLINGS]
    tasks += [("bias", (T, k), {}) for T in C.E6_TS for k in C.E6_KS]
    tasks += [("lagged", (b,), {}) for b in (0.0, 0.4, 0.8, 1.2)]
    if quick:
        kw = {"seeds": (7,), "batches": 1, "size": 20}
        tasks = [("power", (52, 0.2), kw), ("null", (26,), kw), ("diff", (0.5, 0.2), kw), ("adapt", (52, 0.2, 1.0), kw), ("granger", (0.2,), kw), ("bias", (50, 3), {"seeds": (7,), "size": 20}),
                 ("lagged", (0.8,), kw)]
    t0 = time.time()
    out: dict = {k: [] for k in JOBS}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for kind, cell in pool.map(_run, tasks, chunksize=1):
            out[kind].append(cell)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("cointegration_grid_quick.json" if quick else "cointegration_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
