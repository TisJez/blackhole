"""Lightweight performance logger for GPU simulation profiling.

Records wall-clock timing for each stage of the simulation timestep loop,
using ``cupy.cuda.Device().synchronize()`` barriers when CuPy is available
to ensure accurate GPU timing.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone


class PerformanceLogger:
    """Collects per-stage, per-timestep wall-clock timings.

    Usage::

        pl = PerformanceLogger()
        for n in range(timesteps):
            pl.set_timestep(n)
            pl.start("solve_temperature")
            ...
            pl.stop()
        pl.save("logs")
    """

    def __init__(self):
        self._records: list[tuple[int, str, float]] = []
        self._timestep = 0
        self._stage: str | None = None
        self._t0: float = 0.0
        self._sync = self._resolve_sync()

    @staticmethod
    def _resolve_sync():
        """Return a GPU sync callable, or None if CuPy is unavailable."""
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            return cupy.cuda.Device().synchronize
        except Exception:
            return None

    def set_timestep(self, n: int) -> None:
        """Tag subsequent records with timestep *n*."""
        self._timestep = n

    def start(self, stage_name: str) -> None:
        """Begin timing *stage_name*. Inserts a GPU sync barrier first."""
        if self._sync is not None:
            self._sync()
        self._stage = stage_name
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        """End timing for the current stage. Inserts a GPU sync barrier."""
        if self._sync is not None:
            self._sync()
        elapsed = time.perf_counter() - self._t0
        self._records.append((self._timestep, self._stage, elapsed))
        self._stage = None

    def save(self, log_dir: str = "logs") -> str:
        """Write timing data to a timestamped subdirectory of *log_dir*.

        Each call creates ``<log_dir>/<YYYYMMDD_HHMMSS>/`` containing:

        - ``timestep_log.csv`` — every individual measurement
        - ``summary.json`` — machine-readable aggregate statistics
        - ``summary.txt`` — human-readable table sorted by total time

        A ``latest`` symlink (or copy on Windows) in *log_dir* always
        points to the most recent run.

        Returns the path to the run directory.
        """
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_dir, stamp)
        os.makedirs(run_dir, exist_ok=True)

        # --- timestep_log.csv ---
        csv_path = os.path.join(run_dir, "timestep_log.csv")
        with open(csv_path, "w") as f:
            f.write("timestep,stage,duration_s\n")
            for ts, stage, dur in self._records:
                f.write(f"{ts},{stage},{dur:.9f}\n")

        # --- Aggregate by stage ---
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for _, stage, dur in self._records:
            totals[stage] += dur
            counts[stage] += 1

        grand_total = sum(totals.values())

        summary: dict[str, dict] = {}
        for stage in sorted(totals, key=totals.get, reverse=True):
            pct = 100.0 * totals[stage] / grand_total if grand_total > 0 else 0.0
            avg_ms = 1000.0 * totals[stage] / counts[stage] if counts[stage] > 0 else 0.0
            summary[stage] = {
                "total_s": round(totals[stage], 6),
                "count": counts[stage],
                "avg_ms": round(avg_ms, 4),
                "pct": round(pct, 2),
            }

        # --- summary.json ---
        json_path = os.path.join(run_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # --- summary.txt ---
        txt_path = os.path.join(run_dir, "summary.txt")
        with open(txt_path, "w") as f:
            header = f"{'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg (ms)':>10} {'%':>7}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for stage, stats in summary.items():
                f.write(
                    f"{stage:<30} {stats['total_s']:>10.3f} {stats['count']:>7d} "
                    f"{stats['avg_ms']:>10.3f} {stats['pct']:>6.1f}%\n"
                )
            f.write("-" * len(header) + "\n")
            f.write(f"{'TOTAL':<30} {grand_total:>10.3f}\n")

        # --- Update ``latest`` pointer ---
        latest = os.path.join(log_dir, "latest")
        # On Windows, symlinks may need elevated privileges; fall back to
        # writing the run directory name into a plain text file.
        try:
            if os.path.islink(latest) or os.path.isfile(latest):
                os.remove(latest)
            os.symlink(stamp, latest)
        except OSError:
            with open(latest, "w") as f:
                f.write(stamp)

        return run_dir
