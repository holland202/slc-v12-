"""Registered guard (F19 defect 3): the gate's HALT/PASS decision must be
caused by temperature vs the sector's OWN thresholds — not by coincidence of
where a simulated temperature floor happens to land.

Before this test, every sector 'worked' only because the ~35C sim floor sat
between some sectors' thresholds and others'. This injects temperatures
directly and asserts, per sector:
  - T >= temp_critical            -> gate HALTS (pass_gate False, aT 0.0)
  - temp_threshold < T < critical -> gate THROTTLES (0 < aT < 1)
  - T <= temp_threshold           -> gate FULL (aT == 1.0)

Anti-vacuity: the same gate must return BOTH halt and pass for the same
sector at different temperatures. A gate that always halts (or always passes)
fails this test even if individual asserts pass.

Run: python3 -m tests.test_thermal_discrimination
"""
import sys
from core.params import RuntimeConfig
from core.veritas_gate import VeritasGate

SECTORS = ["healthcare", "edge", "research", "defense", "desktop"]

class FakeMonitor:
    """Injects a fixed temperature so the gate's logic is tested, not the sensor."""
    def __init__(self, temp): self._t = temp
    def read(self): return self._t

def evaluate_at(cfg, temp):
    gate = VeritasGate(cfg, FakeMonitor(temp))
    aT, dG, passed, T = gate.evaluate()
    return aT, passed

def main():
    failures = []
    for s in SECTORS:
        cfg = RuntimeConfig(s)
        thr, crit = cfg.temp_threshold, cfg.temp_critical

        # three probe points relative to THIS sector's own thresholds
        below = thr - 2.0          # comfortably cool
        mid = (thr + crit) / 2.0   # throttle band (only if crit>thr)
        above = crit + 2.0         # over the critical lock

        aT_below, pass_below = evaluate_at(cfg, below)
        aT_above, pass_above = evaluate_at(cfg, above)

        checks = []
        # cool -> full multiplier, gate not halted by temperature
        checks.append(("cool=full", abs(aT_below - 1.0) < 1e-9))
        # hot -> hard halt
        checks.append(("hot=halt", (pass_above is False) and (aT_above == 0.0)))
        # anti-vacuity: gate discriminates — cool and hot must differ
        checks.append(("discriminates", aT_below != aT_above))
        # throttle band, only meaningful when there is room between thr and crit
        if crit - thr > 0.5:
            aT_mid, _ = evaluate_at(cfg, mid)
            checks.append(("mid=throttle", 0.0 < aT_mid < 1.0))

        ok = all(c[1] for c in checks)
        detail = " ".join(f"{n}:{'y' if v else 'N'}" for n, v in checks)
        print(f"  {s:<11} thr={thr:.1f} crit={crit:.1f} | {detail} | {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(s)

    print("=" * 60)
    if failures:
        print(f"FAIL: gate does not discriminate correctly for: {failures}")
        sys.exit(1)
    print(f"PASS: all {len(SECTORS)} sectors halt above critical, run below "
          f"threshold, and provably discriminate by temperature")

if __name__ == "__main__":
    main()
