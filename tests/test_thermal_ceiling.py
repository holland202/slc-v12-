"""Registered guard (F19): NO sector may load a threshold above the 38.5C wall.
Turns defect 2 into a permanent test. Run: python3 -m tests.test_thermal_ceiling"""
import sys
from core.params import RuntimeConfig

WALL = 38.5
SECTORS = ["healthcare", "edge", "research", "defense", "desktop"]

def main():
    failed = []
    for s in SECTORS:
        cfg = RuntimeConfig(s)
        t, c = cfg.temp_threshold, cfg.temp_critical
        ok = t <= WALL and c <= WALL
        print(f"  {s:<11} threshold={t:.1f}C critical={c:.1f}C  {'OK' if ok else 'FAIL — EXCEEDS WALL'}")
        if not ok:
            failed.append(s)
    print("=" * 52)
    if failed:
        print(f"FAIL: {len(failed)} sector(s) exceed the {WALL}C wall: {failed}")
        sys.exit(1)
    print(f"PASS: all {len(SECTORS)} sectors within the {WALL}C hardware wall")

if __name__ == "__main__":
    main()
