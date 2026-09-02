#!/data/data/com.termux/files/usr/bin/env python3
"""Selftest gate for sovereign_daemon classify/thermal logic.
Both directions proven: --selftest exits 0 on correct behavior,
--sabotage flips one rule and MUST exit 1. Matches slc-v12- gate convention."""
import sys, importlib.util
spec = importlib.util.spec_from_file_location("d", "sovereign_daemon.py")
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)

def check(sabotage=False):
    cases = [
        ("am start -n a/b",                    "ALLOW"),
        ("[!] LOGIC_HOLE_DETECTED",            "REFUSE"),
        ("I cannot comply",                    "REFUSE"),
        ("",                                   "MALFORMED"),
        ("word " * 60,                         "MALFORMED"),
    ]
    passed = 0
    for text, want in cases:
        got, _ = d.classify(text)
        if sabotage and want == "REFUSE":
            want = "ALLOW"   # deliberately wrong — gate must catch this
        ok = (got == want)
        passed += ok
        print(f"  [{'ok' if ok else 'XX'}] classify({text[:20]!r:24}) = {got:10} want {want}")
    # anti-vacuity: the instrument CAN return each verdict (not stuck on one)
    verdicts = {d.classify(t)[0] for t, _ in cases}
    multi = len(verdicts) >= 3
    print(f"  anti-vacuity: {len(verdicts)} distinct verdicts {sorted(verdicts)} "
          f"({'ok' if multi else 'FAIL — classifier inert'})")
    return passed == len(cases) and multi

if "--sabotage" in sys.argv:
    ok = check(sabotage=True)
    print(f"SABOTAGE: {'exit 0 (BAD — gate blind)' if ok else 'exit 1 (correct — caught)'}")
    sys.exit(0 if ok else 1)
else:
    ok = check(sabotage=False)
    print(f"SELFTEST: {'5/5 exit 0' if ok else 'FAIL exit 1'}")
    sys.exit(0 if ok else 1)
