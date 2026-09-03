#!/data/data/com.termux/files/usr/bin/env python3
"""
sovereign_daemon.py  — replacement for the inert ~/.shortcuts/SOVEREIGN_DAEMON loop.

Fixes three recorded defects in the old shortcut:
  1. Old loop ran `|| rish -c "am kill-all"` on ANY nonzero exit, so a refusal,
     a malformed line, and a real crash all collapsed into the same kill. This
     daemon CLASSIFIES the decision (ALLOW / REFUSE / MALFORMED / ERROR) and logs
     WHY. Nothing is ever executed on a refusal or a parse failure.
  2. Old thermal reading maxed over all ~68 sysfs zones incl. modem/battery that
     idle at 60C (the F13 bug), so a fixed envelope never unpaused. This reads a
     SELECTED zone set relative to an idle baseline captured at startup.
  3. Termux-safe: state under $HOME (never /tmp), single-process PID lock,
     graceful SIGTERM.

SAFETY DEFAULT: dry-run. It decides and logs but executes NOTHING until you pass
--arm. Even armed, it only executes lines classified ALLOW, and never has an
unconditional kill fallback. Runs fine whether or not `rish` exists.
"""
import os, sys, time, json, signal, subprocess, shutil, hashlib, argparse, re
from pathlib import Path
from datetime import datetime, timezone

HOME = Path(os.environ["HOME"])
STATE = HOME / "sovereign_daemon"          # $HOME, not /tmp
LOG = STATE / "daemon.log"
DECISIONS = STATE / "decisions.jsonl"
PIDFILE = STATE / "daemon.pid"

# --- thermal: SELECTED zones only, not all 68. Edit this list once the probe
#     shows the real type names. These substrings are matched against
#     /sys/class/thermal/thermal_zone*/type; modem/battery deliberately excluded. ---
# Aggregate subsystem zones only. Per-core cpu-0-X-Y sensors spike for
# milliseconds when a core wakes — measured 2026-09-03: max over 17 of them
# swung 59.2 -> 37.0 -> 57.3 in six seconds. Aggregates are stable.
THERMAL_INCLUDE_SUBSTR = ("cpuss", "gpuss")
THERMAL_EXCLUDE_SUBSTR = ("modem", "mdmss", "batt", "usb", "pa-", "wifi", "charger")
# Reject any zone whose name is a per-core sensor: cpu-<n>-<n>-<n>
PER_CORE_RE = re.compile(r"^cpu-\d+-\d+-\d+$")
EMA_ALPHA = 0.3   # smoothing; 1.0 = no smoothing, lower = smoother

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    try:
        with LOG.open("a") as f: f.write(line + "\n")
    except Exception: pass

def record(obj):
    try:
        with DECISIONS.open("a") as f: f.write(json.dumps(obj) + "\n")
    except Exception as e: log(f"WARN could not write decision: {e}")

# ---------- thermal ----------
def _zone_paths():
    base = Path("/sys/class/thermal")
    if not base.exists(): return []
    out = []
    for z in sorted(base.glob("thermal_zone*")):
        try: t = (z / "type").read_text().strip().lower()
        except Exception: continue
        if any(s in t for s in THERMAL_EXCLUDE_SUBSTR): continue
        if PER_CORE_RE.match(t): continue          # per-core sensor: too twitchy
        if THERMAL_INCLUDE_SUBSTR and not any(s in t for s in THERMAL_INCLUDE_SUBSTR):
            continue
        out.append((t, z / "temp"))
    return out

_ema = {"v": None}

def read_temp_smoothed(zones, alpha=None):
    """EMA-smoothed max over aggregate zones. Returns (smoothed, raw)."""
    raw = read_temp_c(zones)
    if raw is None: return None, None
    a = EMA_ALPHA if alpha is None else alpha
    _ema["v"] = raw if _ema["v"] is None else (a*raw + (1-a)*_ema["v"])
    return round(_ema["v"], 1), raw

def read_temp_c(zones):
    """Max over SELECTED zones only. Returns None if no zone is readable."""
    vals = []
    for _, p in zones:
        try:
            raw = int(p.read_text().strip())
            vals.append(raw / 1000.0 if raw > 1000 else float(raw))
        except Exception: continue
    return max(vals) if vals else None

# ---------- decision classification ----------
# The OLD loop treated everything nonzero as "kill". We instead read one line of
# proposed action and classify it. Only ALLOW is eligible for execution.
REFUSE_MARKERS = ("[refuse", "refused", "i can't", "i cannot", "not permitted",
                  "logic_hole", "err:", "[!]")

def classify(line: str):
    s = (line or "").strip()
    if not s:
        return ("MALFORMED", "empty line")
    low = s.lower()
    if any(m in low for m in REFUSE_MARKERS):
        return ("REFUSE", "refusal marker present")
    # An allowed action must look like a bounded command, not prose.
    if len(s) > 400:
        return ("MALFORMED", f"over-long ({len(s)} chars) — prose, not a command")
    if s.count(" ") > 40:
        return ("MALFORMED", "too many tokens — not a single command")
    return ("ALLOW", "passed classifier")

def rish_available():
    return shutil.which("rish") is not None

def execute(cmd: str, armed: bool):
    """Only ever called for ALLOW. No unconditional kill fallback, ever."""
    if not armed:
        return ("DRY_RUN", "armed=False; not executed")
    if not rish_available():
        return ("NO_EXECUTOR", "rish not on PATH; logged, not executed")
    try:
        r = subprocess.run(["rish", "-c", cmd], capture_output=True,
                           text=True, timeout=30)
        return (f"EXEC_EXIT_{r.returncode}", (r.stdout or r.stderr or "").strip()[:200])
    except subprocess.TimeoutExpired:
        return ("EXEC_TIMEOUT", "30s")
    except Exception as e:
        return ("EXEC_ERROR", str(e)[:200])

# ---------- input source ----------
def read_proposed_action(source_cmd):
    """Get one proposed-action line. If no source given, this daemon idles and
    only heartbeats thermal — it never fabricates an action."""
    if not source_cmd:
        return None
    try:
        r = subprocess.run(source_cmd, shell=True, capture_output=True,
                           text=True, timeout=60)
        lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
        return lines[-1] if lines else None
    except Exception as e:
        log(f"WARN source failed: {e}")
        return None

# ---------- lifecycle ----------
_stop = False
def _sig(_s, _f):
    global _stop; _stop = True
    log("SIGTERM received — shutting down cleanly")

def acquire_lock():
    STATE.mkdir(parents=True, exist_ok=True)
    if PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text().strip())
            os.kill(old, 0)          # raises if not running
            log(f"REFUSING START: daemon already running pid {old}")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            log("stale pidfile, reclaiming")
        except PermissionError:
            log(f"pid {old} exists (perm) — refusing"); sys.exit(1)
    PIDFILE.write_text(str(os.getpid()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="store_true",
                    help="actually execute ALLOW lines (default: dry-run, execute nothing)")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--source", default="",
                    help="shell command whose LAST stdout line is the proposed action; "
                         "omit to run as a thermal-only heartbeat that executes nothing")
    ap.add_argument("--thermal-ceiling", type=float, default=None,
                    help="pause-execution ceiling in C above idle baseline (see probe output first)")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    acquire_lock()

    zones = _zone_paths()
    baseline = read_temp_c(zones)
    log(f"START armed={args.arm} source={'yes' if args.source else 'none (heartbeat)'} "
        f"rish={'yes' if rish_available() else 'no'} "
        f"zones={len(zones)} baseline={baseline}")
    if not args.arm:
        log("DRY-RUN: decisions logged, NOTHING executed. Pass --arm to enable execution.")

    try:
        while not _stop:
            temp, temp_raw = read_temp_smoothed(zones)
            hot = (args.thermal_ceiling is not None and baseline is not None
                   and temp is not None and (temp - baseline) > args.thermal_ceiling)

            proposed = read_proposed_action(args.source)
            if proposed is None:
                record({"t": now(), "temp_c": temp, "temp_raw": temp_raw, "event": "heartbeat", "hot": hot})
            else:
                verdict, reason = classify(proposed)
                result = ("SKIPPED_THERMAL", f"+{(temp-baseline):.1f}C over baseline") if hot \
                         else (execute(proposed, args.arm) if verdict == "ALLOW" \
                               else ("NOT_EXECUTED", verdict))
                rec = {"t": now(), "temp_c": temp, "hot": hot,
                       "proposed_sha": hashlib.sha256(proposed.encode()).hexdigest()[:12],
                       "verdict": verdict, "reason": reason,
                       "result": result[0], "detail": result[1]}
                record(rec)
                log(f"{verdict} ({reason}) -> {result[0]}  temp={temp} (raw {temp_raw})")
            time.sleep(args.interval)
    finally:
        try: PIDFILE.unlink()
        except Exception: pass
        log("STOPPED, pidfile released")

if __name__ == "__main__":
    main()
