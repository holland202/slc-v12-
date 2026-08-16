# P13 — Does the attached cooler bring this substrate under the defense profile?

**Status:** REGISTERED, NOT RUN.
**Registered:** 2026-08-16, before the fan was switched on for the first time.
**Instrument:** `probe_cooler.py` (selftest 6/6, sabotage exit 1).
**Repo state at registration:** `0e2dbdb`.

## Background

`probe_slc_claims.py` P0 is a substrate claim — "this substrate can run the
defense profile (live T < temp_threshold)" — and it MISMATCHES: measured
66.90 C live max compute zone against the defense profile's 36.5 C
(`core/params_sector.py`). The published reading of that is that the defense
sector cannot run on this device.

A cooler is now physically attached. P13 asks whether it closes that gap.

P13 does **not** amend P0. P0 keeps its wording and its MISMATCH verdict.
Adding a `hardware_config` field to P0's record is a separate commit, because
after a fan exists P0's number silently depends on whether the fan happened to
be plugged in when the probe ran — two runs, opposite verdicts, both honest,
and nothing in the record to tell them apart.

Numbering: P0-P10 and P12a-d are taken. **P11 is a gap** in the tree and is
left as one — its history is not visible here, and filling a gap is how a
retracted claim gets resurrected under its old number.

## G — instrument gate (not a prediction)

`ThermalMonitor` must be bound to real zones at startup and at every read.

`core/hardware_link.py::read()` falls back to simulation mode — a synthetic
30-45 C drift — if every bound zone fails to read. Under that fallback **c2
passes on fabricated data**. G exists so a sensor dropout and a cool phone
cannot look the same.

If G fails, **no condition is reported in either direction** and the script
exits 1. G is not a result.

## Registered conditions

| id | config | phase | registered claim | falsified if |
|---|---|---|---|---|
| **c1** | fan OFF | idle, 60 s | max compute zone **> 36.5 C** | max <= 36.5 C |
| **c2** | fan ON | idle, 60 s | max compute zone **< 36.5 C sustained**, every sample | any sample >= 36.5 C |
| **c3** | fan ON | under load, 20 cycles | `VeritasGate.evaluate()` does **not** HALT | any HALT |

**c1 is the anti-vacuity control.** If c1 fails, the phone is already under the
defense threshold unaided, there is no deficit for the cooler to close, and
**c2 and c3 are VOID — not passed.** A cooler cannot be credited with closing a
gap that was not there.

**c3 is registered EXPECTED TO FAIL.** Recorded now so that a failure is a
confirmed prediction rather than a disappointment, and so that a pass is a
surprise that cannot be reinterpreted after the fact. Prior measurement:
`cpu-0-1-1` passes 65 C under numpy load against a 36.5 C threshold; a fan is
not expected to cover a ~29 C margin.

c3 calls the real `VeritasGate`, not a reimplementation of the HALT rule. A
reimplementation would test this script, not the governor.

## Required with every run

- **ambient room temperature**, both fan-off and fan-on. Without it c1 and c2
  are not comparable and the result is a weather report. The script refuses to
  run without `--ambient`.
- hardware config string, recorded in the JSON output.
- the zone that produced the max, by name — not just the value.

## Known limits of this instrument, stated in advance

- Measures against the **shipped** 28-zone matcher (`["cpu","cpuss","gpuss"]`).
  The `exclude` work that narrows this is stranded uncommitted in the
  non-canonical `~/slc-v12-` checkout and is deliberately not used here. If
  `exclude` later lands, P13's numbers are the before-picture.
- `max()` over zones, never a mean — three zones on this SoC report -273 C and
  are filtered by `_read_one`.
- 60 s is a short window. It tests whether the fan holds idle temperature, not
  whether it holds it over an hour.

## Open

- P13-c4, unrun: if c3 fails, what is the highest sector profile that does NOT
  halt under load with the fan on — healthcare 32.0 / edge 34.0 / research
  35.5 / defense 36.5 / desktop 38.0? That converts a failure into a
  measurement of what the cooler actually bought.
