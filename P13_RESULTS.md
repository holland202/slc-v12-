# P13 — RESULTS

**Status:** ALL THREE CONDITIONS RESOLVED. c1 PASS, c2 REFUTED, c3 CONFIRMED-FAIL.
**Registered:** 2026-08-16 at `8704c11`, before the fan was ever switched on.
**Run:** 2026-08-17, ambient 24.5 C throughout, Black Shark phone cooler.
**Raw data:** `p13_results.json` (three records, appended in run order).

Registration preceded the hardware. Nothing below was reworded after the fact.

## G — instrument gate

PASSED on every run: 28 zones bound, `sim=False` at startup and at every read.
No condition was reported on fabricated data.

## Results

| id | config | phase | registered | measured | verdict |
|---|---|---|---|---|---|
| c1 | fan OFF | idle 60 s | max > 36.5 C | **51.3 C** (thermal_zone13) | **PASS** |
| c2 | fan ON | idle 60 s | max < 36.5 C sustained | **39.7 C** (thermal_zone8) | **REFUTED** |
| c3 | fan ON | load, 20 cycles | VeritasGate does not HALT | **20 halts / 20 cycles**, 53.4–70.7 C | **FAIL — as predicted** |

c1 PASSED, so a thermal deficit existed and c2/c3 are meaningful rather than
void. That was the whole purpose of the anti-vacuity control.

c3 was registered EXPECTED TO FAIL. It failed. That is a confirmed prediction,
not a disappointment — recorded that way in advance precisely so this outcome
could not be reinterpreted afterward.

## The finding c1 and c2 only give together

| | fan OFF | fan ON | delta |
|---|---|---|---|
| median | 35.9 | 29.8 | **−6.1** |
| min | 35.1 | 28.8 | −6.3 |
| max | 51.3 | 39.7 | −11.6 |

**The cooler works.** It removes roughly 6 C from the median compute-zone
temperature at idle. c2 failed anyway because the verdict is `max()` across 28
bound zones and a single outlier decides it — `thermal_zone13` fan-off,
`thermal_zone8` fan-on. **Different zones each time.**

Under the shipped matcher `["cpu", "cpuss", "gpuss"]` that is correct behaviour,
and P13 was deliberately registered against the shipped matcher so this result
is a before-picture, not a bug report.

**Stated plainly because it cuts against the verdict:** fan-on median 29.8 C is
*below* 36.5. With a narrower zone set, c2 might have passed. That does not
change c2's verdict — it was registered against `max()` over the shipped
matcher and it failed there. It makes the zone set the next question.

## c3 in context

Under numpy load the max compute zone runs 53.4–70.7 C against a 36.5 C
threshold: 17 to 34 degrees over. No phone cooler closes that. Every published
sector ceiling — healthcare 32.0, edge 34.0, research 35.5, defense 36.5,
desktop 38.0 — sits below the measured *minimum* under load of 53.4 C.

This reframes the open c4. It is not "which profile survives with the fan on."
The measured answer is **none of them do**. c4 becomes: what would a profile
have to look like to be honest about this silicon under compute? That is a
finding about the profile table in `core/params_sector.py`, not about the fan.

## Open, unrun

- **c4** — as reframed above. What ceiling would a load-bearing profile need?
- **c5** — does c2 pass under a narrowed zone set? The `exclude` work in
  `hardware_link.py` is stranded uncommitted in the NON-canonical `~/slc-v12-`
  checkout. This must be registered as its own claim before being run. It is
  not a retrofit of c2, and c2's REFUTED verdict stands regardless of outcome.
- **c6** — 60 s tests whether the fan holds idle temperature, not whether it
  holds it over an hour. Sustained-duration behaviour is unmeasured.
