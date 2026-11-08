"""Read one F1 season's weekends from the cached f1db zip (~/.cache/winning/f1db-csv.zip)."""
import csv, io, os, zipfile
from collections import defaultdict

ZIP = os.path.expanduser("~/.cache/winning/f1db-csv.zip")


def _rows(z, name, year):
    with z.open(name) as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
            if r["year"] == str(year):
                yield r


def load_season(year):
    """[{round, name, drivers, team, q1_out, q2_out, q3_order, q3_gap_s, finishers, dnf, winner, gap_s}]"""
    z = zipfile.ZipFile(ZIP)
    races = {r["round"]: r for r in _rows(z, "f1db-races.csv", year)}
    quali = defaultdict(list)
    for r in _rows(z, "f1db-races-qualifying-results.csv", year):
        quali[r["round"]].append(r)
    res = defaultdict(list)
    for r in _rows(z, "f1db-races-race-results.csv", year):
        res[r["round"]].append(r)
    out = []
    for rnd in sorted(races, key=int):
        q, rr = quali[rnd], res[rnd]
        if not q or not rr:
            continue
        q_drivers = [r["driverId"] for r in q]
        drivers = q_drivers + [r["driverId"] for r in rr if r["driverId"] not in set(q_drivers)]
        team = {r["driverId"]: r["constructorId"] for r in q}
        team.update({r["driverId"]: r["constructorId"] for r in rr if r["driverId"] not in team})
        no_quali = [d for d in drivers if d not in set(q_drivers)]
        raced = {r["driverId"] for r in rr}
        no_race = [d for d in drivers if d not in raced]
        if os.environ.get("Q_BY_TIMES"):
            # the old reading, kept for the record: a Q3 participant who set no
            # time (tyre saving, common in 2013) is read as eliminated in Q2 and
            # the prefix then places him behind drivers he out-qualified
            has_q2 = [r["driverId"] for r in q if r["q2Millis"]]
            has_q3 = [r["driverId"] for r in q if r["q3Millis"]]
        else:
            # default: the qualifying classification decides who reached Q2/Q3;
            # an untimed Q3 participant is top ten without an order
            def posn(r):
                return int(r["positionNumber"]) if r["positionNumber"] else 99
            has_q2 = [r["driverId"] for r in q if posn(r) <= 15]
            has_q3 = [r["driverId"] for r in q if posn(r) <= 10]
        # only classified Q3 drivers carry an order: a driver with a Q3 time but
        # no classification (excluded after the session, Silverstone 2013) is
        # neither placed nor behind anyone
        q3 = sorted([r for r in q if r["q3Millis"] and r["driverId"] in set(has_q3)], key=lambda r: int(r["q3Millis"]))
        best = int(q3[0]["q3Millis"]) if q3 else None
        classified = sorted([r for r in rr if r["positionNumber"]], key=lambda r: int(r["positionNumber"]))
        finishers = [r["driverId"] for r in classified]
        dnf = [r["driverId"] for r in rr if not r["positionNumber"]]
        gap = {r["driverId"]: (int(r["gapMillis"]) / 1000.0 if r["gapMillis"] else None) for r in classified}
        out.append(dict(round=int(rnd), name=races[rnd]["officialName"], drivers=drivers, team=team,
                        no_quali=no_quali, no_race=no_race,
                        q1_out=[d for d in q_drivers if d not in has_q2],
                        q2_out=[d for d in has_q2 if d not in has_q3],
                        q3_order=[r["driverId"] for r in q3],
                        q3_untimed=[d for d in has_q3 if d not in {r["driverId"] for r in q3}],
                        q3_gap_s={r["driverId"]: (int(r["q3Millis"]) - best) / 1000.0 for r in q3},
                        finishers=finishers, dnf=dnf + no_race, winner=finishers[0] if finishers else None,
                        gap_s=gap, reasons={r["driverId"]: r["reasonRetired"] for r in rr if not r["positionNumber"]}))
    return out


if __name__ == "__main__":
    s = load_season(2024)
    print(len(s), "weekends")
    w = s[0]
    print(w["name"], len(w["drivers"]), "drivers;", "Q1 out", w["q1_out"], "Q2 out", w["q2_out"])
    print("Q3 order", w["q3_order"][:5], "gaps", {k: w["q3_gap_s"][k] for k in w["q3_order"][:3]})
    print("winner", w["winner"], "dnf", w["dnf"], w["reasons"])
