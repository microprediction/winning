"""Join the Oddschecker snapshots (f1_market_all.csv) to f1db weekends:
race slug -> grandPrixId (GP_MAP from research/f1_market_test.py), driver
name -> driverId (normalised full name), snapshot time -> phase relative
to qualifying and race (pre_quali / post_quali / post_race)."""
import ast, csv, io, os, re, sys, unicodedata, zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.expanduser("~/.cache/winning/f1db-csv.zip")
MARKET = os.path.join(HERE, "f1_market_all.csv")   # from parse_all.py
_src = open(os.path.join(HERE, "..", "..", "f1_market_test.py")).read()
GP_MAP = ast.literal_eval(_src[_src.index("GP_MAP = {") + len("GP_MAP = "):_src.index("\n}\n", _src.index("GP_MAP = {")) + 2])


ALIASES = {"daniil kyvat": "daniil kvyat"}          # bookmaker typos


def _norm(s):
    s = s.split("(")[0].strip()
    if "," in s:
        last, first = [t.strip() for t in s.split(",", 1)]; s = f"{first} {last}"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()   # accents to base letters
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def load_drivers():
    z = zipfile.ZipFile(ZIP)
    with z.open("f1db-drivers.csv") as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
    by_name = {}
    for r in rows:
        for key in {_norm(r["fullName"]), _norm(r["lastName"]), _norm(r["firstName"] + " " + r["lastName"])}:
            by_name.setdefault(key, set()).add(r["id"])
    return by_name


def load_races(year):
    z = zipfile.ZipFile(ZIP)
    out = {}
    with z.open("f1db-races.csv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
            if r["year"] != str(year):
                continue
            def dt(d, t):
                if not d:
                    return None
                return datetime.fromisoformat(d + "T" + (t or "12:00:00").replace("Z", "")).replace(tzinfo=timezone.utc)
            out[r["grandPrixId"]] = dict(round=int(r["round"]), race=dt(r["date"], r["time"]),
                                         quali=dt(r.get("qualifyingDate"), r.get("qualifyingTime")))
    return out


def snapshots(year, season_drivers):
    """{round: [(phase, ts, {driverId: prob})]} for one season; season_drivers = ids seen in f1db that year."""
    races = load_races(year); names = load_drivers()
    by = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(MARKET)):
        if not r["race_key"].startswith(str(year)):
            continue
        slug = r["race_key"][5:]
        gp = GP_MAP.get(slug)
        if gp is None or gp not in races:
            continue
        cands = names.get(_norm(r["driver"]), set()) & season_drivers
        if len(cands) != 1:
            continue
        by[gp][r["snapshot_ts"]][next(iter(cands))] = float(r["consensus_prob"])
    out = defaultdict(list)
    for gp, snaps in by.items():
        info = races[gp]
        for ts, probs in snaps.items():
            t = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            # f1db carries no qualifying time for these seasons; qualifying is
            # the afternoon before the race in every one of them
            quali = info["quali"] or (info["race"] - timedelta(days=1) if info["race"] else None)
            if info["race"] and t > info["race"]:
                phase = "post_race"
            elif quali and t > quali:
                phase = "post_quali"
            else:
                phase = "pre_quali"
            out[info["round"]].append((phase, ts, probs))
    return out


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    from f1db_load import load_season
    for year in (2013, 2014, 2015):
        season = load_season(year)
        ids = {d for w in season for d in w["drivers"]}
        snaps = snapshots(year, ids)
        phases = defaultdict(int); cover = 0
        for rnd, lst in snaps.items():
            for ph, ts, probs in lst:
                phases[ph] += 1
            cover += 1
        n_q3 = sum(1 for w in season if w["q3_order"])
        print(f"{year}: {len(season)} weekends, {n_q3} with Q3 times, {cover} with odds; snapshots by phase {dict(phases)};",
              "drivers matched per snapshot:", sorted({len(p) for lst in snaps.values() for _, _, p in lst})[:3], "...")
