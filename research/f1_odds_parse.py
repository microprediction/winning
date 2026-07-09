"""Parse archived Oddschecker F1 race-winner pages into a per-race market-probability dataset.

Input : ~/.cache/winning/oddschecker/{timestamp}_{urlencoded-url}.html  (some are gzip-compressed)
Output: ~/.cache/winning/f1_market.csv with columns
        race_key, snapshot_ts, driver, n_bookies, consensus_prob

Method:
  - For each race page (race slug + snapshot year), take the LATEST snapshot whose
    parse yields >= 10 drivers each quoted by >= 3 bookmakers (pre-race quality proxy).
  - Per driver: median across bookmakers of 1/decimal_odds; then normalize within the
    race so probabilities sum to 1 (normalization happens AFTER the median).
  - Fractional odds a/b -> decimal a/b + 1. 'SP', spreads, and empty cells are ignored.

Page eras handled:
  A) 2007-2015 "eventTableRow" tables: fractional odds as cell text; from ~2010 cells
     also carry a dodds_bslip decimal attribute (preferred when present).
  B) 2016-2025 "diff-row" tables: decimal odds in data-odig cell attributes
     (data-bname holds the driver name). 2021+ snapshots are gzip-compressed on disk.
  C) Expired-market snapshots (common 2010-2011) render no odds table and are skipped
     by the quality gate.
"""

import csv
import gzip
import html as htmllib
import os
import re
import statistics
import urllib.parse
from collections import defaultdict

CACHE_DIR = os.path.expanduser('~/.cache/winning/oddschecker')
OUT_CSV = os.path.expanduser('~/.cache/winning/f1_market.csv')

MIN_DRIVERS = 10          # quality gate: at least this many drivers ...
MIN_BOOKIES = 3           # ... each with at least this many bookmaker quotes

EXCLUDE_IN_NAME = ('championship', 'specials', 'each-way')
EXCLUDE_SEGMENTS = ('sprint', 'qualif', 'double', 'bet-history', 'exchanges',
                    'spreads', 'best-odds', 'w-o')

# ---------------------------------------------------------------------------
# filename -> (snapshot_ts, race_slug) or None
# ---------------------------------------------------------------------------

def classify(fname):
    """Return (ts, race_slug) for a race win-market page, else None."""
    if len(fname) < 16 or not fname[:14].isdigit() or fname[14] != '_':
        return None
    ts = fname[:14]
    url = urllib.parse.unquote(fname[15:])
    if url.endswith('.html'):
        url = url[:-5]
    low = fname.lower()
    if any(k in low for k in EXCLUDE_IN_NAME):
        return None
    # strip scheme/host and query string
    path = re.sub(r'^https?://[^/]+', '', url).split('?')[0]
    segs = [s for s in path.strip('/').split('/') if s]
    if len(segs) < 2 or segs[-1] not in ('win-market', 'winner'):
        return None
    race = segs[-2]
    if not ('grand-prix' in race or race.endswith('-gp')):
        return None
    if any(k in race for k in EXCLUDE_SEGMENTS):
        return None
    if race.endswith('-gp'):
        race = race[:-3] + '-grand-prix'
    return ts, race


# ---------------------------------------------------------------------------
# odds-text -> decimal odds
# ---------------------------------------------------------------------------

_FRAC_RE = re.compile(r'^(\d+)/(\d+)$')

def to_decimal(text):
    """Convert a displayed odds string to decimal odds, or None."""
    t = text.strip()
    if not t or t.upper() == 'SP':
        return None
    if t.lower() in ('evs', 'evens'):
        return 2.0
    m = _FRAC_RE.match(t)
    if m:
        b = int(m.group(2))
        return int(m.group(1)) / b + 1.0 if b else None
    if re.match(r'^\d+$', t):                 # bare integer n means n/1
        return float(t) + 1.0
    if re.match(r'^\d+\.\d+$', t):            # exchange display: already decimal
        v = float(t)
        return v if v > 1.0 else None
    return None                                # spreads ("66-70"), blanks, junk


# ---------------------------------------------------------------------------
# HTML -> {driver: [decimal odds, ...]}
# ---------------------------------------------------------------------------

_TR_RE = re.compile(
    r'<tr[^>]*class="[^"]*(?:diff-row|eventTableRow|evTabRow)[^"]*"(.*?)</tr>',
    re.S)
_BNAME_RE = re.compile(r'data-bname="([^"]*)"')
_SELTXT_RE = re.compile(r'<(?:a|span)[^>]*class="[^"]*selTxt[^"]*"[^>]*>([^<]+)<')
_DATANAME_RE = re.compile(r'data-name="([^"]+)"')
_SEL_ANCHOR_RE = re.compile(
    r'class="sel(?:ections)?[^"]*".*?<a[^>]*>(?!Show Graph<)([^<]+)</a>', re.S)
_HIDDEN_NAME_RE = re.compile(r'id="\d+_name"[^>]*>([^<]+)<')
_ODIG_RE = re.compile(r'data-odig="([0-9.]+)"')
_DODDS_RE = re.compile(r'dodds_bslip="([0-9.]+)"')
_CELL_RE = re.compile(
    r'<td[^>]*id="\d+_[A-Za-z0-9]{2,3}"[^>]*class="oo?(?:\s[^"]*)?"([^>]*)>(.*?)</td>',
    re.S)
_TAG_RE = re.compile(r'<[^>]+>')


def parse_rows(page):
    """Extract {driver_name: [decimal_odds]} from one page's HTML."""
    out = defaultdict(list)
    for m in _TR_RE.finditer(page):
        row = m.group(0)
        # --- driver name, as printed on the page ---
        name = None
        bm = _BNAME_RE.search(row)
        if bm and bm.group(1).strip():
            name = bm.group(1)
        if name is None:
            sm = (_SELTXT_RE.search(row) or _DATANAME_RE.search(row)
                  or _SEL_ANCHOR_RE.search(row))
            if sm and sm.group(1).strip():
                name = sm.group(1)
        if name is None:
            hm = _HIDDEN_NAME_RE.search(row)
            if hm and hm.group(1).strip():
                name = hm.group(1)
        if name is None:
            continue
        name = htmllib.unescape(name).strip()

        # --- odds ---
        odds = []
        odigs = _ODIG_RE.findall(row)
        if odigs:                              # diff-row era: decimal attributes
            odds = [float(v) for v in odigs if float(v) > 1.0]
        else:                                  # classic era: table cells
            for attrs, cell in _CELL_RE.findall(row):
                dm = _DODDS_RE.search(attrs)
                if dm:                         # ~2010-2012: decimal attribute
                    v = float(dm.group(1))
                    if v > 1.0:
                        odds.append(v)
                    continue
                v = to_decimal(_TAG_RE.sub('', htmllib.unescape(cell)))
                if v is not None:
                    odds.append(v)
        out[name].extend(odds)
    return dict(out)


def slug_fraction(page, slug):
    """Fraction of odds-table rows mentioning the race slug.

    On a live race page every row's bet-history link contains the race slug.
    When a race market expires, oddschecker served the generic F1 page whose
    odds table is the Drivers' Championship market: same drivers, no slug in
    the rows. Those must not be mistaken for race odds.
    """
    rows = [m.group(0) for m in _TR_RE.finditer(page)]
    if not rows:
        return 0.0
    return sum(1 for r in rows if slug in r) / len(rows)


def read_page(path):
    with open(path, 'rb') as fh:
        raw = fh.read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return raw.decode('utf-8', 'replace')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    by_race = defaultdict(list)                # race_key -> [(ts, fname, slug)]
    for fname in os.listdir(CACHE_DIR):
        info = classify(fname)
        if info is None:
            continue
        ts, race = info
        by_race[f'{ts[:4]}_{race}'].append((ts, fname, race))

    chosen_by_race, stats, failures = {}, defaultdict(lambda: [0, 0, 0]), []
    for race_key in sorted(by_race):
        stats[race_key[:4]][0] += 1            # candidate races
        for ts, fname, slug in sorted(by_race[race_key], reverse=True):
            page = read_page(os.path.join(CACHE_DIR, fname))
            if slug_fraction(page, slug) < 0.5:
                continue                       # expired market / generic page
            drivers = parse_rows(page)
            good = {d: o for d, o in drivers.items() if len(o) >= MIN_BOOKIES}
            if len(good) >= MIN_DRIVERS:
                chosen_by_race[race_key] = (ts, drivers)
                break
        else:
            failures.append(race_key)

    # --- season-consistency guard -----------------------------------------
    # Some archived captures are misdated (the wayback fetch served a snapshot
    # from a different year). The set of driver names is a season fingerprint:
    # drop races whose grid matches another year's races much better than
    # their own year's. Team tags are stripped because some page variants
    # print "Adrian Sutil" and others "Adrian Sutil (For)".
    def fingerprint(drivers):
        out = set()
        for d in drivers:
            n = re.sub(r'\s*\([^)]*\)\s*$', '', d).lower()
            if ',' in n:                       # "Barrichello, Rubens" style
                last, _, first = n.partition(',')
                n = f'{first.strip()} {last.strip()}'
            if n and 'any other' not in n:
                out.add(n)
        return out

    vocab = defaultdict(lambda: defaultdict(int))   # year -> name -> n races
    for race_key, (_, drivers) in chosen_by_race.items():
        for n in fingerprint(drivers):
            vocab[race_key[:4]][n] += 1
    misdated = []
    for race_key, (_, drivers) in chosen_by_race.items():
        year, fp = race_key[:4], fingerprint(drivers)
        if not fp:
            continue
        def overlap(y):
            v, own = vocab[y], (1 if y == year else 0)
            return sum(1 for n in fp if v.get(n, 0) > own) / len(fp)
        own_score = overlap(year)
        others = {y: overlap(y) for y in vocab if y != year}
        if not others:
            continue
        best_year, best = max(others.items(), key=lambda kv: kv[1])
        alone = len([k for k in chosen_by_race if k[:4] == year]) == 1
        if alone:
            # no same-year race to corroborate: only an (almost) exact match
            # to another season's grid is damning, since adjacent seasons
            # legitimately share most drivers
            if best >= 0.9:
                misdated.append((race_key, best_year, own_score, best))
        elif own_score < 0.5 and best > own_score + 0.3:
            misdated.append((race_key, best_year, own_score, best))
    for race_key, *_ in misdated:
        del chosen_by_race[race_key]

    rows = []
    for race_key, (ts, drivers) in sorted(chosen_by_race.items()):
        year = race_key[:4]
        snapshot_ts = (f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}T'
                       f'{ts[8:10]}:{ts[10:12]}:{ts[12:14]}')
        med = {d: statistics.median(1.0 / v for v in o)
               for d, o in drivers.items() if o}
        total = sum(med.values())
        stats[year][1] += 1
        stats[year][2] += len(med)
        for d in sorted(med, key=med.get, reverse=True):
            rows.append({'race_key': race_key,
                         'snapshot_ts': snapshot_ts,
                         'driver': d,
                         'n_bookies': len(drivers[d]),
                         'consensus_prob': round(med[d] / total, 6)})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['race_key', 'snapshot_ts', 'driver',
                                           'n_bookies', 'consensus_prob'])
        w.writeheader()
        w.writerows(rows)

    print(f'wrote {len(rows)} rows to {OUT_CSV}')
    print('year  races_ok/candidates  mean_drivers')
    for y in sorted(stats):
        cand, ok, ndrv = stats[y]
        mean = ndrv / ok if ok else 0.0
        print(f'{y}    {ok:3d}/{cand:<3d}            {mean:5.1f}')
    if misdated:
        print(f'\n{len(misdated)} race pages dropped as misdated captures '
              '(driver grid matches another season):')
        for rk, by, own, best in misdated:
            print(f'   {rk}: grid matches {by} (overlap {best:.0%} vs '
                  f'own-year {own:.0%})')
    if failures:
        print(f'\n{len(failures)} race pages had no snapshot passing the '
              f'>= {MIN_DRIVERS} drivers x >= {MIN_BOOKIES} bookies gate:')
        for rk in failures:
            print('  ', rk)


if __name__ == '__main__':
    main()
