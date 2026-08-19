#!/usr/bin/env python3
"""Convert the Yeon & Rahnev (2020, Nat Commun 11:3857) OSF MATLAB files into
tidy trial-level CSVs plus long-form versions of the aggregate count matrices
the authors actually fit.

Requires: numpy, scipy.  Run from anywhere:  python3 build_tidy.py
"""
import csv, glob, os, re, sys
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
OSF = os.path.join(HERE, "osf", "d2b9v")
OUT = os.path.join(HERE, "tidy")
os.makedirs(OUT, exist_ok=True)

def load(path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)

def write(name, header, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"{name}: {len(rows)} rows, {os.path.getsize(p)} bytes")

# ---------------------------------------------------------------- Experiment 1
# condition 1 = 4-alternative; 2 = 2-alternative (menu revealed AFTER offset);
# 3 = 2-alternative (menu announced in advance / "advance warning")
rows = []
for s in range(1, 33):
    p = load(f"{OSF}/Experiment 1/data/subject_responses/raw responses/results_s{s}.mat")["p"]
    D = np.array(p.data)
    for run in range(D.shape[0]):
        for blk in range(D.shape[1]):
            b = D[run, blk]
            cond = int(b.condition)
            resp = np.atleast_1d(b.response).ravel()
            dom  = np.atleast_1d(b.correctColor).ravel()
            conf = np.atleast_1d(b.confidence).ravel()
            corr = np.atleast_1d(b.correct).ravel()
            rt   = np.atleast_2d(b.rt)
            alt  = np.atleast_1d(b.wrongColor).ravel() if hasattr(b, "wrongColor") else None
            for t in range(len(resp)):
                a = int(alt[t]) if alt is not None else ""
                rows.append([s, run + 1, blk + 1, cond, t + 1, int(dom[t]), a,
                             int(resp[t]), int(corr[t]), int(conf[t]),
                             round(float(rt[t, 0]), 4), round(float(rt[t, 1]), 4)])
write("exp1_trials.csv",
      ["subject","run","block","condition","trial_in_block","dominant_color",
       "alternative_color","response","correct","confidence","rt_choice","rt_confidence"], rows)
exp1_rows = rows

# ---------------------------------------------------------------- Experiment 2
rows = []
for s in range(1, 11):
    for sess in (1, 2, 3):
        p = load(f"{OSF}/Experiment 2/data/subject_responses/raw responses/sub{s}_{sess}.mat")["p"]
        opt = np.array([np.atleast_1d(x).ravel() for x in np.atleast_1d(p.optionN2)])  # run x block
        M = np.atleast_1d(p.main)
        for i, b in enumerate(M):
            run, blk = i // 4 + 1, i % 4 + 1
            n_opt = int(b.NchoiceOption)
            resp = np.atleast_1d(b.response).ravel()
            dom  = np.atleast_1d(b.targetOrder).ravel()
            corr = np.atleast_1d(b.correct).ravel()
            rt   = np.atleast_1d(b.rt).ravel()
            alt  = np.atleast_1d(b.pairOrder).ravel() if hasattr(b, "pairOrder") else None
            for t in range(len(resp)):
                a = int(alt[t]) if alt is not None else ""
                rows.append([s, sess, run, blk, t + 1, n_opt, int(dom[t]), a,
                             int(resp[t]), int(corr[t]), round(float(rt[t]), 4)])
write("exp2_trials.csv",
      ["subject","session","run","block","trial_in_block","n_options","dominant_symbol",
       "alternative_symbol","response","correct","rt"], rows)

# ---------------------------------------------------------------- Experiment 3
rows = []
for s in range(1, 11):
    for sess in (1, 2, 3):
        p = load(f"{OSF}/Experiment 3/data/subject_responses/raw responses/sub{s}_{sess}.mat")["p"]
        M = np.atleast_1d(p.main)
        for i, b in enumerate(M):
            run, blk = i // 4 + 1, i % 4 + 1
            resp = np.atleast_2d(b.response)
            corr = np.atleast_2d(b.correct)
            dom  = np.atleast_1d(b.targetOrder).ravel()
            rt   = np.atleast_2d(b.rt)
            for t in range(resp.shape[0]):
                r2 = int(resp[t, 1]); c2 = int(corr[t, 1])
                rows.append([s, sess, run, blk, t + 1, int(dom[t]),
                             int(resp[t, 0]), int(corr[t, 0]),
                             r2 if r2 != 0 else "", c2 if c2 != 99 else "",
                             round(float(rt[t, 0]), 4), round(float(rt[t, 1]), 4)])
write("exp3_trials.csv",
      ["subject","session","run","block","trial_in_block","dominant_symbol",
       "response1","correct1","response2","correct2","rt1","rt2"], rows)

# ---------------------------------------------------------------- Experiment 4
rows = []
for s in range(1, 12):
    for sess in (1, 2, 3):
        f = f"{OSF}/Experiment 4/data/subject_responses/raw responses/sub{s}_{sess}.mat"
        p = load(f)["p"]
        lc  = np.array(p.limitChoices)
        resp = np.array(p.responses); corr = np.array(p.correct)
        ca   = np.array(p.correctAnswer); rtm = np.array(p.response_time)
        du   = np.array(p.direction_used)
        prop = float(np.array(p.proportions).ravel()[0])
        ch   = np.array(p.choices) if hasattr(p, "choices") else None
        for run in range(5):
            row = 0 if (run + 1) % 2 == 1 else 1
            for blk in range(4):
                n_opt = 2 if lc[row, blk] == 1 else 3
                for t in range(50):
                    if ch is not None:
                        alt, exc = int(ch[run, blk, t, 1]), int(ch[run, blk, t, 2])
                    else:
                        alt, exc = "", ""
                    rows.append([s, sess, run + 1, blk + 1, t + 1, n_opt,
                                 int(ca[run, blk, t]), alt, exc,
                                 int(resp[run, blk, t]), int(corr[run, blk, t]),
                                 round(float(rtm[run, blk, t]), 4), prop,
                                 *[round(float(np.degrees(du[run, blk, t, k])) % 360, 3) for k in range(3)]])
write("exp4_trials.csv",
      ["subject","session","run","block","trial_in_block","n_options","dominant_label",
       "alternative_label","excluded_label","response","correct","rt","prop_dominant_dots",
       "dir1_deg","dir2_deg","dir3_deg"], rows)

# ------------------------------------------- aggregate count matrices (as fit)
d = load(f"{OSF}/Experiment 1/data/subject_responses/dataForModeling.mat")["data"]
a = np.array(d.respPattern_cond1)
write("exp1_full_menu_counts.csv", ["subject","dominant_color","response","n"],
      [[s+1,i+1,j+1,int(a[s,i,j])] for s in range(a.shape[0]) for i in range(4) for j in range(4)])
a = np.array(d.respPattern_cond2)
write("exp1_pair_menu_counts.csv", ["subject","dominant_color","alternative_color","n_correct","n_wrong"],
      [[s+1,i+1,j+1,int(a[s,i,j,0]),int(a[s,i,j,1])] for s in range(a.shape[0]) for i in range(4) for j in range(4) if i!=j])
# Condition 3 is the same two-alternative menu ANNOUNCED BEFORE the stimulus, so the
# observer can allocate attention to the pair. It is the control arm for condition 2 and
# the authors' dataForModeling.mat does not save it, so it is counted from the trials.
# Counting condition 2 the same way reproduces respPattern_cond2 exactly; that check runs
# below and raises if it ever fails.
def pair_counts(cond):
    d = {}
    for r in exp1_rows:
        if r[3] != cond:
            continue
        s, i, j, ok = r[0], r[5], r[6], r[8]
        c = d.setdefault((s, i, j), [0, 0])
        c[0 if ok else 1] += 1
    return d

check = pair_counts(2)
for s in range(a.shape[0]):
    for i in range(4):
        for j in range(4):
            if i != j:
                got = check.get((s + 1, i + 1, j + 1), [0, 0])
                assert got == [int(a[s, i, j, 0]), int(a[s, i, j, 1])], (s, i, j, got)
print("exp1 condition 2 recounted from trials matches respPattern_cond2")

c3 = pair_counts(3)
write("exp1_pair_menu_advance_counts.csv",
      ["subject","dominant_color","alternative_color","n_correct","n_wrong"],
      [[s+1,i+1,j+1,*c3.get((s+1,i+1,j+1),[0,0])]
       for s in range(a.shape[0]) for i in range(4) for j in range(4) if i!=j])

d = load(f"{OSF}/Experiment 2/data/subject_responses/dataForModeling.mat")["data"]
a = np.array(d.respPattern_cond1)
write("exp2_full_menu_counts.csv", ["subject","dominant_symbol","response","n"],
      [[s+1,i+1,j+1,int(a[s,i,j])] for s in range(a.shape[0]) for i in range(6) for j in range(6)])
a = np.array(d.respPattern_cond2)
write("exp2_pair_menu_counts.csv", ["subject","dominant_symbol","alternative_symbol","n_correct","n_wrong"],
      [[s+1,i+1,j+1,int(a[s,i,j,0]),int(a[s,i,j,1])] for s in range(a.shape[0]) for i in range(6) for j in range(6) if i!=j])

d = load(f"{OSF}/Experiment 3/data/subject_responses/dataForModeling.mat")["data"]
a = np.array(d.respPattern_cond1)
write("exp3_full_menu_counts.csv", ["subject","dominant_symbol","response1","n"],
      [[s+1,i+1,j+1,int(a[s,i,j])] for s in range(a.shape[0]) for i in range(6) for j in range(6)])
a = np.array(d.respPattern_cond2)
write("exp3_second_answer_counts.csv", ["subject","dominant_symbol","response1","response2","n"],
      [[s+1,i+1,j+1,k+1,int(a[s,i,j,k])] for s in range(a.shape[0]) for i in range(6)
       for j in range(6) for k in range(6) if int(a[s,i,j,k])>0])
