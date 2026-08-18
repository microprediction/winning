# Recognition memory, nested foil sets

Utochkin, Azarov and Grigorev, "Invariant recognition memory spaces for real-world
objects revealed with signal-detection analysis", Psychological Science 2025,
doi 10.1177/09567976251384640. Data and code public at OSF project fap2q, under
"Data and Code"; these are the Experiment 1 trial files plus their README.

Why this matters for the restriction question. The 4AFC trials give choice shares over
{target, foil1, foil2, foil3} with the chosen alternative identified by the response
code (hit, fa1, fa2, fa3). The 2AFC trials give shares over {target, foil_k} for each k
separately, with foil.type naming which foil was shown. Same 120 targets and the same
foil images across both arms, so the smaller menu is a strict subset of the larger with
no matching step and no inferred menu.

The competing alternatives are memory representations of photographs, which is the
psychological case the paper wants: nobody would describe recognising an object as a
race, yet the latent structure is exactly competing identifications.

Two features to handle in analysis. Foil1 is another exemplar of the target's category
while foils 2 and 3 come from a different category, so similarity is manipulated and
independence across alternatives is directly in question. And the two arms are
different participants, 100 in 4AFC against 301 across three 2AFC variants, with
different inclusion thresholds, 35 per cent hits for 4AFC and 60 per cent for 2AFC, so
the comparison is between-subject with a selection difference the null must absorb.

Not yet analysed.
