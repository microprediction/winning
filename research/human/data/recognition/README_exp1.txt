This file explains variable names in the data files (.csv) located in the current folder

WARNING: It is recommended that the current folder with all its content data files are copied under their original names into the folder with R scripts kipping the directory structure as on OSF. This will help avoid issued with opening files


###### TRIAL-BY-TRIAL DATA FILES ########

### File "2afc_alltrials_exp1.csv"
Description: Trial-by-trial data from the 2-AFC groups (only observers with no less than 60% hits are included)
Variables:
	'participantID': participant's identifier
	'test.variant': which version of the 2-AFC task <1, 2, 3> is assigned
	'target': the file name of the target image presented in this trial
	'foil': the file name of the foil image presented in this trial
	'foil.type': which foil type <foil1, foil2, foil3> was presented:
		'foil1' - another exemplar from the same category
		'foil2', 'foil3' - two exemplars from another category
	'response': target choice of foil choice  <hit, fa>


### File "4afc_alltrials_exp1.csv"
Description: Trial-by-trial data from the 4-AFC groups (only observers with no less than 35% hits are included)
Variables:
	'participantID': participant's identifier
	'target': the file name of the target image presented in this trial
	'foil1', 'foil2', 'foil3': the file names of the foil images presented in this trial
	'response': response outcome: choosing target or one of the three foils  <hit, fa1, fa2, fa3> 


###### RESULTS OF SPLIT-HALF CONSISTENCY ANALYSES ########

### Files "2afc_splithalf_exp1.csv" & "4afc_splithalf_exp1.csv"
Description: Correlations and regression parameters between d'-s in slit-half samples across target-foil combinations, 2-AFC or 4-AFC task (10,000 runs per group)
Variables:
	'slope': the slope of a Deming regression model between the half-sample d'-s
	'intercept': the intercept of a Deming regression model between the half-sample d'-s
	'error': a measurement standard error (MSE) estimated by the Deming regression model
	'spearman': Spearman's correlation between half-samples
	'sb': Spearman-Brown corrected correlation extrapolating a full sample correlation
	'permut': Spearman's correlation for shuffled d'-s (permutation test)
	'sb.permut': Spearman-Brown's correlation for shuffled d'-s (permutation test)



###### SDT D' SUMMARY ########

### File "ALL_DPRIME_Exp1.csv"
Description: d'-s for each target-foil pair recovered from 2-AFC (Eq. 1) and 4-AFC (Eq. 2) tasks
Variables:
	'target': the file name of the target image
	'foil': the file name of the foil image
	'foil.type': foil type tested against this target <foil1, foil2, foil3>
	'dpr_2afc': d' recovered from 2-AFC (Eq. 1)
	'dpr_4afc': d' recovered from 4-AFC (Eq. 2)