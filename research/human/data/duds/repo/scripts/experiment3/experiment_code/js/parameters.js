var AutomaticResponse 	= 0;		// runs alone, for debugging
var DataToSave 			= [];

var sc 					= [];
sc.maxtrials    		= [];		// the maximum number of trials
sc.response				= [];		// subject's response (1 to 3)
sc.correct				= [];
sc.trial 		    	= 0;		// global trial counter
sc.nstim        		= [];		// number of items to display
sc.dist_12_values 		= [];		// for the area differences to be stored

const T1 				= 800;		// stimuli onset 800
const T2 				= 20800;	// hide stimuli 1800

const numberofstimuli   = [3, 3];   // number of items to display each trial
const NREP              = 20;       // 40 = 240 trials, 20 = 120 trials

//const Likelihood_ratio_1_2 = [1.03,1.04,1.05,1.08, 1.15, 1.6 ]; [1.05, 1.15, 1.3, 1.6, 2.05, 2.2]; 

const Distance_ratio_1_2 = [2.1, 2.5, 3]; 

var Trials_save                 = [];
var Nalternativas_save          = [];
var Distance_ratio_1_2_save     = [];
var X1_save                     = [];
var X2_save                     = [];
var X3_save                     = [];
var Y1_save                     = [];
var Y2_save                     = [];
var Y3_save                     = [];
var Xtarget_save                = [];
var Ytarget_save                = [];
var angle_save                  = [];
var SD_save                     = [];
var Nsamples_save               = [];
var Color_clouds_save           = [];
var Color_buttons_save          = [];
var CorrectColor_save           = [];
var CorrectCloud_save           = [];
var RT_type1_save               = [];
var Response_save               = [];
var nube_save                   = [];
var Correct_save                = [];
var Confidence_save             = [];
var RT_Confidence_save          = [];

