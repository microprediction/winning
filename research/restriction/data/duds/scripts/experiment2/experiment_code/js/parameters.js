
var AutomaticResponse 	= 0;		// runs alone, for debugging
var Level 		 		= 1;
var DataToSave 			= [];

var sc 					= [];
sc.maxtrials    		= [];		// the maximum number of trials
sc.whichisbigger		= [];		// circle or square
sc.response				= [];		// subject's response (1 to 3)
sc.correct				= [];
sc.trial 		    	= 0;		// global trial counter
sc.nstim        		= [];		// number of items to display
sc.stimvalues   		= [];		// for the area differences to be stored
sc.tomultiply   		= [];		// to multiply stimval for Area3
sc.whichisbigger		= [];		// bigger the square or the circle

const T1 					= 800;		// stimuli onset 800
const T2 					= 1800;		// hide stimuli 1800
const CT 					= 10; 		// number of correct trials to increase level 
const NREP                  = 2;        // number of repetition

const numberofstimuli     = [2, 3, 4, 5];           // number of items to display each trial
const bigger              = [1, 2];                 // bigger the square or the circle
const areavalues          = [0.7, 0.8, 0.9, 0.93, 0.95];  // Nico [0.6, 0.7, 0.8, 0.9, 0.95]; area difference between figures. area2 = area1 * areavalues[i] 
const areaforthethirdone  = [0.1, 0.2, 0.3, 0.4, 0.5 , 0.6];   // Nico [0.1, 0.2, 0.3, 0.4, 0.5 , 0.6]; area difference between figures. area3 = area2 * areavalues[i]

var Trials_save         = [];
var Angle_save          = [];
var StepAngle_save      = [];
var Area1_save          = [];
var Area2_save          = [];
var Area3_save          = [];
var Area4_save          = [];
var Area5_save          = [];
var ThirdSquareOrCircle_save  = [];
var FourthSquareOrCircle_save = [];
var FifthSquareOrCircle_save  = [];
var BiggerCircleOrSquare_save = [];
var PosBig_save         = [];
var StimVal_save        = [];
var StimVal3_save       = [];
var Nalternativas_save  = [];
var RT_type1_save       = [];
var Response_save       = [];
var Correct_save        = [];
var Confidence_save     = [];
var RT_Confidence_save  = [];

