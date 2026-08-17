# confidence_dud

This repository contains all the experimental data and scripts used for the study "_The presence of irrelevant alternatives paradoxically increases confidence in perceptual decisions_" (Comay, Della Bella, Lamberti, Sigman, Solovey & Barttfeld; 2023). Publication available [here](https://www.sciencedirect.com/science/article/pii/S0010027723000112?dgcid=coauthor). Preprint available [here](https://psyarxiv.com/zq7gw). 

## Usage

In order to run the scripts you'll have to download the experimental data. All scripts are in R language, and in order to be runned you will need to install 3 libraries: "tidyverse" and "DirichletReg". To do that, simply type in the R console the command "_install.packages()_" with the name of the library as a parameter. 

In the 'scripts' folder there are 6 sub-folders. First 5 contains the source html/js code of the experiments, the .jzip archives to directly upload the experiments to JATOS and scripts that reproduce the main figures of the paper. In sub-folder 'modeling' you'll find the code that performs the model fitting and simulations of the 4 models proposed for this study: the max model, the diff model, the contrast model and the average residual model. In order to perform model fitting simply run the wrapper.R file.

## How is data organized 

In folder "data" you will find the data for the 5 experiments. Rows are trials, and the columns are organized as follows:

#### Experiments 1, 2 & 4

- _n3SquareOrCircle:_ indicates if the 3rd alternative was a square (1), a circle (2) or was not present (0).
- _BiggerCircleOrSquare:_ indicates if the biggest figure was a circle (1) or a square (2)
- _PosBig:_ this variable is the same as the former, 1 means that the biggest figure was a circle and 2 that was a square. Was used mostly for debugging.
- _StimVal:_ indicates the sizes of the second biggest stimulus, in proportions to the biggest figure. Is used to indicate the levels of task difficulty.
- _StimVal3:_ indicates the sizes of the third biggest stimulus, in proportions to the second biggest figure. 
- _Nalternativas:_ indicates the number of alternatives present. 
- _RT_type1:_ indicates the response time (in ms) of the type 1 task (the perceptual decision).
- _Response:_ indicates which response made the subject: a circle (1), a square (2) or the third alternative (3).
- _Correct:_ boolean indicator for correct (True) and incorrect (False) responses. 
- _Confidence:_ reported confidence in a scale from 0 to 100.
- _RT_confidence:_ indicates the response time (in ms) of the type 2 task (confidence report).
- _N_sujeto:_ indicates the subject's number.
- _binary_correct:_ a binary variable for correct (1) and incorrect (0) responses.
- _Trial_number:_ trial number from 1 to 120 in experiment 1 and from to 480 in experiment 2.
- _Angle:_ random angle for stimuli arrangement. 
- _Step_Angle:_ a rotation of +/-120 degrees of stimuli arrangement. Is expressed in radians (+/- 2.0943...) 
- _Area1:_ size of stimulus 1 (the final size will depend on the screen size).
- _Area2:_ size of stimulus 2 (the final size will depend on the screen size).
- _Area3:_ size of stimulus 3 (the final size will depend on the screen size).
- _Mobile:_ binary indicator, a 1 if the subject performed the experiment on a mobile and a 0 if did it on a computer
- _Code:_ an alphanumeric code that identifies the subject.
- _Gender:_ a variable for gender identification of the participant. f=female; m=male; NoBinario=non binary gender identifications.
- _Age:_ subject's age.

_Note: data from experiment 2 contains more columns because it had more alternatives. For example, there are variables called n4SquareOrCircle and n5SquareOrCircle, indicating if the fourth and fifth alternative was a square (1) or a circle (2) respectively (and 0 if it was not shown). The same is for the other variables, with a 3, 4 or 5 indicating to which alternative refers._ 

#### Experiments 3 & 5

- _distance_ratio:_ the distance of the target to the mean of the closest dot cloud. It can take 3 values: 3, 2.5 and 2.1. A value of 2 means that the target is in the middle of the 2 closest clouds.
- _xtarget:_ x position of target on the screen.
- _ytarget:_ y position of target on the screen.
- _sd:_ standard deviation of clouds' dots (60 as in Li & Ma, 2020).
- _nsamples:_ number of dots on each cloud (375 as in Li & Ma, 2020).
- _color_nube1:_ index for the color of the cloud 1. 0 = red; 1 = green; 2 = blue.
- _color_nube2:_ index for the color of the cloud 2. 0 = red; 1 = green; 2 = blue.
- _color_nube3:_ index for the color of the cloud 3. 0 = red; 1 = green; 2 = blue.
- _color_buttons1:_ index for the color of the button 1. Is always 0 as button 1 was always red.
- _color_buttons2:_ index for the color of the button 2. Is always 1 as button 2 was always green.
- _color_buttons3:_ index for the color of the button 3. Is always 2 as button 3 was always blue.
- _correct_color:_ the color of the correct cloud.
- _correct_cloud:_ the number of the correct cloud.

_The rest of the variables are the same as in experiment 1, 2 & 4._

## Contact

For any doubts or suggestions you can send an e-mail to the corresponding author at nicocomay@gmail.com

