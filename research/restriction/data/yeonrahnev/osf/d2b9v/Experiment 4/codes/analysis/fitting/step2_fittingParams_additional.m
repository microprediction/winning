% ------------------------------------------------------------------------
% Additional parameter fitting code for Experimnet 4 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% After first fitting the paramters, we ran fitting processes more 
% based on the latest fitted parameters. The code will save 'fitting_N' 
% file under the '.../Experiment 4/data/fitting_results/' folder 
% ('N' is the number of current fitting process). To avoid 
% overwriting the line 39 is currently commented. To run this code, 
% locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Sep.14.2019.
% ------------------------------------------------------------------------

clear all, clc
global data_sub

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data'];
load([dataPath '/subject_responses/dataForModeling.mat']);
savingPath = [dataPath '/fitting results'];
nSub = length(data.proportion_used);

fittingLists = dir([savingPath '/fitting_*.mat']);
load([savingPath '/' fittingLists(end).name]);
numFit = length(fittingLists);
lapse = 0;

for sub = 1:nSub
    disp('----------')
    disp(['Fitting subject ' num2str(sub)])
    disp('----------')
    data_sub.c3 = data.c3.correct(:,sub);
    data_sub.c2 = data.c2.correct(:,sub);
    
    startingParamSet = params(1,sub)';
    startingParamSet = [startingParamSet; lapse];
    
    [params(:,sub), logL(sub), modelFit{sub}] = fitOneSub(startingParamSet);
%     save([savingPath '/fitting_' num2str(numFit+1)], 'params', 'logL', 'modelFit');
end