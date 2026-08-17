% ------------------------------------------------------------------------
% Initial parameter fitting code for Experimnet 4 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code imports 'dataForModeling' data and fits mean activation levels
% and a lapse rate for each subject.The code will save 'fitting_1' 
% file under '.../Experiment 4/data/fitting_results' folder,
% To avoid overwrite the line 33 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Sep.14.2019.
% ------------------------------------------------------------------------

clear all, clc
global data_sub 
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data'];
savingPath = [dataPath '/fitting results/'];
if ~exist(savingPath), mkdir(savingPath); end
load([dataPath '/subject_responses/dataForModeling.mat']);
nSub = length(data.proportion_used);
mu = -1.5;
lapse = 0;
startingParamSet = [mu, lapse];

for sub = 1:nSub
    disp('----------')
    disp(['Fitting subject ' num2str(sub)])
    disp('----------')
    data_sub.c3 = data.c3.correct(:,sub);
    data_sub.c2 = data.c2.correct(:,sub);
    
    [params(:,sub), logL(sub), modelFit{sub}] = fitOneSub(startingParamSet);
%     save([savingPath '/fitting_1.mat'], 'params', 'logL', 'modelFit');
end