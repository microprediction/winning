% ------------------------------------------------------------------------
% Additional parameter fitting code for Experimnet 1 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% After first fitting the paramters, we ran fitting processes more 
% based on the latest fitted parameters. The code will save 'fitting_N' 
% file under the '.../Experiment 1/data/fitting results/extended'
% folder ('N' is the number of current fitting process). To avoid 
% overwriting the line 39 is currently commented. To run this code, 
% locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------

clear all, clc
global data_sub

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
load([dataPath '/dataForModeling.mat'])

savingPath = [fileparts(dataPath) '/fitting results/extended/'];
fittingDataList = dir([savingPath 'fitting*.mat']);
load([savingPath '/' fittingDataList(end).name]);
numFit = length(fittingDataList);

% Loop over all subjects
for subNum = 1:size(data.respPattern_cond1,1)
    disp('----------')
    disp(['Fitting subject ' num2str(subNum)])
    disp('----------')
    
    % Make the data visible to all functions
    data_sub.respPattern_cond1 = squeeze(data.respPattern_cond1(subNum,:,:));
    data_sub.respPattern_cond2 = squeeze(data.respPattern_cond2(subNum,:,:,:));
    startingParamSet = params(:,subNum)';
    
    % Fit the model and save the results
    [params(:,subNum), logL(subNum), modelFit{subNum}] = fitOneSub_extended(startingParamSet);
%     save([savingPath 'fitting_' num2str(numFit+1)], 'params', 'logL', 'modelFit');
end