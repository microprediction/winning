% ------------------------------------------------------------------------
% Initial parameter fitting code for Experimnet 1 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code imports 'dataForModeling' data and fits mean activation levels
% and a lapse rate for each subject.The code will save 'fitting_1' 
% file under the '.../Experiment 1/data/fitting results/extended'
% folder. To avoid overwrite the line 33 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------

clear all, clc
global data_sub

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
load([dataPath '/dataForModeling.mat'])
savingPath = [fileparts(dataPath) '/fitting results/extended/'];

% Loop over all subjects
for subNum = 1:size(data.respPattern_cond1,1)
    disp('----------')
    disp(['Fitting subject ' num2str(subNum)])
    disp('----------')
    
    % Make the data visible to all functions
    data_sub.respPattern_cond1 = squeeze(data.respPattern_cond1(subNum,:,:));
    data_sub.respPattern_cond2 = squeeze(data.respPattern_cond2(subNum,:,:,:));
    
    % Fit the model and save the results
    [params(:,subNum), logL(subNum), modelFit{subNum}] = fitOneSub_extended;
%     save([savingPath 'fitting_1'], 'params', 'logL', 'modelFit');
end