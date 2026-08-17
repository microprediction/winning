% ------------------------------------------------------------------------
% Initial parameter fitting code for Experimnet 2 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The model fits parameters when people only attend to a specific number
% of items (symbols). The number of items attneded is assigned 
% in 'nItem' variable. The items attended are selected randomly.
%
% The code imports 'dataForModeling' data and fits mean activation levels
% and a lapse rate for each subject.The code will save 'fitting_1' 
% file under the '.../Experiment 2/data/fitting results/attention_extended'
% folder. To avoid overwrite the line 40 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Sep.14.2019.
% ------------------------------------------------------------------------

clear all, clc
global data_sub

data_sub.nItem = 2; % number of items attended: either 2 or 3

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
load([dataPath '/dataForModeling.mat'])
savingPath = [fileparts(dataPath) '/fitting results/attention_extended/'];
if ~exist(savingPath), mkdir(savingPath), end

% Loop over all subjects
for subNum = 1:size(data.respPattern_cond1,1)
    disp('----------')
    disp(['Fitting subject ' num2str(subNum)])
    disp('----------')
    
    % Make the data visible to all functions
    data_sub.respPattern_cond1 = squeeze(data.respPattern_cond1(subNum,:,:));
    data_sub.respPattern_cond2 = squeeze(data.respPattern_cond2(subNum,:,:,:));
    
    % Fit the model and save the results
    [params(:,subNum), logL(subNum), modelFit{subNum}] = fitOneSub_attention;    
%     save([savingPath 'fitting_1_attend' num2str(data_sub.nItem)], 'params', 'logL', 'modelFit');
end