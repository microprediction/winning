% ------------------------------------------------------------------------
% Initial parameter fitting code for Experimnet 2 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code imports 'dataForModeling' data and fits mean activation levels
% and a lapse rate for each subject.The code will save 'fitting_1' 
% file under the '.../Experiment 2/data/fitting results/simple'
% folder. To avoid overwriting, the line 36 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------

clear, clc
global data_sub

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
load([dataPath '/dataForModeling.mat'])
savingPath = [fileparts(dataPath) '/fitting results/simple/'];

% Loop over all subjects
for subNum = 1:size(data.respPattern_cond1,1)
    disp('----------')
    disp(['Fitting subject ' num2str(subNum)])
    disp('----------')
    
    % Make the data visible to all functions
    cond1 = squeeze(data.respPattern_cond1(subNum,:,:));
    cond2 = squeeze(data.respPattern_cond2(subNum,:,:,:));
    
    data_sub.respPattern_cond1 = cond1;
    data_sub.respPattern_cond2 = cond2;
    
    % Fit the model and save the results
    [params(:,subNum), logL(subNum), modelFit{subNum}] = fitOneSub_simple;
%     save([savingPath '/fitting_1'], 'params', 'logL', 'modelFit');
end