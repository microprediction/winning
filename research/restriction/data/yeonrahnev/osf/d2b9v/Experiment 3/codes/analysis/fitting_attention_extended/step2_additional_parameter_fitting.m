% ------------------------------------------------------------------------
% Additional parameter fitting code for Experimnet 3 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The model fits parameters when people only attend to a specific number
% of items (symbols). The number of items attneded is assigned 
% in 'nItem' variable. The items attended are selected randomly.
%
% After first fitting the paramters, we ran fitting processes more 
% based on the latest fitted parameters. The code will save 'fitting_N' 
% file under the '.../Experiment 3/data/fitting results/attention_extended'
% folder ('N' is the number of current fitting process). To avoid 
% overwriting, the line 52 is currently commented. To run this code, 
% locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Sep.14.2019.
% ------------------------------------------------------------------------

clear all, clc
global data_sub

data_sub.nItem = 2; % number of items attended: either 2 or 3

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
load([dataPath '/dataForModeling.mat'])

savingPath = [fileparts(dataPath) '/fitting results/attention_extended/'];
fittingDataList = dir([savingPath 'fitting*' num2str(data_sub.nItem) '.mat']);
load([savingPath '/' fittingDataList(end).name]);
numFit = length(fittingDataList);
flag = 1;

while flag
    fittingDataList = dir([savingPath 'fitting*' num2str(data_sub.nItem) '.mat']);
    load([savingPath '/' fittingDataList(end).name]);
    numFit = length(fittingDataList);
    if numFit == 6, flag = 0; end
    
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
        [params(:,subNum), logL(subNum), modelFit{subNum}] = fitOneSub_attention(startingParamSet);
%         save([savingPath 'fitting_' num2str(numFit+1) '_attend' num2str(data_sub.nItem)], 'params', 'logL', 'modelFit');
    end
end
