% ------------------------------------------------------------------------
% Data clean up code for Experiment 1 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code imports subjects' behavioral responses and organize data.
% In the end, the code will save 'dataForModeling.mat' file in 
% '.../Experiment 1/data/subject_responses'. To avoid overwrite on the
% existing file, the line 84 is currently commented. To run this code, 
% locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.22.2019.
% ------------------------------------------------------------------------

clear all, clc

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses/raw responses'];
savingName = [fileparts(dataPath) '/dataForModeling'];
fileList = dir([dataPath, '/results*']);

for subject_num = 1:length(fileList)    
    load([dataPath, '/results_s' num2str(subject_num) '.mat']);
    
    % Define some variables
    stimulus = {[], [], []};
    response = {[], [], []};
    accuracy = {[], [], []};
    correctColor_cond2 = [];
    wrongColor_cond2 = [];
    correctColor_cond3 = [];
    wrongColor_cond3 = [];
    
    % Go through all runs and blocks
    for run=1:p.total_runs
        for block=1:p.blocks_per_run
            stimulus{p.data{run,block}.condition}(end+1:end+p.trials_per_block) = p.data{run,block}.correctColor;
            response{p.data{run,block}.condition}(end+1:end+p.trials_per_block) = p.data{run,block}.response;
            accuracy{p.data{run,block}.condition}(end+1:end+p.trials_per_block) = p.data{run,block}.correct;
            if p.data{run,block}.condition == 2
                correctColor_cond2(end+1:end+p.trials_per_block) = p.data{run,block}.correctColor;
                wrongColor_cond2(end+1:end+p.trials_per_block) = p.data{run,block}.wrongColor;
            elseif p.data{run,block}.condition == 3 
                correctColor_cond3(end+1:end+p.trials_per_block) = p.data{run,block}.correctColor;
                wrongColor_cond3(end+1:end+p.trials_per_block) = p.data{run,block}.wrongColor;
            end
        end
    end
    
    % Compute accuracy for each condition
    for cond=1:3
        acc(subject_num,cond) = mean(accuracy{cond});
    end
    
    % Compute response pattern for modeling
    for stimPresented=1:4
        
        % Condition 1 (4 choices)
        for stimResponded=1:4
            respPattern_cond1(subject_num,stimPresented,stimResponded) = ...
                sum(stimulus{1}==stimPresented & response{1}==stimResponded);
        end
        
        % Condition 2 (2 choices know after)
        for stimPair=1:4
            for correct=1:2 %1: correct, 2: wrong
                respPattern_cond2(subject_num,stimPresented,stimPair,correct) = ...
                    sum(stimulus{2}==stimPresented & wrongColor_cond2==stimPair & accuracy{2}==2-correct);
            end
        end
        
        % Condition 3 (2 choices know before)
        for stimPair=1:4
            for correct=1:2
                respPattern_cond3(subject_num,stimPresented,stimPair,correct) = ...
                    sum(stimulus{3}==stimPresented & wrongColor_cond3==stimPair & accuracy{3}==2-correct);
            end
        end
    end
end

%% Save data for modeling
data.respPattern_cond1 = respPattern_cond1;
data.respPattern_cond2 = respPattern_cond2;
data.accuracy = acc;
% save(savingName, 'data')