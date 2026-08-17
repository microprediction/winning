% ------------------------------------------------------------------------
% Data clean up code for Experiment 2 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code imports subjects' behavioral responses and organize data.
% In the end, the code will save 'dataForModeling.mat' file in 
% '.../Experiment 2/data/subject_responses'. To avoid overwrite on the
% existing file, the line 67 is currently commented. To run this code, 
% locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------
clear all, close all, clc

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses/raw responses'];
savingName = [fileparts(dataPath) '/dataForModeling'];
fileList = dir([dataPath, '/sub*']);

sessions = 1:3;
nStim = 6;

for sub = 1:10
    clear resp1 resp2
    options = []; main = [];
    correct1 = []; correct2 = [];
    for ss = sessions        
        filename = ['/sub' num2str(sub), '_' num2str(ss) '.mat'];
        load([dataPath filename])        
        options = [options; cell2mat(p.optionN2)'];
        main = [main; p.main'];
    end
    
    for blocks = 1:length(main)
        for stimPresented = 1:nStim            
            % case 6 choices
            if options(blocks) == 0
                for stimResp = 1:nStim
                    resp1(blocks,stimPresented,stimResp) = sum(main{blocks}.targetOrder' == stimPresented & ...
                        main{blocks}.response == stimResp);
                end
                correct1 = [correct1; main{blocks}.correct];
                
            % case 2 choices
            else
                for stimPair = 1:nStim
                    for correct = 1:2   % 1: correct, 2: wrong
                        resp2(blocks,stimPresented,stimPair,correct) = ...
                            sum(main{blocks}.targetOrder' == stimPresented & ...
                            main{blocks}.pairOrder == stimPair & ...
                            2-main{blocks}.correct == correct);
                    end
                end
                correct2 = [correct2; main{blocks}.correct];
            end
        end
    end    
    
    respPattern_cond1(sub,:,:) = squeeze(sum(resp1,1));
    respPattern_cond2(sub,:,:,:) = squeeze(sum(resp2,1));
    acc(sub,:) = [mean(correct1), mean(correct2)];
end

data.respPattern_cond1 = respPattern_cond1;
data.respPattern_cond2 = respPattern_cond2;
data.acc = acc;

% save(savingName, 'data')
    