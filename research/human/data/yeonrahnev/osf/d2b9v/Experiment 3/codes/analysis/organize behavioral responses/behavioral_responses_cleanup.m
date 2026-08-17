% ------------------------------------------------------------------------
% Data clean up code for Experiment 3 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code imports subjects' behavioral responses and organize data.
% In the end, the code will save 'dataForModeling.mat' file in 
% '.../Experiment 3/data/subject_responses'. To avoid overwrite on the
% existing file, the line 56 is currently commented. To run this code, 
% locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------
clear all, close all, clc

dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses/raw responses'];
savingName = [fileparts(dataPath) '/dataForModeling'];
fileList = dir([dataPath, '/sub*']);

nSub = 10;
sessions = 1:3;
nStim = 6;

for sub = 1:nSub
    target = []; correct = []; response = [];
    for ss = sessions        
        filename = ['sub' num2str(sub) '_' num2str(ss) '.mat'];
        load([dataPath '/' filename])        
        for blocks = 1:length(p.main)
            target = [target; p.main{blocks}.targetOrder'];
            correct = [correct; p.main{blocks}.correct];
            response = [response; p.main{blocks}.response];
        end
    end
    
    for stimPresented = 1:nStim
        for Resp1 = 1:nStim
            resp1(stimPresented,Resp1) = sum(target == stimPresented & response(:,1) == Resp1);
            if stimPresented ~= Resp1 
                for Resp2 = 1:nStim
                    resp2(stimPresented,Resp1,Resp2) = sum(target == stimPresented & response(:,1) == Resp1 & ...
                        response(:,2) == Resp2);
                end
            end
        end
    end
    
    respPattern_cond1(sub,:,:) = resp1;
    respPattern_cond2(sub,:,:,:) = resp2;
    acc(sub,:) = [mean(correct(:,1)), mean(correct(correct(:,2)~=99,2))];
end

data.respPattern_cond1 = respPattern_cond1;
data.respPattern_cond2 = respPattern_cond2;
data.acc = acc;

% save(savingName, 'data')
    