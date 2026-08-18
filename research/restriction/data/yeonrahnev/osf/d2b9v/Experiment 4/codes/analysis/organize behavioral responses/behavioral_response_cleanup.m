% ------------------------------------------------------------------------
% Data clean up code for Experiment 4 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code imports subjects' behavioral responses and organize data.
% In the end, the code will save 'dataForModeling.mat' file in 
% '.../Experiment 4/data/subject_responses'. To avoid overwrite on the
% existing file, the line 77 is currently commented. To run this code, 
% locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Sep.13.2019.
% ------------------------------------------------------------------------

clear all, close all, clc

dataDir = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses/raw responses'];
savingName = [dataDir '/dataForModeling'];
subjects = 1:11;

for sub = 1:length(subjects)
    subname = ['sub' num2str(subjects(sub))];
    files = dir([dataDir '/' subname '_*.mat']);
    
    correct_2c = []; correct_3c = [];
    resp_2c = []; resp_3c = []; 
    correct_answer_2c = []; correct_answer_3c = [];
    wrong_answer_2c = NaN(1,3000);
    rt_2c = []; rt_3c = [];
    
    for sess = 1:length(files)
        load([dataDir '/' files(sess).name])
        
        for run = 1:5
            if mod(run,2) == 1, rowInd = 1; else rowInd = 2; end
            
            for block = 1:4                
                choice_limit = p.limitChoices(rowInd,block);
                
                switch choice_limit
                    case 0  % 3-choice
                        correct_3c = [correct_3c; squeeze(p.correct(run,block,:))];
                        resp_3c = [resp_3c; squeeze(p.responses(run,block,:))];
                        correct_answer_3c = [correct_answer_3c; squeeze(p.correctAnswer(run,block,:))];
                        rt_3c = [rt_3c; squeeze(p.response_time(run,block,:))];
                    case 1 % 2-choice
                        correct_2c = [correct_2c; squeeze(p.correct(run,block,:))];
                        resp_2c = [resp_2c; squeeze(p.responses(run,block,:))];
                        correct_answer_2c = [correct_answer_2c; squeeze(p.correctAnswer(run,block,:))];
                        if isfield(p,'choices')
                            idx = ((sess-1)*20+(run-1)*4+block-1)*50+1:((sess-1)*20+(run-1)*4+block)*50;
                            wrong_answer_2c(idx) = squeeze(p.choices(run,block,:,2));
                        end
                        rt_2c = [rt_2c; squeeze(p.response_time(run,block,:))];
                end
            end
        end
        
        if sess == 1
            data.proportion_used(sub) = p.proportions(1);
        end
    end
 

    %% save subject's data    
    data.c3.correct(:,sub) = correct_3c;
    data.c3.correct_answer(:,sub) = correct_answer_3c;
    data.c3.resp(:,sub) = resp_3c;
    data.c3.rt(:,sub) = rt_3c;
        
    data.c2.correct(:,sub) = correct_2c;
    data.c2.correct_answer(:,sub) = correct_answer_2c;
    data.c2.wrong_answer(:,sub) = wrong_answer_2c;
    data.c2.resp(:,sub) = resp_2c;
    data.c2.rt(:,sub) = rt_2c;
end

% save(savingName,'data')