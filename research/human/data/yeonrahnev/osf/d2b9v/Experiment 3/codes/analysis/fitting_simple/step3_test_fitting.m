% ------------------------------------------------------------------------
% Testing model-fits code for Experimnet 3 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code compares model-fits and uses the best-fitting parameters for 
% model testing. The testing results will be saved in 
% '.../Experiment 3/data/fitting results/simple'.
% To avoid overwriting, the line 157 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------

clear, clc

% Select model to test
modelToFit = 'summary+strategic';  %'population', 'summary+random', 'summary+strategic'

% directories
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
savingPath = [fileparts(dataPath) '/fitting results/simple/'];
fitDataList = dir([savingPath '/fit*.mat']);

% Parameters needed for model fitting
N = 100000; % number of testing
nStim = 6;  % number of stimuli

% Load the data
load([dataPath '/dataForModeling']);

% compare AICs and select param_set
for nfit = 1:length(fitDataList)
    load([savingPath fitDataList(nfit).name]);    
    for subject = 1:length(modelFit)
        AIC(subject,nfit) = modelFit{subject}.AIC;
    end
end
[~, idx] = max(mean(AIC));
load([savingPath fitDataList(idx).name]);


%% testing
for subject = 1:length(modelFit)
    
    data_sub.respPattern_cond1 = squeeze(data.respPattern_cond1(subject,:,:));
    data_sub.respPattern_cond2 = squeeze(data.respPattern_cond2(subject,:,:,:));
    paramSet = params(:,subject);
    logL = 0;
    
    % Define parameters
    p   = paramSet(1);
    mu  = [0, paramSet(2:end)']; %assume that mu for stimulus 1, when not dominant is 0
    
    % Simulate model activations
    for stim=1:nStim
        for presented=0:1
            signal(stim,presented+1,1:floor(N*p)) = normrnd(0, 1, 1, floor(N*p)); %for these trials, all responses are random
            signal(stim,presented+1,floor(N*p)+1:N) = normrnd(mu(stim+nStim*presented), 1, 1, N-floor(N*p));
        end
    end
    
    % simulation
    for stimPresented=1:nStim
        relevantSignal = squeeze(signal(:,1,:)); %all non-presented conditions
        relevantSignal(stimPresented,:) = signal(stimPresented,2,:); %use presented condition for stimPresented
        
        % relevant response pattern for 2nd choice
        relevantPattern = squeeze(data.respPattern_cond2(subject,stimPresented,:,:)); % relevantPattern(1stchoice,2ndchoice)
        
        %%%%%% find first choice %%%%%%
        [maxval, firstchoice] = max(relevantSignal);
        numCorrect1 = sum(firstchoice == stimPresented);
        numWrong = sum(firstchoice ~= stimPresented);
        p = numCorrect1 / N;
        p_cond1(subject,stimPresented) = p;
        
        %%%%%% find second choice %%%%%%
        if strcmp(modelToFit, 'population')
            maxval = repmat(maxval,[6,1]);
            relevantSignal(relevantSignal == maxval) = NaN;
            [~, secondchoice] = max(relevantSignal);
            secondchoice(firstchoice == stimPresented) = [];
            numCorrect2 = sum(secondchoice == stimPresented);
            
            % create pattern response from the model
            choices = [firstchoice(firstchoice~=stimPresented)', secondchoice'];
            modelPattern = zeros(nStim,nStim);
            idx = sub2ind(size(modelPattern),choices(:,1),choices(:,2));
            u_idx = unique(idx);
            clear count_pattern
            for patterns = 1:length(u_idx)
                count_pattern(patterns) = length(find(idx == u_idx(patterns)));
            end
            modelPattern(u_idx) = count_pattern;
            modelPattern = modelPattern ./ repmat(sum(modelPattern,2), [1,nStim]);
            modelPattern(stimPresented,:) = 0;
            
        elseif strcmp(modelToFit, 'summary+random')
            % completely random
            numCorrect2 = numWrong/(nStim-1);
            modelPattern = (ones(nStim,nStim)-eye(nStim,nStim))*numCorrect2;
            modelPattern = modelPattern ./ (repmat(sum(modelPattern,2),[1,nStim]));
            modelPattern(stimPresented,:) = 0;
            
        elseif strcmp(modelToFit, 'summary+strategic')
            % 33.3% chance of correct for 2nd choice - according to the frequency of each stim            
            freq_nontarget = 7/(49-7);
            freq_target = 14/(49-7);
            modelPattern = (ones(nStim,nStim)-eye(nStim,nStim))*freq_nontarget;
            modelPattern(:,stimPresented) = freq_target;
            modelPattern(stimPresented,:) = 0;
            numCorrect2 = numWrong*freq_target;
        
        end
        p = numCorrect2 / numWrong;
        p_cond2(subject,stimPresented) = p;
        
        %%%%%%% logL %%%%%%
        idx = ones(nStim,nStim) - eye(nStim,nStim);
        idx(stimPresented,:) = 0;
        idx = ind2sub(size(idx),find(idx==1));
        
        input_modelPattern = modelPattern(idx);
        input_modelPattern(input_modelPattern == 0) = 10^-10;
        log_modelPattern = log(input_modelPattern);
        
        gain = log_modelPattern .* relevantPattern(idx);
        logL = logL - sum(gain);
    end
    
    %%%%%% accuracy(unbalanced) %%%%%%
    accuracy_cond1(subject,1) = (p_cond1(subject,:)*sum(data_sub.respPattern_cond1,2))/sum(sum(data_sub.respPattern_cond1));
    accuracy_cond2(subject,1) = (p_cond2(subject,:)*sum(sum(data_sub.respPattern_cond2,3),2))/sum(sum(sum(data_sub.respPattern_cond2)));
    
    % wrap up
    logL = -logL;
    k    = length(paramSet);
    n    = sum(sum(data_sub.respPattern_cond1)) + sum(sum(sum(data_sub.respPattern_cond2)));
    
    resfit{subject}.modelName = modelToFit;
    resfit{subject}.logL = logL;
    resfit{subject}.k    = k;
    resfit{subject}.n    = n;
    resfit{subject}.AIC  = -2*logL + 2*k;
    resfit{subject}.AICc = -2*logL + (2*k*n)/(n-k-1);
    resfit{subject}.BIC  = -2*logL + k*log(n);
end


%% Display results
observed = round([mean(data.acc(:,1)); mean(data.acc(:,2))],3);
model = round([mean(accuracy_cond1); mean(accuracy_cond2)],3);
display_accuracy = table(observed, model, 'RowNames', {'1st answer'; '2nd answer'})


%% Save data
% save([savingPath '/' modelToFit '_simple'], 'resfit', 'accuracy_cond1', 'accuracy_cond2')