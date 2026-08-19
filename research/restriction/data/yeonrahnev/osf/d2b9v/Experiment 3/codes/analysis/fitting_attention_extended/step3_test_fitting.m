% ------------------------------------------------------------------------
% Testing model-fits code for Experimnet 3 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The model fits parameters when people only attend to a specific number
% of items (symbols). The number of items attneded is assigned 
% in 'nItem' variable. The items attended are selected randomly.
%
% The code compares model-fits and uses the best-fitting parameters for 
% model testing. The testing results will be saved in 
% '.../Experiment 3/data/fitting results/attention_extended'.
% To avoid overwriting, the line 153 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Sep.14.2019.
% ------------------------------------------------------------------------

clear, clc

% Select model to test
nItem = 3;  % item attended, 2 or 3
modelToFit = 'attention'; 

% directories
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
savingPath = [fileparts(dataPath) '/fitting results/attention_extended/'];
fittingDataList = dir([savingPath '/fitting_*', '_attend' num2str(nItem) '.mat']);

% Parameters needed for model fitting
N = 100000; % number of testing
nStim = 6;  % number of stimuli

% Load the data
load([dataPath '/dataForModeling']);

% compare AICs and select param_set
for nfit = 1:length(fittingDataList)
    load([savingPath fittingDataList(nfit).name]);        
    for subject = 1:length(modelFit)
        AIC(subject,nfit) = modelFit{subject}.AIC;
    end
end
[~, idx] = max(mean(AIC));
load([savingPath fittingDataList(idx).name]);


%% testing
for subject = 1:length(modelFit)
    data_sub.respPattern_cond1 = squeeze(data.respPattern_cond1(subject,:,:));
    data_sub.respPattern_cond2 = squeeze(data.respPattern_cond2(subject,:,:,:));    
    paramSet = params(:,subject);
    logL = 0;
    
    % Define parameters
    p   = paramSet(1);
    idx = ones(nStim,nStim)-eye(nStim,nStim);
    idx = ind2sub(size(idx),find(idx==1));
    mu = zeros(nStim,nStim);
    mu(idx) = paramSet(2:end);
    
    % Simulate model activations
    for stimPresented=1:nStim
        mu_relevant = mu(:,stimPresented);
        signal(stimPresented,:,1:floor(N*p)) = normrnd(0, 1, nStim, floor(N*p)); %for these trials, all responses are random
        signal(stimPresented,:,floor(N*p)+1:N) = normrnd(repmat(mu_relevant,1,N-floor(N*p)), ...
            ones(nStim,N-floor(N*p)), nStim, N-floor(N*p));
    end
    
    
    for stimPresented=1:nStim
        relevantSignal{stimPresented} = squeeze(signal(stimPresented,:,:));
        
        % relevant response pattern of 2nd choice
        relevantPattern = squeeze(data.respPattern_cond2(subject,stimPresented,:,:));
        
        % select attended item/stimulus
        [~, att_item] = sort(rand(N,nStim),2);
        att_item = att_item(:,1:nItem)';
        
        % select attended siganls and response
        clear attendedSignal
        attendedSignal(1,:) = relevantSignal{stimPresented}(sub2ind(size(relevantSignal{stimPresented}),att_item(1,:),[1:N]));
        attendedSignal(2,:) = relevantSignal{stimPresented}(sub2ind(size(relevantSignal{stimPresented}),att_item(2,:),[1:N]));
        if nItem == 3
            attendedSignal(3,:) = relevantSignal{stimPresented}(sub2ind(size(relevantSignal{stimPresented}),att_item(3,:),[1:N]));
        end
        
        % Compute accuracy for 1st answer
        [~, response] = max(attendedSignal);
        response_cond1{stimPresented} = att_item(sub2ind(size(att_item),response,[1:N]));
        correctTrials = (response_cond1{stimPresented} == stimPresented);
        numCorrect1 = sum(correctTrials); 
        numWrong = N - numCorrect1;
        p_cond1(subject,stimPresented) = numCorrect1 / N;
        
        % Compute accuracy for 2nd answer
        attendedSignal(sub2ind(size(attendedSignal),response,[1:N])) = nan;
        attendedSignal = attendedSignal(:,correctTrials==0);
        att_item = att_item(:,correctTrials==0);
        [~, response] = max(attendedSignal);
        response_cond2{stimPresented} = att_item(sub2ind(size(att_item),response,1:numWrong));
        
        modelPattern = zeros(nStim,nStim);
        choices = [response_cond1{stimPresented}(response_cond1{stimPresented}~=stimPresented)', response_cond2{stimPresented}'];
        pattern = sub2ind(size(modelPattern),choices(:,1),choices(:,2));
        for idx = 1:length(pattern)
            modelPattern(pattern(idx)) = modelPattern(pattern(idx)) + 1;
        end
        
        numCorrect2 = sum(response_cond2{stimPresented} == stimPresented);
        p_cond2(subject,stimPresented) = numCorrect2 / numWrong;
        
        %%%%%%% logL %%%%%%
        idx = ones(nStim,nStim) - eye(nStim,nStim);
        idx(stimPresented,:) = 0;
        idx = ind2sub(size(idx),find(idx==1));
        
        modelPattern = modelPattern ./ repmat(sum(modelPattern,2),[1,nStim]);
        modelPattern(stimPresented,:) = 0;
        input_modelPattern = modelPattern(idx);
        input_modelPattern(input_modelPattern == 0) = 10^-10;
        log_modelPattern = log(input_modelPattern);
        
        gain = log_modelPattern .* relevantPattern(idx);
        logL = logL - sum(gain);       
    end
    %%%%%% accuracy(unbalanced) %%%%%%
    accuracy_cond1(subject,1) = (p_cond1(subject,:)*sum(data_sub.respPattern_cond1,2))/sum(sum(data_sub.respPattern_cond1));
    accuracy_cond2(subject,1) = (p_cond2(subject,:)*sum(sum(data_sub.respPattern_cond2,3),2))/sum(sum(sum(data_sub.respPattern_cond2)));
   
    % organize data to save
    logL = -logL;
    k    = length(paramSet)-1;
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
observation = round([mean(data.acc(:,1)); mean(data.acc(:,2))],3);
model = round([mean(accuracy_cond1); mean(accuracy_cond2)],3);
display_accuracy = table(observation, model, 'RowNames', {'1st answer'; '2nd answer'})

%% Save data
% save([savingPath '/' modelToFit '_' num2str(nItem)], 'resfit', 'accuracy_cond1', 'accuracy_cond2')