% ------------------------------------------------------------------------
% Testing model-fits code for Experimnet 2 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The model fits parameters when people only attend to a specific number
% of items (symbols). The number of items attneded is assigned 
% in 'nItem' variable. The items attended are selected randomly.
%
% The code compares model-fits and uses the best-fitting parameters for 
% model testing. The testing results will be saved in 
% '.../Experiment 2/data/fitting results/attention_extended'.
% To avoid overwritting the line 155 is currently commented. 
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
    
    % Compute accuracy for condition 1
    for stimPresented=1:nStim
        relevantSignal{stimPresented} = squeeze(signal(stimPresented,:,:)); %all non-presented conditions
        
        % select attended item/stimulus
        [~, att_item{stimPresented}] = sort(rand(N,nStim),2);
        att_item{stimPresented} = att_item{stimPresented}(:,1:nItem)';
        
        % select attended siganls and response
        attendedSignal{stimPresented}(1,:) = relevantSignal{stimPresented}(sub2ind(size(relevantSignal{stimPresented}),att_item{stimPresented}(1,:),[1:N]));
        attendedSignal{stimPresented}(2,:) = relevantSignal{stimPresented}(sub2ind(size(relevantSignal{stimPresented}),att_item{stimPresented}(2,:),[1:N]));
        if nItem == 3
            attendedSignal{stimPresented}(3,:) = relevantSignal{stimPresented}(sub2ind(size(relevantSignal{stimPresented}),att_item{stimPresented}(3,:),[1:N]));
        end
        [~, response] = max(attendedSignal{stimPresented});
        response_cond1{stimPresented} = att_item{stimPresented}(sub2ind(size(att_item{stimPresented}),response,[1:N]));
        p_cond1(subject,stimPresented) = sum(response_cond1{stimPresented} == stimPresented) / N;
    end
    accuracy_cond1(subject,1) = mean(p_cond1(subject,:));
    
    % Compute accuracy for condition 2
    for stimPresented=1:nStim
        for stimPair=1:nStim
            if stimPresented~=stimPair %the pair needs to contain two different colors
                
                % When both stimPresented and stimPair are attended,
                % directly compare the activities of the signal
                numCorrect = sum(sum(att_item{stimPresented} == stimPresented | att_item{stimPresented} == stimPair)==2 & ...
                    relevantSignal{stimPresented}(stimPresented,:) > relevantSignal{stimPresented}(stimPair,:));
                numWrong = sum(sum(att_item{stimPresented} == stimPresented | att_item{stimPresented} == stimPair)==2 & ...
                    relevantSignal{stimPresented}(stimPresented,:) < relevantSignal{stimPresented}(stimPair,:));
                
                % Case when stimPresented is one of attended items, but
                % stimPair is not attended: Correct
                check_attended_items = cat(3,(att_item{stimPresented} == stimPresented), -(att_item{stimPresented} == stimPair));
                check_attended_items = sum(sum(check_attended_items,3),1);
                numCorrect = numCorrect + sum(check_attended_items==1);
                 
                % When the stimPair get attention, but not stimPresented,
                % then Wrong
                numWrong = numWrong + sum(check_attended_items==-1);
                  
                % When any attended items are not given as options, then
                % choose anything
                numCorrect = numCorrect + (N-(numCorrect+numWrong))/2;
                numWrong = numWrong + (N-(numCorrect+numWrong));
                
                respProbability = [numCorrect, numWrong] / N;
                                
                % compute logL and accuracy
                for accuracy=1:2
                    if respProbability(accuracy)==0; respProbability(accuracy)=10^-10; end %replace probability of 0, in order to avoid -inf when taking log
                    logL = logL - log(respProbability(accuracy)) * data_sub.respPattern_cond2(stimPresented,stimPair,accuracy);
                end
                p_cond2(subject,stimPresented,stimPair) = respProbability(1);               
            end
        end
    end
   
    % organize data to save
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

    % compute accuracy_cond2
    idx = (ones(nStim,nStim) - eye(nStim,nStim));
    idx = ind2sub(size(idx),find(idx==1));
    p_input = squeeze(p_cond2(subject,:,:));    
    accuracy_cond2(subject,:) = mean(p_input(idx));
    
end


%% Display results
observation = round([mean(data.acc(:,1)); mean(data.acc(:,2))],3);
model = round([mean(accuracy_cond1); mean(accuracy_cond2)],3);
display_accuracy = table(observation, model, 'RowNames', {'6-alternative'; '2-alternative'})

%% Save data
% save([savingPath '/' modelToFit '_' num2str(nItem)], 'resfit', 'accuracy_cond1', 'accuracy_cond2')