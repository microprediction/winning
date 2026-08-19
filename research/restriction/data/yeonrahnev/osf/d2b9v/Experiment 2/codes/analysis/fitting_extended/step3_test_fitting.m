% ------------------------------------------------------------------------
% Testing model-fits code for Experimnet 2 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code compares model-fits and uses the best-fitting parameters for 
% model testing. The testing results will be saved in 
% '.../Experiment 2/data/fitting results/extended'.
% To avoid overwriting, the line 148 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------

clear, clc

% Select model to test
modelToFit = 'population';  %'population', 'summary', 'twohighest', 'threehighest'

% directories
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
savingPath = [fileparts(dataPath) '/fitting results/extended/'];
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
        relevantSignal = squeeze(signal(stimPresented,:,:)); %all non-presented conditions
        [~, response_cond1{stimPresented}] = max(relevantSignal);
        p_cond1(subject,stimPresented) = sum(response_cond1{stimPresented} == stimPresented) / N;
    end
    accuracy_cond1(subject,1) = mean(p_cond1(subject,:));
    
    % Compute accuracy for condition 2
    for stimPresented=1:nStim
        for stimPair=1:nStim
            if stimPresented~=stimPair %the pair needs to contain two different colors
                
                % get probability according to the model
                if strcmp(modelToFit, 'population')                    
                    p = sum(signal(stimPresented,stimPresented,:) >= signal(stimPresented,stimPair,:)) / N;                                  
                     
                else
                    if  strcmp(modelToFit, 'summary')
                        numCorrect = sum(response_cond1{stimPresented}==stimPresented);
                        numWrong = sum(response_cond1{stimPresented}==stimPair);
                        
                    elseif strcmp(modelToFit, 'twohighest')  % 2-highest activities
                        [~, signal_ranking] = sort(squeeze(signal(stimPresented,:,:)),1,'descend');                        
                        [~, signal_ranking] = sort(signal_ranking,1,'ascend');
                                                
                        numCorrect = sum(signal_ranking(stimPresented,:)==1);
                        numCorrect = numCorrect + sum(signal_ranking(stimPresented,:)==2 & signal_ranking(stimPair,:)~=1);
                        
                        numWrong = sum(signal_ranking(stimPair,:)==1);
                        numWrong = numWrong + sum(signal_ranking(stimPair,:)==2 & signal_ranking(stimPresented,:)~=1);
                        
                    elseif strcmp(modelToFit, 'threehighest')
                        [~, signal_ranking] = sort(squeeze(signal(stimPresented,:,:)),1,'descend');            
                        [~, signal_ranking] = sort(signal_ranking,1,'ascend');
                        
                        numCorrect = sum(signal_ranking(stimPresented,:)<4 & (signal_ranking(stimPresented,:)<signal_ranking(stimPair,:)));
                        numWrong = sum(signal_ranking(stimPair,:)<4 & signal_ranking(stimPair,:)<signal_ranking(stimPresented,:));
                      
                    end
                    p = (numCorrect + (N-numCorrect-numWrong)/2) / N;
                    
                end
                respProbability = [p, 1-p];     
                                
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
observed = round([mean(data.acc(:,1)); mean(data.acc(:,2))],3);
model = round([mean(accuracy_cond1); mean(accuracy_cond2)],3);
display_accuracy = table(observed, model, 'RowNames', {'6-alternative'; '2-alternative'})


%% Save data
% save([savingPath '/' modelToFit '_extended'], 'resfit', 'accuracy_cond1', 'accuracy_cond2')