% ------------------------------------------------------------------------
% Testing model-fits code for Experimnet 1 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code compares model-fits and uses the best-fitting parameters for 
% model testing. The testing results will be saved in 
% '.../Experiment 1/data/fitting results/modeling simple'.
% To avoid overwriting the line 134 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------

clear, clc

% Select model to test
modelToFit = 'population'; %'population', 'summary', 'two_highest'

% directories
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
savingPath = [fileparts(dataPath) '/fitting results/extended/'];
fitDataList = dir([savingPath '/fit*.mat']);

% Parameters needed for model fitting
N = 100000; % number of testing
nStim = 4;  % number of stimuli

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
    for stim=1:4
        for presented=0:1
            signal(stim,presented+1,1:floor(N*p)) = normrnd(0, 1, 1, floor(N*p)); %for these trials, all responses are random
            signal(stim,presented+1,floor(N*p)+1:N) = normrnd(mu(stim+4*presented), 1, 1, N-floor(N*p));
        end
    end
    
    % Compute accuracy for condition 1
    for stimPresented=1:nStim
        relevantSignal{stimPresented} = squeeze(signal(:,1,:)); %all non-presented conditions
        relevantSignal{stimPresented}(stimPresented,:) = signal(stimPresented,2,:); %use presented condition for stimPresented
        [~, response_cond1{stimPresented}] = max(relevantSignal{stimPresented});
        p_cond1(subject,stimPresented) = sum(response_cond1{stimPresented} == stimPresented) / N;
    end
    accuracy_cond1(subject,1) = mean(p_cond1(subject,:));
    
    % Compute accuracy for condition 2
    for stimPresented=1:nStim
        for stimPair=1:nStim
            if stimPresented~=stimPair %the pair needs to contain two different colors
                
                % get probability according to the model
                if strcmp(modelToFit, 'population')                    
                    p = sum(relevantSignal{stimPresented}(stimPresented,:) >= relevantSignal{stimPresented}(stimPair,:)) / N;
                    respProbability = [p, 1-p];                    
                     
                else    
                    if  strcmp(modelToFit, 'summary')
                        numCorrect_direct = sum(response_cond1{stimPresented}==stimPresented);
                        numWrong_direct = sum(response_cond1{stimPresented}==stimPair);   
                       
                    elseif strcmp(modelToFit, 'twohighest')  % 2-highest activities
                        [~, signal_ranking] = sort(relevantSignal{stimPresented},1,'descend');
                        [~, signal_ranking] = sort(signal_ranking,1,'ascend');
                        numCorrect_direct = sum(signal_ranking(stimPresented,:)<signal_ranking(stimPair,:) & signal_ranking(stimPresented,:)<3);
                        numWrong_direct = sum(signal_ranking(stimPresented,:)>signal_ranking(stimPair,:) & signal_ranking(stimPair,:)<3);
                                                               
                    end
                    numCorrect = numCorrect_direct + (N-numCorrect_direct-numWrong_direct)/2;
                    numWrong = numWrong_direct + (N-numCorrect_direct-numWrong_direct)/2;
                    
                    respProbability = [numCorrect, numWrong] / (numCorrect + numWrong);
                end
                
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
observation = round([mean(data.accuracy(:,1)); mean(data.accuracy(:,2))],3);
model = round([mean(accuracy_cond1); mean(accuracy_cond2)],3);
display_accuracy = table(observation, model, 'RowNames', {'4-alternative'; '2-alternative'})

%% Save data
% save([savingPath '/' modelToFit '_simple'], 'resfit', 'accuracy_cond1', 'accuracy_cond2')