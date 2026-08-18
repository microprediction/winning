% ------------------------------------------------------------------------
% Testing model-fits code for Experimnet 2 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code compares model-fits and uses the best-fitting parameters for 
% model testing. The testing results will be saved in 
% '.../Experiment 1/data/fitting results/simple'.
% To avoid overwriting, the line 148 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------
clear, clc

% Select model to test
modelToFit = 'twohighest';  %'population', 'summary', 'twohighest', 'threehighest'

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



%% Testing
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
     
    %% Compute accuracy for condition 1   
    % p_cond1
    for stimPresented=1:nStim
        relevantSignal{stimPresented} = squeeze(signal(:,1,:)); %all non-presented conditions
        relevantSignal{stimPresented}(stimPresented,:) = signal(stimPresented,2,:); %use presented condition for stimPresented
        [~, response_cond1{stimPresented}] = max(relevantSignal{stimPresented});        
        p = sum(response_cond1{stimPresented} == stimPresented) / N;
        p_cond1(subject, stimPresented) = p;
    end
    
    % accuracy
    presented_N = sum(data_sub.respPattern_cond1,2);
    total_N = sum(presented_N);
    accuracy_cond1(subject,1) = p_cond1(subject,:)*presented_N / total_N;
            
    %% Compute accuracy for condition 2
    for stimPresented=1:nStim
        for stimPair=1:nStim
            if stimPresented~=stimPair %the pair needs to contain two different colors
                
                % p_cond2 depending on the model
                if strcmp(modelToFit, 'population')
                    p = sum(signal(stimPresented,2,:) >= signal(stimPair,1,:)) / N;
                    
                else 
                    if strcmp(modelToFit, 'summary')
                        numCorrect_direct = sum(response_cond1{stimPresented}==stimPresented);
                        numWrong_direct = sum(response_cond1{stimPresented}==stimPair);
                                                
                    elseif strcmp(modelToFit, 'twohighest')
                        [~, signal_ranking] = sort(relevantSignal{stimPresented},1,'descend');
                        [~, signal_ranking] = sort(signal_ranking,1,'ascend');
                        numCorrect_direct = sum(signal_ranking(stimPresented,:)<signal_ranking(stimPair,:) & signal_ranking(stimPresented,:)<3);
                        numWrong_direct = sum(signal_ranking(stimPresented,:)>signal_ranking(stimPair,:) & signal_ranking(stimPair,:)<3);
                      
                    elseif strcmp(modelToFit, 'threehighest') 
                        [~, signal_ranking] = sort(relevantSignal{stimPresented},1,'descend');                        
                        [~, signal_ranking] = sort(signal_ranking,1,'ascend');
                        numCorrect_direct = sum(signal_ranking(stimPresented,:)<signal_ranking(stimPair,:) & signal_ranking(stimPresented,:)<4);
                        numWrong_direct = sum(signal_ranking(stimPresented,:)>signal_ranking(stimPair,:) & signal_ranking(stimPair,:)<4);
                      
                    end
                    numCorrect = numCorrect_direct + (N - numCorrect_direct - numWrong_direct)/2;
                    numWrong = numWrong_direct + (N - numCorrect_direct - numWrong_direct)/2;                
                    p = numCorrect / (numCorrect + numWrong);
                end
                
                % logL
                p_cond2(subject,stimPresented,stimPair) = p;
                respProbability = [p, 1-p];
                for accuracy=1:2
                    if respProbability(accuracy)==0; respProbability(accuracy)=10^-10; end %replace probability of 0, in order to avoid -inf when taking log
                    logL = logL - log(respProbability(accuracy)) * data_sub.respPattern_cond2(stimPresented,stimPair,accuracy);
                end                
            end
        end
    end        
    
    % accuracy
    p_sub = squeeze(p_cond2(subject,:,:)); %(stimPresented, stimPair)
    presented_N = sum(data_sub.respPattern_cond2,3); %(stimPresented, stimPair)
    total_N = sum(sum(presented_N));
    p_sub = sum(diag(p_sub * presented_N'))/total_N; %Transpose presented_N to multiply correct stimPresented trials
    accuracy_cond2(subject,1) = p_sub;
    
    %% resfit   
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
display_accuracy = table(observed, model, 'RowNames', {'6-alternative'; '2-alternative'})

%% Save data
% save([savingPath '/' modelToFit '_extended'], 'resfit', 'accuracy_cond1', 'accuracy_cond2')